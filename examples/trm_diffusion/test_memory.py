import pytest
import torch
import torch.nn as nn
import gc
from diffusers import UNet2DModel

from trm_models import StandardTRM, StandardTRMv2, bypass_projections

# ---------------------------------------------------------------------------
# 1. MEMORY LEAK TESTS (BPTT Graph Accumulation)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Memory leak tests require CUDA")
def test_standard_trm_memory_leak():
    """
    Simulates the N_supervision loop and strictly asserts that
    CUDA memory does not grow across iterations using StandardTRM.
    """
    core_unet = UNet2DModel(
        sample_size=32,
        in_channels=9,  # x(3) + y(3) + z(3)
        out_channels=6,  # y(3) + z(3)
        block_out_channels=(64, 128),
        layers_per_block=1,
        down_block_types=("AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D"),
    )

    model = StandardTRM(core_unet, state_channels=3, resolution=32, n=2, T=2).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    bsz, c, h, w = 2, 3, 32, 32
    x = torch.randn(bsz, c, h, w, device="cuda")
    timesteps = torch.randint(0, 1000, (bsz,), device="cuda")
    y, z = model.get_initial_states(bsz)
    y, z = y.to("cuda"), z.to("cuda")

    # Warmup
    for _ in range(3):
        model_output, y, z = model.reasoning_step(x, y, z, timesteps)
        loss = model_output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_baseline = torch.cuda.memory_allocated()

    # Test Loop
    for _ in range(15):
        model_output, y, z = model.reasoning_step(x, y, z, timesteps)
        loss = model_output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    gc.collect()
    torch.cuda.empty_cache()
    mem_final = torch.cuda.memory_allocated()
    memory_growth_mb = (mem_final - mem_baseline) / (1024 * 1024)

    assert memory_growth_mb <= 1.0, f"StandardTRM MEMORY LEAK: Grew by {memory_growth_mb:.2f} MB"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Memory leak tests require CUDA")
def test_high_dim_trm_memory_leak():
    """
    Asserts memory stability for HighDimensionalTRM.
    Crucially, the base UNet here has standard in/out channels!
    """
    core_unet = UNet2DModel(
        sample_size=32,
        in_channels=3,  # Standard input
        out_channels=3,  # Standard output
        block_out_channels=(64, 128),
        layers_per_block=1,
        down_block_types=("AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D"),
    )

    model = StandardTRMv2(core_unet, resolution=32, n=2, T=2).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    bsz, c, h, w = 2, 3, 32, 32
    x = torch.randn(bsz, c, h, w, device="cuda")
    timesteps = torch.randint(0, 1000, (bsz,), device="cuda")

    # y and z are fetched in high-D space, so we just use what the model gives us
    y, z = model.get_initial_states(bsz)
    y, z = y.to("cuda"), z.to("cuda")

    # Warmup
    for _ in range(3):
        model_output, y, z = model.reasoning_step(x, y, z, timesteps)
        loss = model_output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_baseline = torch.cuda.memory_allocated()

    # Test Loop
    for _ in range(15):
        model_output, y, z = model.reasoning_step(x, y, z, timesteps)
        loss = model_output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    gc.collect()
    torch.cuda.empty_cache()
    mem_final = torch.cuda.memory_allocated()
    memory_growth_mb = (mem_final - mem_baseline) / (1024 * 1024)

    assert memory_growth_mb <= 1.0, f"HighDimensionalTRM MEMORY LEAK: Grew by {memory_growth_mb:.2f} MB"


# ---------------------------------------------------------------------------
# 2. STRUCTURAL & SAFETY TESTS
# ---------------------------------------------------------------------------


def test_high_dim_trm_projection_restoration():
    """
    Ensures the context manager strictly restores the original layers,
    even if an exception occurs mid-loop, preventing corrupted checkpoint saves.
    """
    core_unet = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=(32, 32),
        layers_per_block=1,
        down_block_types=("AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D"),
    )
    model = StandardTRMv2(core_unet, resolution=32, n=1, T=1)

    original_conv_in = model.core_model.conv_in
    original_conv_out = model.core_model.conv_out

    # 1. Test normal execution restoration
    bsz, c, h, w = 2, 3, 32, 32
    x = torch.randn(bsz, c, h, w)
    timesteps = torch.randint(0, 1000, (bsz,))
    y, z = model.get_initial_states(bsz)

    model.reasoning_step(x, y, z, timesteps)

    assert model.core_model.conv_in is original_conv_in, "conv_in was not restored!"
    assert model.core_model.conv_out is original_conv_out, "conv_out was not restored!"
    assert not isinstance(model.core_model.conv_in, nn.Identity), "conv_in stuck as Identity!"

    # 2. Test exception handling restoration
    try:
        with bypass_projections(model.core_model, "unet"):
            assert isinstance(model.core_model.conv_in, nn.Identity)
            raise RuntimeError("Simulated Forward Pass Crash")
    except RuntimeError:
        pass

    assert model.core_model.conv_in is original_conv_in, "conv_in not restored after crash!"


def test_trm_gradient_detachment():
    """
    Strictly asserts that y_next and z_next are detached from the compute graph.
    If this fails, BPTT will attempt to backpropagate through all previous steps.
    """
    core_unet = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=(32, 32),
        layers_per_block=1,
        down_block_types=("AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D"),
    )

    # Test HighDim TRM
    model = StandardTRMv2(core_unet, resolution=32, n=2, T=2)
    bsz, c, h, w = 2, 3, 32, 32
    x = torch.randn(bsz, c, h, w, requires_grad=True)
    timesteps = torch.randint(0, 1000, (bsz,))
    y, z = model.get_initial_states(bsz)

    model_output, y_next, z_next = model.reasoning_step(x, y, z, timesteps)

    # Output MUST have a gradient graph for the loss function
    assert model_output.grad_fn is not None, "Loss output is missing gradient history!"

    # States for the next step MUST NOT have a gradient graph
    assert y_next.grad_fn is None, "y_next is attached to the graph (Memory Leak Trap)!"
    assert z_next.grad_fn is None, "z_next is attached to the graph (Memory Leak Trap)!"


def test_trm_output_shapes():
    """
    Verifies that the forward pass and reasoning_step cleanly output
    the standard image dimensions, despite operating in high-D space internally.
    """
    core_unet = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=(64, 64),  # Keeping this slightly wider as originally written
        layers_per_block=1,
        down_block_types=("AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D"),
    )
    model = StandardTRMv2(core_unet, resolution=32, n=1, T=1)

    bsz, c, h, w = 2, 3, 32, 32
    x = torch.randn(bsz, c, h, w)
    timesteps = torch.randint(0, 1000, (bsz,))
    y, z = model.get_initial_states(bsz)

    # 1. Check reasoning step shapes
    model_output, y_next, z_next = model.reasoning_step(x, y, z, timesteps)

    assert model_output.shape == (bsz, c, h, w), "Loss output shape is incorrect!"
    assert y_next.shape == (bsz, 64, h, w), "y_next shape does not match high-D space!"
    assert z_next.shape == (bsz, 64, h, w), "z_next shape does not match high-D space!"

    # 2. Check standard inference pipeline forward pass
    eval_output = model(x, timesteps)
    assert eval_output.sample.shape == (bsz, c, h, w), "Inference forward pass shape is incorrect!"
