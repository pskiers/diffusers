import pytest
import torch
import gc
from diffusers import UNet2DModel
from trm_utils import deep_recursion


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Memory leak tests require CUDA")
def test_deep_recursion_memory_leak():
    """
    Simulates the N_supervision loop and strictly asserts that
    CUDA memory does not grow across iterations.
    """
    # 1. Setup a dummy unconditional small-loop model
    model = UNet2DModel(
        sample_size=32,
        in_channels=9,   # x(3) + y(3) + z(3)
        out_channels=6,  # y(3) + z(3)
        block_out_channels=(64, 128),
        layers_per_block=1,
        down_block_types=("AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D"),
    ).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 2. Setup dummy inputs
    bsz, c, h, w = 2, 3, 32, 32
    x = torch.randn(bsz, c, h, w, device="cuda")
    y = torch.randn(bsz, c, h, w, device="cuda")
    z = torch.randn(bsz, c, h, w, device="cuda")
    timesteps = torch.randint(0, 1000, (bsz,), device="cuda")

    # 3. Warmup (PyTorch allocator takes a few steps to settle)
    for _ in range(3):
        model_output, y, z = deep_recursion(model, x, y, z, timesteps, n=2, T=2)
        loss = model_output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Force garbage collection and get baseline memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_baseline = torch.cuda.memory_allocated()

    # 4. Test Loop (Simulating N_supervision = 15)
    for _ in range(15):
        model_output, y, z = deep_recursion(model, x, y, z, timesteps, n=2, T=2)
        loss = model_output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 5. Measure Final Memory
    gc.collect()
    torch.cuda.empty_cache()
    mem_final = torch.cuda.memory_allocated()

    # Allow 1MB of variance (allocator jitter), but generally it should be identical.
    memory_growth_mb = (mem_final - mem_baseline) / (1024 * 1024)

    assert memory_growth_mb <= 1.0, f"MEMORY LEAK DETECTED: Grew by {memory_growth_mb:.2f} MB during recursion loop!"
