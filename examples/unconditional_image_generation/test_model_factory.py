import pytest
from config import parse_args
from model_factory import build_model
from diffusers import UNet2DModel
from conditional_unet_model import UnifiedConditionUNet

# ---------------------------------------------------------
# 1. Unconditional Models
# ---------------------------------------------------------

def test_model_factory_unconditional_standard(monkeypatch):
    """Test standard unconditional model creation."""
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--dataset_name", "uoft-cs/cifar100",
        "--num_classes", "0",
        "--input_channels", "3",
        "--model_type", "unet2d",
        "--dataset_type", "hf"
    ])
    args = parse_args()
    model = build_model(args)

    assert type(model) is UNet2DModel
    assert model.config.in_channels == 3
    assert model.config.out_channels == 3
    assert model.config.num_class_embeds is None
    assert not hasattr(model, "y_init")

def test_model_factory_unconditional_small_loop(monkeypatch):
    """Test recursive unconditional model creation."""
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--dataset_name", "uoft-cs/cifar100",
        "--num_classes", "0",
        "--input_channels", "3",
        "--use_small_loop",
        "--model_type", "unet2d",
        "--dataset_type", "hf"
    ])
    args = parse_args()
    model = build_model(args)

    assert type(model) is UNet2DModel
    assert model.config.in_channels == 9  # 3 channels * 3 (x, y, z)
    assert model.config.out_channels == 6 # 3 channels * 2 (y, z)
    assert model.config.num_class_embeds is None
    assert hasattr(model, "y_init")
    assert hasattr(model, "z_init")

# ---------------------------------------------------------
# 2. CIFAR-100 (Simple Class-Conditional, No Cross-Attn)
# ---------------------------------------------------------

def test_model_factory_cifar_conditional_standard(monkeypatch):
    """Test CIFAR conditional using standard class embeddings."""
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--dataset_name", "uoft-cs/cifar100",
        "--num_classes", "100",
        "--input_channels", "3",
        "--down_block_types", "AttnDownBlock2D", "DownBlock2D",
        "--model_type", "unet2d",
        "--dataset_type", "hf"
    ])
    args = parse_args()
    model = build_model(args)

    assert type(model) is UNet2DModel
    assert model.config.num_class_embeds == 100
    assert model.config.in_channels == 3
    assert not hasattr(model, "y_init")

def test_model_factory_cifar_conditional_small_loop(monkeypatch):
    """Test CIFAR conditional recursive loop."""
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--dataset_name", "uoft-cs/cifar100",
        "--num_classes", "100",
        "--input_channels", "3",
        "--use_small_loop",
        "--down_block_types", "AttnDownBlock2D", "DownBlock2D",
        "--model_type", "unet2d",
        "--dataset_type", "hf"
    ])
    args = parse_args()
    model = build_model(args)

    assert type(model) is UNet2DModel
    assert model.config.num_class_embeds == 100
    assert model.config.in_channels == 9
    assert hasattr(model, "y_init")

# ---------------------------------------------------------
# 3. ImageNet (Complex Class-Conditional, Cross-Attn)
# ---------------------------------------------------------

def test_model_factory_imagenet_conditional_standard(monkeypatch):
    """Test ImageNet conditional using UnifiedConditionUNet (class mode)."""
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--dataset_name", "ILSVRC/imagenet-1k",
        "--num_classes", "1000",
        "--input_channels", "4",  # VAE latent space
        "--down_block_types", "CrossAttnDownBlock2D", "DownBlock2D",
        "--model_type", "unified_class",
        "--dataset_type", "hf",
    ])
    args = parse_args()
    model = build_model(args)

    assert type(model) is UnifiedConditionUNet
    assert model.config.condition_mode == "class"
    assert model.config.in_channels == 4
    # The parent UNet2DConditionModel should NOT build its own class embeds
    assert model.config.num_class_embeds is None
    assert not hasattr(model, "y_init")

def test_model_factory_imagenet_conditional_small_loop(monkeypatch):
    """Test ImageNet conditional recursive loop."""
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--dataset_name", "ILSVRC/imagenet-1k",
        "--num_classes", "1000",
        "--input_channels", "4",
        "--use_small_loop",
        "--down_block_types", "CrossAttnDownBlock2D", "DownBlock2D",
        "--model_type", "unified_class",
        "--dataset_type", "hf",
    ])
    args = parse_args()
    model = build_model(args)

    assert type(model) is UnifiedConditionUNet
    assert model.config.condition_mode == "class"
    assert model.config.in_channels == 12 # 4 * 3
    assert hasattr(model, "y_init")
    assert model.y_init.shape[1] == 4

# ---------------------------------------------------------
# 4. CLEVR (Sequence-Conditional, Cross-Attn)
# ---------------------------------------------------------

def test_model_factory_clevr_standard(monkeypatch):
    """Test CLEVR conditional using UnifiedConditionUNet (sequence mode)."""
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--train_data_dir", "cache_dir",
        "--output_dir", "clevr-standard-test",
        "--dataset_mode", "absolute",
        "--input_channels", "4",
        "--down_block_types", "CrossAttnDownBlock2D", "DownBlock2D",
        "--model_type", "unified_sequence",
        "--dataset_type", "clevr"
    ])
    args = parse_args()
    model = build_model(args)

    assert type(model) is UnifiedConditionUNet
    assert model.config.condition_mode == "sequence"
    assert model.config.raw_dim == 21
    assert model.config.in_channels == 4
    assert not hasattr(model, "y_init")

def test_model_factory_clevr_small_loop(monkeypatch):
    """Test CLEVR conditional recursive loop with relative mode."""
    monkeypatch.setattr("sys.argv", [
        "train.py",
        "--train_data_dir", "clevr_data",
        "--dataset_mode", "relative",
        "--input_channels", "4",
        "--use_small_loop",
        "--down_block_types", "CrossAttnDownBlock2D", "DownBlock2D",
        "--model_type", "unified_sequence",
        "--dataset_type", "clevr"
    ])
    args = parse_args()
    model = build_model(args)

    assert type(model) is UnifiedConditionUNet
    assert model.config.condition_mode == "sequence"
    assert model.config.raw_dim == 55
    assert model.config.in_channels == 12
    assert hasattr(model, "y_init")
    assert hasattr(model, "z_init")
