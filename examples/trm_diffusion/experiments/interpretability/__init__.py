from .z_changes_ablation import ZChangesAblation
from .z_path_encoding_ablation import ZPathEncodingAblation
from .z_representation_ablation import ZRepresentationAblation


def launch_interpretability_ablation(ablation_name: str):
    ablation_map = {
        "z_changes_ablation": ZChangesAblation,
        "z_representation_ablation": ZRepresentationAblation,
        "z_path_encoding_ablation": ZPathEncodingAblation,
    }
    if ablation_name not in ablation_map:
        raise ValueError(f"Unknown ablation {ablation_name!r}, choose from {sorted(ablation_map)}")
    return ablation_map[ablation_name]
