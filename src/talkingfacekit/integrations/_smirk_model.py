"""SMIRK encoder architecture adapted under its MIT license.

Source: georgeretsi/smirk, c7de404c4389f073906a6db1adabf62efcea3f35,
src/smirk_encoder.py. Copyright (c) 2024 George Retsinas. See SMIRK_LICENSE.txt.
Module names and forward operations match the upstream encoder checkpoint. Initial random
weights are replaced strictly; no ImageNet pretrained weights are requested or downloaded.
"""

from typing import Any, cast


def build_smirk_model(torch_module: object, timm_module: object) -> object:
    """Build the three upstream towers without loading weights or using the network.

    Parameters
    ----------
    torch_module, timm_module
        Already imported optional frameworks, kept inside the integration boundary.

    Returns
    -------
    object
        PyTorch module with upstream-compatible state names, 300 shape coefficients and
        50 expression coefficients. A complete checkpoint must be loaded before inference.
    """
    torch = cast(Any, torch_module)
    timm = cast(Any, timm_module)
    nn = torch.nn

    def tower(backbone_name: str, head_name: str, output_size: int) -> Any:
        branch = nn.Module()
        branch.encoder = timm.create_model(backbone_name, pretrained=False, features_only=True)
        feature_dim = branch.encoder.feature_info[-1]["num_chs"]
        setattr(branch, head_name, nn.Sequential(nn.Linear(feature_dim, output_size)))
        return branch

    model = nn.Module()
    model.pose_encoder = tower("tf_mobilenetv3_small_minimal_100", "pose_cam_layers", 6)
    model.shape_encoder = tower("tf_mobilenetv3_large_minimal_100", "shape_layers", 300)
    model.expression_encoder = tower("tf_mobilenetv3_large_minimal_100", "expression_layers", 55)

    def encode(branch: Any, head_name: str, image: object) -> Any:
        features = branch.encoder(image)[-1]
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.squeeze(-1).squeeze(-1)
        return getattr(branch, head_name)(features)

    def forward(image: object) -> dict[str, object]:
        pose_camera = encode(model.pose_encoder, "pose_cam_layers", image)
        shape = encode(model.shape_encoder, "shape_layers", image)
        expression = encode(model.expression_encoder, "expression_layers", image)
        jaw = torch.cat(
            [
                torch.nn.functional.relu(expression[..., 52].unsqueeze(-1)),
                torch.clamp(expression[..., 53:55], -0.2, 0.2),
            ],
            dim=-1,
        )
        return {
            "pose_params": pose_camera[..., :3],
            "cam": pose_camera[..., 3:],
            "shape_params": shape,
            "expression_params": expression[..., :50],
            "eyelid_params": torch.clamp(expression[..., 50:52], 0, 1),
            "jaw_params": jaw,
        }

    model.forward = forward
    return model
