"""NumPy evaluation of FLAME blend shapes, joint articulation and linear blend skinning.

This module implements the numerical model equations, without importing upstream executables.
Asset access, parameter validation and result construction belong to the public adapter.
"""

import numpy as np
from numpy.typing import NDArray

from talkingfacekit.integrations._flame_assets import FlameModelData


def evaluate_flame(
    model: FlameModelData,
    shape: NDArray[np.float32],
    expression: NDArray[np.float32],
    global_pose: NDArray[np.float32],
    jaw_pose: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Evaluate a validated batch in native model coordinates with neutral neck and eyeballs."""
    count = shape.shape[0]
    components = np.concatenate(
        (
            model.directions[:, :, : shape.shape[1]],
            model.directions[:, :, 300 : 300 + expression.shape[1]],
        ),
        axis=2,
    )
    coefficients = np.concatenate((shape, expression), axis=1)
    shaped = model.template[None] + (
        coefficients @ components.reshape(-1, coefficients.shape[1]).T
    ).reshape(count, -1, 3)
    joints = np.einsum("jv,bvc->bjc", model.joint_regressor, shaped, optimize=True)
    pose = np.zeros((count, 5, 3), dtype=np.float32)
    pose[:, 0] = global_pose
    pose[:, 2] = jaw_pose
    rotations = _rotation_matrices(pose)
    corrective = (rotations[:, 1:] - np.eye(3, dtype=np.float32)).reshape(count, 36)
    posed = shaped + (corrective @ model.pose_directions.reshape(-1, 36).T).reshape(count, -1, 3)

    transforms = np.broadcast_to(np.eye(4, dtype=np.float32), (count, 5, 4, 4)).copy()
    transforms[:, :, :3, :3] = rotations
    transforms[:, 0, :3, 3] = joints[:, 0]
    for joint in range(1, 5):
        parent = int(model.parents[joint])
        transforms[:, joint, :3, 3] = joints[:, joint] - joints[:, parent]
        transforms[:, joint] = transforms[:, parent] @ transforms[:, joint]
    transforms[:, :, :3, 3] -= np.einsum("bjik,bjk->bji", transforms[:, :, :3, :3], joints)
    weighted = np.einsum("vj,bjkl->bvkl", model.weights, transforms[:, :, :3], optimize=True)
    vertices = np.einsum("bvik,bvk->bvi", weighted[:, :, :, :3], posed) + weighted[:, :, :, 3]
    return np.asarray(vertices, dtype=np.float32)


def _rotation_matrices(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Use Rodrigues' formula with sinc factors that remain defined at zero angle."""
    skew = np.zeros((*vectors.shape[:-1], 3, 3), dtype=np.float32)
    x, y, z = np.moveaxis(vectors, -1, 0)
    skew[..., 0, 1], skew[..., 0, 2] = -z, y
    skew[..., 1, 0], skew[..., 1, 2] = z, -x
    skew[..., 2, 0], skew[..., 2, 1] = -y, x
    angle = np.linalg.norm(vectors, axis=-1)[..., None, None]
    sine_factor = np.sinc(angle / np.float32(np.pi))
    cosine_factor = np.float32(0.5) * np.sinc(angle / np.float32(2 * np.pi)) ** 2
    return np.asarray(
        np.eye(3, dtype=np.float32) + sine_factor * skew + cosine_factor * (skew @ skew),
        dtype=np.float32,
    )
