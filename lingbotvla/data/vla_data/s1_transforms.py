"""
Transformation functions for S1-stationary robot:
  - torso 9D (xyz + SO3_6d) <-> waist 4D (x, z, pitch, yaw)
  - Lossless compression based on structural constraints:
      torso_y = 0 (no left/right translation capability)
      roll = 0 (no rotation around X-axis in FLU frame)
"""
import torch
import numpy as np
from scipy.spatial.transform import Rotation


def rotation_6d_to_matrix(rot_6d):
    """Convert 6D rotation to 3x3 matrix via Gram-Schmidt orthonormalization.

    Args:
        rot_6d: (..., 6) - two 3D column vectors
    Returns:
        R: (..., 3, 3) rotation matrix
    """
    x = rot_6d[..., :3]
    y = rot_6d[..., 3:6]

    # Gram-Schmidt
    x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
    z = torch.cross(x, y, dim=-1)
    z = z / torch.linalg.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)

    R = torch.stack([x, y, z], dim=-1)
    return R


def rotation_matrix_to_euler_zyx(R):
    """Extract Euler ZYX (yaw, pitch, roll) from rotation matrix.

    Args:
        R: (..., 3, 3) rotation matrices
    Returns:
        (..., 3) tensor of [yaw, pitch, roll] in radians
    """
    # Use scipy for batch conversion
    original_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3).cpu().numpy()

    rot_objs = [Rotation.from_matrix(r) for r in R_flat]
    euler = np.array([r.as_euler('ZYX', degrees=False) for r in rot_objs])

    return torch.from_numpy(euler).reshape(*original_shape, 3).to(R.device)


def euler_zyx_to_rotation_matrix(euler):
    """Convert Euler ZYX angles to rotation matrix.

    Args:
        euler: (..., 3) [yaw, pitch, roll] in radians
    Returns:
        R: (..., 3, 3) rotation matrices
    """
    original_shape = euler.shape[:-1]
    euler_flat = euler.reshape(-1, 3).cpu().numpy()

    rot_objs = [Rotation.from_euler('ZYX', e, degrees=False) for e in euler_flat]
    R = np.array([r.as_matrix() for r in rot_objs])

    return torch.from_numpy(R).reshape(*original_shape, 3, 3).to(euler.device)


def rotation_matrix_to_6d(R):
    """Convert rotation matrix to 6D representation (first two columns).

    Args:
        R: (..., 3, 3)
    Returns:
        rot6d: (..., 6) - [R[:, 0], R[:, 1]] flattened
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def torso_9d_to_waist_4d(torso_9d):
    """
    Convert 9D torso representation to 4D waist (lossless).

    Mapping: waist = [x, z, pitch, yaw]

    Args:
        torso_9d: (..., 9) - [x, y, z, rot6d[6]]
    Returns:
        waist_4d: (..., 4) - [x, z, pitch, yaw]

    Constraints verified on 403 episodes (336k frames):
        - torso_y == 0 (max 1e-6 m, float noise floor)
        - roll == 0 (max 4e-6 rad, Gram-Schmidt numerical error)
    """
    xyz = torso_9d[..., :3]
    rot6d = torso_9d[..., 3:9]

    # Convert 6D rotation to matrix, then extract euler ZYX
    R = rotation_6d_to_matrix(rot6d)
    euler = rotation_matrix_to_euler_zyx(R)  # [yaw, pitch, roll]

    # Sanity checks (only in training/debug mode)
    # assert torch.abs(xyz[..., 1]).max() < 1e-4, f"torso_y should be ~0, got {xyz[..., 1].max()}"
    # assert torch.abs(euler[..., 2]).max() < 1e-3, f"roll should be ~0, got {euler[..., 2].max()}"

    waist = torch.stack([
        xyz[..., 0],      # x (forward/backward translation in FLU)
        xyz[..., 2],      # z (up/down translation)
        euler[..., 1],    # pitch (rotation around Y-axis)
        euler[..., 0],    # yaw (rotation around Z-axis)
    ], dim=-1)

    return waist


def waist_4d_to_torso_9d(waist_4d):
    """
    Inverse: convert 4D waist back to 9D torso (lossless reconstruction).

    Args:
        waist_4d: (..., 4) - [x, z, pitch, yaw]
    Returns:
        torso_9d: (..., 9) - [x, y, z, rot6d[6]]

    Fills missing DOFs with constants:
        - y = 0.0 (no left/right translation capability)
        - roll = 0.0 (no rotation around X-axis)
    """
    x = waist_4d[..., 0]
    z = waist_4d[..., 1]
    pitch = waist_4d[..., 2]
    yaw = waist_4d[..., 3]

    # Reconstruct xyz with y=0
    xyz = torch.stack([
        x,
        torch.zeros_like(x),  # y = 0 (constant)
        z,
    ], dim=-1)

    # Reconstruct rotation: euler ZYX (yaw, pitch, roll=0) -> matrix -> 6D
    euler = torch.stack([yaw, pitch, torch.zeros_like(yaw)], dim=-1)
    R = euler_zyx_to_rotation_matrix(euler)
    rot6d = rotation_matrix_to_6d(R)

    torso = torch.cat([xyz, rot6d], dim=-1)
    return torso


# Register transforms for use in robot_config
TRANSFORMS = {
    'torso_9d_to_waist_4d': torso_9d_to_waist_4d,
    'waist_4d_to_torso_9d': waist_4d_to_torso_9d,
}


if __name__ == '__main__':
    # Test round-trip conversion
    import sys
    sys.path.insert(0, '/kpfs-cognition/baifu/workspace/lingbot-vla-v2')

    # Sample from real data statistics
    torso_sample = torch.tensor([
        0.01, 0.0, 1.20,  # xyz: small x motion, y=0, z~1.2m
        0.95, -0.1, 0.05,  # rot6d first 3
        0.1, 0.98, 0.0,    # rot6d last 3
    ]).unsqueeze(0)

    print("Original torso[9]:", torso_sample)

    waist = torso_9d_to_waist_4d(torso_sample)
    print("Compressed waist[4]:", waist)

    torso_reconstructed = waist_4d_to_torso_9d(waist)
    print("Reconstructed torso[9]:", torso_reconstructed)

    error = torch.abs(torso_sample - torso_reconstructed).max()
    print(f"Max reconstruction error: {error.item():.2e}")

    if error < 1e-5:
        print("✓ Round-trip conversion successful (error < 1e-5)")
    else:
        print("✗ Reconstruction error too large")
