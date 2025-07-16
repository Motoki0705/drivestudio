"""
Pure DriveStudio SE(3) spline trajectory library

This module implements SE(3) spline interpolation without external dependencies
like pypose, providing the core functionality needed for BAD-Gaussians
camera trajectory generation.
"""

from __future__ import annotations
from typing import Literal

import torch
import torch.nn.functional as F

_EPS = 1e-6

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion inverse (conjugate for unit quaternions)."""
    # Normalize quaternion
    q_norm = F.normalize(q, p=2, dim=-1)
    # Conjugate: (w, -x, -y, -z)
    q_inv = q_norm.clone()
    q_inv[..., 1:] *= -1
    return q_inv

def quaternion_slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between quaternions."""
    # Ensure quaternions are unit
    q1 = F.normalize(q1, p=2, dim=-1)
    q2 = F.normalize(q2, p=2, dim=-1)
    
    # Compute dot product
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    
    # If dot product is negative, use -q2 to take shorter path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)
    
    # If quaternions are very close, use linear interpolation
    close_mask = dot > 0.9995
    
    # Linear interpolation for close quaternions
    linear_interp = q1 + t.unsqueeze(-1) * (q2 - q1)
    linear_interp = F.normalize(linear_interp, p=2, dim=-1)
    
    # Spherical interpolation for distant quaternions
    theta_0 = torch.acos(torch.clamp(dot, -1, 1))
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * t.unsqueeze(-1)
    sin_theta = torch.sin(theta)
    
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    spherical_interp = s0 * q1 + s1 * q2
    
    # Choose interpolation method based on distance
    result = torch.where(close_mask.unsqueeze(-1), linear_interp, spherical_interp)
    return F.normalize(result, p=2, dim=-1)

def se3_exp_map(xi: torch.Tensor) -> torch.Tensor:
    """SE(3) exponential map from 6DOF to 4x4 matrix.
    
    Args:
        xi: [N, 6] tensor in format [tx, ty, tz, rx, ry, rz]
        
    Returns:
        SE(3) matrices [N, 4, 4]
    """
    batch_size = xi.shape[0]
    translation = xi[..., :3]  # [N, 3]
    rotation = xi[..., 3:]     # [N, 3] (axis-angle)
    
    # Convert axis-angle to rotation matrix using Rodrigues' formula
    angle = torch.norm(rotation, dim=-1, keepdim=True)  # [N, 1]
    
    # Handle small angles
    small_angle_mask = (angle < _EPS).squeeze(-1)
    
    # For small angles, use series expansion
    axis = torch.where(
        angle < _EPS,
        torch.zeros_like(rotation),
        rotation / angle
    )
    
    cos_angle = torch.cos(angle)  # [N, 1]
    sin_angle = torch.sin(angle)  # [N, 1]
    one_minus_cos = 1 - cos_angle  # [N, 1]
    
    # Skew-symmetric matrix
    K = torch.zeros(batch_size, 3, 3, device=xi.device, dtype=xi.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    # Rotation matrix using Rodrigues' formula
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    
    R = torch.where(
        small_angle_mask.unsqueeze(-1).unsqueeze(-1),
        I + K,  # First-order approximation for small angles
        I + sin_angle.unsqueeze(-1) * K + one_minus_cos.unsqueeze(-1) * torch.bmm(K, K)
    )
    
    # SE(3) matrix construction
    T = torch.zeros(batch_size, 4, 4, device=xi.device, dtype=xi.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = translation
    T[:, 3, 3] = 1.0
    
    return T

def se3_log_map(T: torch.Tensor) -> torch.Tensor:
    """SE(3) logarithm map from 4x4 matrix to 6DOF.
    
    Args:
        T: SE(3) matrices [N, 4, 4]
        
    Returns:
        6DOF vectors [N, 6] in format [tx, ty, tz, rx, ry, rz]
    """
    batch_size = T.shape[0]
    R = T[:, :3, :3]  # [N, 3, 3]
    t = T[:, :3, 3]   # [N, 3]
    
    # Convert rotation matrix to axis-angle
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)  # [N]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))  # [N]
    
    # Handle small angles
    small_angle_mask = angle < _EPS
    
    # For small angles, use series expansion
    axis = torch.zeros(batch_size, 3, device=T.device, dtype=T.dtype)
    
    # For non-small angles
    large_angle_mask = ~small_angle_mask
    if large_angle_mask.any():
        sin_angle = torch.sin(angle[large_angle_mask])
        axis[large_angle_mask, 0] = (R[large_angle_mask, 2, 1] - R[large_angle_mask, 1, 2]) / (2 * sin_angle)
        axis[large_angle_mask, 1] = (R[large_angle_mask, 0, 2] - R[large_angle_mask, 2, 0]) / (2 * sin_angle)
        axis[large_angle_mask, 2] = (R[large_angle_mask, 1, 0] - R[large_angle_mask, 0, 1]) / (2 * sin_angle)
    
    # Axis-angle representation
    rotation_vec = axis * angle.unsqueeze(-1)
    
    # Combine translation and rotation
    xi = torch.cat([t, rotation_vec], dim=-1)
    
    return xi

def linear_interpolation_mid_se3(
    ctrl_knots: torch.Tensor,
) -> torch.Tensor:
    """Get the midpoint between two SE(3) poses by proper interpolation.
    
    Args:
        ctrl_knots: [N, 2, 6] control knots in 6DOF format
        
    Returns:
        Midpoint poses [N, 6]
    """
    start_xi, end_xi = ctrl_knots[:, 0, :], ctrl_knots[:, 1, :]  # [N, 6]
    
    # Convert to SE(3) matrices
    T_start = se3_exp_map(start_xi)  # [N, 4, 4]
    T_end = se3_exp_map(end_xi)      # [N, 4, 4]
    
    # Compute relative transformation
    T_rel = torch.bmm(torch.linalg.inv(T_start), T_end)  # [N, 4, 4]
    
    # Half-way transformation
    xi_rel = se3_log_map(T_rel)  # [N, 6]
    xi_half = xi_rel * 0.5
    T_half = se3_exp_map(xi_half)  # [N, 4, 4]
    
    # Final midpoint transformation
    T_mid = torch.bmm(T_start, T_half)  # [N, 4, 4]
    
    return se3_log_map(T_mid)

def linear_interpolation_se3(
    ctrl_knots: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Linear interpolation between two SE(3) poses.
    
    Args:
        ctrl_knots: [N, 2, 6] control knots
        u: [M] interpolation parameters in [0, 1]
        
    Returns:
        Interpolated poses [N, M, 6]
    """
    batch_size, _, _ = ctrl_knots.shape
    num_interp = u.shape[0]
    
    start_xi, end_xi = ctrl_knots[:, 0, :], ctrl_knots[:, 1, :]  # [N, 6]
    
    # Convert to SE(3) matrices
    T_start = se3_exp_map(start_xi)  # [N, 4, 4]
    T_end = se3_exp_map(end_xi)      # [N, 4, 4]
    
    # Compute relative transformation
    T_rel = torch.bmm(torch.linalg.inv(T_start), T_end)  # [N, 4, 4]
    xi_rel = se3_log_map(T_rel)  # [N, 6]
    
    # Interpolate in tangent space
    results = []
    for t in u:
        xi_t = xi_rel * t
        T_t = se3_exp_map(xi_t)  # [N, 4, 4]
        T_interp = torch.bmm(T_start, T_t)  # [N, 4, 4]
        xi_interp = se3_log_map(T_interp)  # [N, 6]
        results.append(xi_interp)
    
    return torch.stack(results, dim=1)  # [N, M, 6]

def cubic_bspline_interpolation_se3(
    ctrl_knots: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Cubic B-spline interpolation with four SE(3) control knots.
    
    Args:
        ctrl_knots: [N, 4, 6] control knots
        u: [M] normalized positions in [0, 1]
        
    Returns:
        Interpolated poses [N, M, 6]
    """
    batch_size = ctrl_knots.shape[0]
    num_interp = u.shape[0]
    
    # Clamp u to avoid numerical issues
    u = torch.clamp(u, _EPS, 1.0 - _EPS)
    
    uu = u * u
    uuu = uu * u
    oos = 1.0 / 6.0  # one over six
    
    # B-spline basis functions
    b0 = oos - 0.5 * u + 0.5 * uu - oos * uuu
    b1 = 4.0 * oos - uu + 0.5 * uuu
    b2 = oos + 0.5 * u + 0.5 * uu - 0.5 * uuu
    b3 = oos * uuu
    
    coeffs = torch.stack([b0, b1, b2, b3], dim=0)  # [4, M]
    
    results = []
    for i in range(num_interp):
        # Weighted combination in 6DOF space (approximation)
        xi_interp = (
            coeffs[0, i] * ctrl_knots[:, 0, :] +
            coeffs[1, i] * ctrl_knots[:, 1, :] +
            coeffs[2, i] * ctrl_knots[:, 2, :] +
            coeffs[3, i] * ctrl_knots[:, 3, :]
        )
        results.append(xi_interp)
    
    return torch.stack(results, dim=1)  # [N, M, 6]

def bezier_interpolation_se3(
    ctrl_knots: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Bezier curve interpolation with SE(3) control points.
    
    Args:
        ctrl_knots: [N, K, 6] control knots (K = bezier degree)
        u: [M] parameters in [0, 1]
        
    Returns:
        Interpolated poses [N, M, 6]
    """
    batch_size, degree, _ = ctrl_knots.shape
    num_interp = u.shape[0]
    
    results = []
    for t in u:
        # De Casteljau's algorithm approximation in 6DOF space
        current_points = ctrl_knots.clone()  # [N, K, 6]
        
        for level in range(degree - 1):
            next_points = []
            for i in range(degree - level - 1):
                # Linear interpolation between adjacent points
                p_interp = (1 - t) * current_points[:, i, :] + t * current_points[:, i + 1, :]
                next_points.append(p_interp)
            current_points = torch.stack(next_points, dim=1)
        
        results.append(current_points[:, 0, :])  # [N, 6]
    
    return torch.stack(results, dim=1)  # [N, M, 6]
