"""
DriveStudio-native camera optimizer for BAD-Gaussians integration.

This module provides camera optimization functionality without requiring
nerfstudio dependencies, using only drivestudio's native components.
"""

from __future__ import annotations

import functools
from copy import deepcopy
from typing import List, Literal, Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import Parameter
from omegaconf import OmegaConf

from models.gaussians.basics import dataclass_camera
from models.bad_gaussians.pure_spline_functor import (
    bezier_interpolation_se3,
    cubic_bspline_interpolation_se3,
    linear_interpolation_se3,
    linear_interpolation_mid_se3,
)

TrajSamplingMode = Literal["uniform", "start", "mid", "end"]
"""How to sample the camera trajectory"""

@dataclass
class DriveStudioCameraOptimizerConfig:
    """Configuration for DriveStudio camera optimizer."""
    
    mode: Literal["off", "linear", "cubic", "bezier"] = "linear"
    """Pose optimization strategy to use.
    linear: linear interpolation on SE(3);
    cubic: cubic b-spline interpolation on SE(3).
    bezier: Bezier curve interpolation on SE(3).
    """
    
    bezier_degree: int = 9
    """Degree of the Bezier curve. Only used when mode is bezier."""
    
    trans_l2_penalty: float = 0.0
    """L2 penalty on translation parameters."""
    
    rot_l2_penalty: float = 0.0
    """L2 penalty on rotation parameters."""
    
    num_virtual_views: int = 10
    """The number of samples used to model the motion-blurring."""
    
    initial_noise_se3_std: float = 1e-5
    """Initial perturbation to pose delta on se(3). Must be non-zero to prevent NaNs."""

class DriveStudioCameraOptimizer(nn.Module):
    """DriveStudio-native camera optimizer for virtual camera trajectories."""
    
    def __init__(
        self,
        config: DriveStudioCameraOptimizerConfig,
        num_cameras: int,
        device: torch.device,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        
        # Initialize learnable parameters if mode is not "off"
        if self.config.mode == "off":
            self.pose_adjustment = None
            return
        elif self.config.mode == "linear":
            self.num_control_knots = 2
        elif self.config.mode == "cubic":
            self.num_control_knots = 4
        elif self.config.mode == "bezier":
            self.num_control_knots = self.config.bezier_degree
        else:
            raise ValueError(f"Unknown camera optimizer mode: {self.config.mode}")
        
        # Initialize pose adjustment parameters
        # Using 6DOF representation: [tx, ty, tz, rx, ry, rz]
        self.pose_adjustment = Parameter(
            torch.randn(
                (num_cameras, self.num_control_knots, 6),
                device=device,
                dtype=torch.float32
            ) * self.config.initial_noise_se3_std
        )
    
    def forward(
        self,
        camera_indices: torch.Tensor,
        mode: TrajSamplingMode = "mid"
    ) -> torch.Tensor:
        """Generate pose adjustments for given camera indices.
        
        Args:
            camera_indices: Camera indices to optimize
            mode: Sampling mode for trajectory
            
        Returns:
            Pose adjustments as transformation matrices [N, 4, 4]
        """
        if self.config.mode == "off" or self.pose_adjustment is None:
            # Return identity transformations
            batch_size = camera_indices.shape[0]
            return torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Get unique indices for efficiency
        camera_indices = camera_indices.long()
        unique_indices, inverse_indices = torch.unique(camera_indices, return_inverse=True)
        
        # Get pose adjustments for unique cameras
        pose_params = self.pose_adjustment[unique_indices]  # [unique_cams, control_knots, 6]
        
        # Interpolate based on mode
        interpolated_params = self._interpolate_pose_params(pose_params, mode)
        
        # Convert 6DOF parameters to transformation matrices
        transform_matrices = self._pose_params_to_matrix(interpolated_params)
        
        # Map back to original indices
        return transform_matrices[inverse_indices]
    
    def _interpolate_pose_params(
        self,
        pose_params: torch.Tensor,
        mode: TrajSamplingMode
    ) -> torch.Tensor:
        """Interpolate pose parameters based on sampling mode.
        
        Args:
            pose_params: [N, control_knots, 6]
            mode: Sampling mode
            
        Returns:
            Interpolated parameters [N, num_virtual_views, 6] or [N, 6]
        """
        if mode == "uniform":
            # Generate uniform samples
            u = torch.linspace(
                0.0, 1.0,
                self.config.num_virtual_views,
                device=pose_params.device,
                dtype=pose_params.dtype
            )
            
            if self.config.mode == "linear":
                return self._linear_interpolation(pose_params, u)
            elif self.config.mode == "cubic":
                return self._cubic_interpolation(pose_params, u)
            elif self.config.mode == "bezier":
                return self._bezier_interpolation(pose_params, u)
                
        elif mode == "mid":
            # Return middle point
            if self.config.mode == "linear":
                return self._linear_interpolation_mid(pose_params)
            else:
                u = torch.tensor([0.5], device=pose_params.device, dtype=pose_params.dtype)
                if self.config.mode == "cubic":
                    return self._cubic_interpolation(pose_params, u).squeeze(1)
                elif self.config.mode == "bezier":
                    return self._bezier_interpolation(pose_params, u).squeeze(1)
                    
        elif mode == "start":
            return pose_params[:, 0, :]  # First control point
            
        elif mode == "end":
            return pose_params[:, -1, :]  # Last control point
        
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
    
    def _linear_interpolation(self, pose_params: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two control points using SE(3)."""
        return linear_interpolation_se3(pose_params, u)
    
    def _linear_interpolation_mid(self, pose_params: torch.Tensor) -> torch.Tensor:
        """Linear interpolation at midpoint using SE(3)."""
        return linear_interpolation_mid_se3(pose_params)
    
    def _cubic_interpolation(self, pose_params: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Cubic B-spline interpolation using SE(3)."""
        return cubic_bspline_interpolation_se3(pose_params, u)
    
    def _bezier_interpolation(self, pose_params: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Bezier curve interpolation using SE(3)."""
        return bezier_interpolation_se3(pose_params, u)
    
    def _pose_params_to_matrix(self, pose_params: torch.Tensor) -> torch.Tensor:
        """Convert 6DOF pose parameters to 4x4 transformation matrices.
        
        Args:
            pose_params: [N, 6] or [N, num_views, 6]
            
        Returns:
            Transformation matrices [N, 4, 4] or [N, num_views, 4, 4]
        """
        original_shape = pose_params.shape
        if len(original_shape) == 3:
            # Reshape for batch processing
            pose_params = pose_params.view(-1, 6)
        
        # Extract translation and rotation
        translation = pose_params[:, :3]  # [N, 3]
        rotation = pose_params[:, 3:]     # [N, 3] (axis-angle representation)
        
        # Convert axis-angle to rotation matrix
        angle = torch.norm(rotation, dim=-1, keepdim=True)  # [N, 1]
        # Avoid division by zero
        angle_safe = torch.where(angle < 1e-8, torch.ones_like(angle), angle)
        axis = rotation / angle_safe  # [N, 3]
        
        # Rodrigues' rotation formula
        cos_angle = torch.cos(angle)  # [N, 1]
        sin_angle = torch.sin(angle)  # [N, 1]
        one_minus_cos = 1 - cos_angle  # [N, 1]
        
        # Skew-symmetric matrix
        K = torch.zeros(pose_params.shape[0], 3, 3, device=pose_params.device, dtype=pose_params.dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rotation matrix using Rodrigues' formula
        I = torch.eye(3, device=pose_params.device, dtype=pose_params.dtype).unsqueeze(0).expand_as(K)
        
        # Handle small angles
        small_angle_mask = (angle < 1e-8).squeeze(-1)
        R = torch.where(
            small_angle_mask.unsqueeze(-1).unsqueeze(-1),
            I,  # Identity for small angles
            I + sin_angle.unsqueeze(-1) * K + one_minus_cos.unsqueeze(-1) * torch.bmm(K, K)
        )
        
        # Construct 4x4 transformation matrix
        T = torch.zeros(pose_params.shape[0], 4, 4, device=pose_params.device, dtype=pose_params.dtype)
        T[:, :3, :3] = R
        T[:, :3, 3] = translation
        T[:, 3, 3] = 1.0
        
        # Reshape back if necessary
        if len(original_shape) == 3:
            T = T.view(original_shape[0], original_shape[1], 4, 4)
        
        return T
    
    def apply_to_camera(
        self,
        cam: dataclass_camera,
        mode: TrajSamplingMode = "uniform"
    ) -> List[dataclass_camera]:
        """Apply camera optimization to generate virtual cameras.
        
        Args:
            cam: Input camera
            mode: Sampling mode
            
        Returns:
            List of virtual cameras
        """
        if self.config.mode == "off":
            return [cam]
        
        # Get camera index
        cam_idx = 0
        if hasattr(cam, 'metadata') and cam.metadata and 'cam_idx' in cam.metadata:
            cam_idx = cam.metadata['cam_idx']
        
        # Generate pose adjustments
        camera_indices = torch.tensor([cam_idx], device=self.device)
        pose_deltas = self.forward(camera_indices, mode)  # [1, num_views, 4, 4] or [1, 4, 4]
        
        if len(pose_deltas.shape) == 3:
            # Single view mode
            pose_deltas = pose_deltas.unsqueeze(1)  # [1, 1, 4, 4]
        
        num_views = pose_deltas.shape[1]
        
        # Get original camera-to-world matrix
        c2w_original = cam.camtoworlds  # [4, 4]
        if c2w_original.dim() == 2:
            c2w_original = c2w_original.unsqueeze(0)  # [1, 4, 4]
        
        # Apply pose adjustments
        virtual_cameras = []
        for i in range(num_views):
            # Apply transformation: c2w_new = c2w_original @ pose_delta
            c2w_new = torch.bmm(
                c2w_original,
                pose_deltas[:, i:i+1, :, :]
            ).squeeze(0)  # [4, 4]
            
            # Create virtual camera
            virtual_cam = deepcopy(cam)
            virtual_cam.camtoworlds = c2w_new
            virtual_cam.camtoworlds_gt = c2w_new.clone()
            
            # Preserve metadata
            if hasattr(cam, 'metadata'):
                virtual_cam.metadata = getattr(cam, 'metadata', {}).copy()
            else:
                virtual_cam.metadata = {}
            virtual_cam.metadata['cam_idx'] = cam_idx
            virtual_cam.metadata['virtual_view_idx'] = i
            
            virtual_cameras.append(virtual_cam)
        
        return virtual_cameras
    
    def get_loss_dict(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        """Add camera optimizer regularization losses."""
        if self.config.mode == "off" or self.pose_adjustment is None:
            return
        
        # Translation L2 penalty
        if self.config.trans_l2_penalty > 0:
            trans_penalty = torch.mean(self.pose_adjustment[:, :, :3] ** 2)
            loss_dict["cam_opt_trans_l2"] = self.config.trans_l2_penalty * trans_penalty
        
        # Rotation L2 penalty
        if self.config.rot_l2_penalty > 0:
            rot_penalty = torch.mean(self.pose_adjustment[:, :, 3:] ** 2)
            loss_dict["cam_opt_rot_l2"] = self.config.rot_l2_penalty * rot_penalty
    
    def get_metrics_dict(self, metrics_dict: Dict[str, torch.Tensor]) -> None:
        """Add camera optimizer metrics."""
        if self.config.mode == "off" or self.pose_adjustment is None:
            return
        
        # Translation magnitude
        trans_mag = torch.mean(torch.norm(self.pose_adjustment[:, :, :3], dim=-1))
        metrics_dict["cam_opt_trans_magnitude"] = trans_mag
        
        # Rotation magnitude
        rot_mag = torch.mean(torch.norm(self.pose_adjustment[:, :, 3:], dim=-1))
        metrics_dict["cam_opt_rot_magnitude"] = rot_mag
        
        # Trajectory length (for multi-knot modes)
        if self.num_control_knots > 1:
            traj_trans = torch.norm(
                self.pose_adjustment[:, 1:, :3] - self.pose_adjustment[:, :-1, :3],
                dim=-1
            ).mean()
            traj_rot = torch.norm(
                self.pose_adjustment[:, 1:, 3:] - self.pose_adjustment[:, :-1, 3:],
                dim=-1
            ).mean()
            metrics_dict["cam_opt_traj_trans"] = traj_trans
            metrics_dict["cam_opt_traj_rot"] = traj_rot
