"""
Pure DriveStudio BAD-Gaussians integrated VanillaGaussians implementation

This module integrates BAD-Gaussians motion blur handling capabilities
into the OmniRe VanillaGaussians background representation without
requiring any nerfstudio dependencies.
"""

from typing import Dict, List, Tuple, Optional, Literal
from omegaconf import OmegaConf
import logging
import copy

import torch
import torch.nn as nn
from torch.nn import Parameter

from models.gaussians.vanilla import VanillaGaussians
from models.gaussians.basics import *
from models.bad_gaussians.drivestudio_camera_optimizer import (
    DriveStudioCameraOptimizer, 
    DriveStudioCameraOptimizerConfig, 
    TrajSamplingMode
)
from models.bad_gaussians.bad_losses import EdgeAwareVariationLoss

logger = logging.getLogger()

class PureBADVanillaGaussians(VanillaGaussians):
    """
    Pure DriveStudio BAD-Gaussians enhanced VanillaGaussians class.
    
    Extends VanillaGaussians with motion blur handling through
    virtual camera trajectory generation and multi-view rendering,
    using only DriveStudio native components.
    """

    def __init__(
        self,
        class_name: str,
        ctrl: OmegaConf,
        reg: OmegaConf = None,
        networks: OmegaConf = None,
        scene_scale: float = 30.,
        scene_origin: torch.Tensor = torch.zeros(3),
        num_train_images: int = 300,
        device: torch.device = torch.device("cuda"),
        camera_optimizer: Optional[OmegaConf] = None,
        **kwargs
    ):
        super().__init__(
            class_name=class_name,
            ctrl=ctrl,
            reg=reg,
            networks=networks,
            scene_scale=scene_scale,
            scene_origin=scene_origin,
            num_train_images=num_train_images,
            device=device,
            **kwargs
        )
        
        # Initialize camera optimizer for motion blur handling
        self.camera_optimizer = None
        self.camera_optimizer_cfg = camera_optimizer
        if camera_optimizer is not None:
            self._setup_camera_optimizer(camera_optimizer, num_train_images, device)
            
        # Initialize TV loss for regularization
        self.tv_loss = EdgeAwareVariationLoss(in1_nc=3)
        
        # Training mode flag
        self._is_training = True
        
        # Store last rendered RGB for TV loss computation
        self._last_rendered_rgb = None
        
    def _setup_camera_optimizer(self, camera_optimizer_cfg: OmegaConf, num_cameras: int, device: torch.device):
        """Setup the DriveStudio camera optimizer"""
        try:
            # Convert OmegaConf to DriveStudioCameraOptimizerConfig
            config = DriveStudioCameraOptimizerConfig(
                mode=camera_optimizer_cfg.get("mode", "linear"),
                bezier_degree=camera_optimizer_cfg.get("bezier_degree", 9),
                trans_l2_penalty=camera_optimizer_cfg.get("trans_l2_penalty", 0.0),
                rot_l2_penalty=camera_optimizer_cfg.get("rot_l2_penalty", 0.0),
                num_virtual_views=camera_optimizer_cfg.get("num_virtual_views", 10),
                initial_noise_se3_std=camera_optimizer_cfg.get("initial_noise_se3_std", 1e-5)
            )
            
            self.camera_optimizer = DriveStudioCameraOptimizer(
                config=config,
                num_cameras=num_cameras,
                device=device
            )
            logger.info(f"Initialized DriveStudio camera optimizer with mode: {config.mode}")
        except Exception as e:
            logger.warning(f"Failed to initialize camera optimizer: {e}")
            self.camera_optimizer = None
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self._is_training = mode
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode"""
        self._is_training = False
        return super().eval()
    
    def get_gaussians(self, cam: dataclass_camera, mode: TrajSamplingMode = "uniform") -> Dict:
        """
        Get Gaussian parameters with optional motion blur handling.
        
        Args:
            cam: Camera parameters
            mode: Trajectory sampling mode for virtual views
            
        Returns:
            Dictionary containing Gaussian parameters
        """
        # Use BAD-Gaussians multi-view rendering if available and in training
        if (self.camera_optimizer is not None and 
            self._is_training and 
            self.camera_optimizer.config.mode != "off"):
            return self._get_gaussians_multi_view(cam, mode)
        else:
            # Fallback to standard single-view rendering
            return super().get_gaussians(cam)
    
    def _get_gaussians_multi_view(self, cam: dataclass_camera, mode: TrajSamplingMode) -> Dict:
        """
        Multi-view rendering with virtual cameras for motion blur effect.
        
        Args:
            cam: Input camera
            mode: Virtual camera sampling mode
            
        Returns:
            Averaged Gaussian parameters from multiple virtual views
        """
        try:
            # Generate virtual cameras using DriveStudio camera optimizer
            virtual_cameras = self._generate_virtual_cameras(cam, mode)
            
            if len(virtual_cameras) == 1:
                # No virtual cameras generated, use standard rendering
                return super().get_gaussians(virtual_cameras[0])
            
            # Render each virtual view
            virtual_renders = []
            for virtual_cam in virtual_cameras:
                gs_dict = super().get_gaussians(virtual_cam)
                virtual_renders.append(gs_dict)
            
            # Average the virtual views for motion blur effect
            return self._average_virtual_views(virtual_renders)
            
        except Exception as e:
            logger.warning(f"Multi-view rendering failed: {e}, falling back to single view")
            return super().get_gaussians(cam)
    
    def _generate_virtual_cameras(self, cam: dataclass_camera, mode: TrajSamplingMode) -> List[dataclass_camera]:
        """
        Generate virtual cameras using DriveStudio camera optimizer.
        
        Args:
            cam: Input camera
            mode: Sampling mode
            
        Returns:
            List of virtual cameras
        """
        if self.camera_optimizer is None:
            return [cam]
            
        try:
            # Ensure camera has metadata with cam_idx
            if not hasattr(cam, 'metadata') or cam.metadata is None:
                cam.metadata = {}
            if 'cam_idx' not in cam.metadata:
                # Use a default cam_idx if not provided
                cam.metadata['cam_idx'] = 0
                
            # Apply camera optimization using DriveStudio optimizer
            virtual_cameras = self.camera_optimizer.apply_to_camera(cam, mode)
            return virtual_cameras if virtual_cameras else [cam]
        except Exception as e:
            logger.warning(f"Virtual camera generation failed: {e}")
            return [cam]
    
    def _average_virtual_views(self, virtual_renders: List[Dict]) -> Dict:
        """
        Average multiple virtual view renders for motion blur effect.
        
        Args:
            virtual_renders: List of rendered Gaussian dictionaries
            
        Returns:
            Averaged Gaussian parameters
        """
        if not virtual_renders:
            raise ValueError("No virtual renders provided")
            
        if len(virtual_renders) == 1:
            return virtual_renders[0]
        
        # Initialize averaged dictionary with first render
        averaged_dict = copy.deepcopy(virtual_renders[0])
        
        # Average numerical parameters
        for key in ['_means', '_opacities', '_rgbs', '_scales', '_quats']:
            if key in averaged_dict:
                # Stack all renders for this parameter
                stacked_params = torch.stack([render[key] for render in virtual_renders if key in render])
                # Average across virtual views
                averaged_dict[key] = stacked_params.mean(dim=0)
        
        return averaged_dict
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get parameter groups including camera optimizer parameters"""
        param_groups = super().get_param_groups()
        
        # Add camera optimizer parameters if available
        if (self.camera_optimizer is not None and 
            hasattr(self.camera_optimizer, 'pose_adjustment') and
            self.camera_optimizer.pose_adjustment is not None):
            param_groups[self.class_prefix + "camera_opt"] = [self.camera_optimizer.pose_adjustment]
            
        return param_groups
    
    def compute_reg_loss(self):
        """Compute regularization losses including TV loss"""
        loss_dict = super().compute_reg_loss()
        
        # Add TV loss if enabled in config
        tv_loss_cfg = self.reg_cfg.get("tv_loss", None) if self.reg_cfg else None
        if tv_loss_cfg is not None and self._last_rendered_rgb is not None:
            try:
                # RGB is in format (H, W, 3), convert to (1, 3, H, W)
                rgb_tensor = self._last_rendered_rgb.permute(2, 0, 1).unsqueeze(0)
                tv_loss_value = self.tv_loss(rgb_tensor, mean=True)
                loss_dict["tv_loss"] = tv_loss_value * tv_loss_cfg.w
            except Exception as e:
                logger.warning(f"TV loss computation failed: {e}")
        
        # Add camera optimizer losses
        if self.camera_optimizer is not None:
            try:
                self.camera_optimizer.get_loss_dict(loss_dict)
            except Exception as e:
                logger.warning(f"Camera optimizer loss computation failed: {e}")
                
        return loss_dict
    
    def set_last_rendered_rgb(self, rgb: torch.Tensor):
        """Store last rendered RGB for TV loss computation"""
        self._last_rendered_rgb = rgb
        
    def get_camera_optimizer_metrics(self) -> Dict[str, float]:
        """Get camera optimizer metrics"""
        if self.camera_optimizer is None:
            return {}
            
        try:
            metrics_dict = {}
            self.camera_optimizer.get_metrics_dict(metrics_dict)
            return metrics_dict
        except Exception as e:
            logger.warning(f"Camera optimizer metrics computation failed: {e}")
            return {}
    
    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        """Post-processing after each training step including camera optimizer updates"""
        # Call parent postprocessing
        super().postprocess_per_train_step(step, optimizer, radii, xys_grad, last_size)
        
        # Update camera optimizer step if needed
        if self.camera_optimizer is not None:
            # Camera optimizer doesn't need special postprocessing in current implementation
            pass
    
    def get_virtual_view_count(self) -> int:
        """Get the number of virtual views configured"""
        if self.camera_optimizer is None:
            return 1
        return self.camera_optimizer.config.num_virtual_views
    
    def set_camera_optimizer_mode(self, mode: str):
        """Dynamically change camera optimizer mode"""
        if self.camera_optimizer is not None:
            self.camera_optimizer.config.mode = mode
            logger.info(f"Camera optimizer mode changed to: {mode}")
    
    def is_motion_blur_enabled(self) -> bool:
        """Check if motion blur (virtual views) is currently enabled"""
        return (
            self.camera_optimizer is not None and 
            self.camera_optimizer.config.mode != "off" and 
            self._is_training
        )
