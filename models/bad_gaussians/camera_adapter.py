"""
Camera adapter for BAD-Gaussians integration with drivestudio.

This module provides adapters to convert between drivestudio's dataclass_camera
and nerfstudio's Cameras class used by BAD-Gaussians.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List
from copy import deepcopy

try:
    from nerfstudio.cameras.cameras import Cameras, CameraType
except ImportError:
    # Fallback if nerfstudio is not available
    Cameras = None
    CameraType = None

from models.gaussians.basics import dataclass_camera

class CameraAdapter:
    """
    Adapter class to convert between drivestudio and nerfstudio camera formats.
    """
    
    @staticmethod
    def dataclass_to_nerfstudio(cam: dataclass_camera, cam_idx: int = 0) -> Optional[object]:
        """
        Convert drivestudio dataclass_camera to nerfstudio Cameras.
        
        Args:
            cam: drivestudio camera dataclass
            cam_idx: camera index for metadata
            
        Returns:
            nerfstudio Cameras object or None if conversion fails
        """
        if Cameras is None:
            return None
            
        try:
            # Extract camera parameters
            c2w = cam.camtoworlds  # [4, 4] or [1, 4, 4]
            K = cam.Ks  # [3, 3] or [1, 3, 3]
            H, W = cam.H, cam.W
            
            # Ensure proper shape
            if c2w.dim() == 2:
                c2w = c2w.unsqueeze(0)  # [1, 4, 4]
            if K.dim() == 2:
                K = K.unsqueeze(0)  # [1, 3, 3]
                
            # Extract intrinsics
            fx = K[0, 0, 0]
            fy = K[0, 1, 1]
            cx = K[0, 0, 2]
            cy = K[0, 1, 2]
            
            # Create nerfstudio camera
            cameras = Cameras(
                camera_to_worlds=c2w,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                height=H,
                width=W,
                distortion_params=None,
                camera_type=CameraType.PERSPECTIVE if CameraType else None,
                metadata={"cam_idx": cam_idx}
            )
            
            return cameras
            
        except Exception as e:
            print(f"Failed to convert camera: {e}")
            return None
    
    @staticmethod
    def nerfstudio_to_dataclass(cameras: object) -> Optional[dataclass_camera]:
        """
        Convert nerfstudio Cameras to drivestudio dataclass_camera.
        
        Args:
            cameras: nerfstudio Cameras object
            
        Returns:
            drivestudio dataclass_camera or None if conversion fails
        """
        if cameras is None:
            return None
            
        try:
            # Extract camera parameters
            c2w = cameras.camera_to_worlds  # [1, 4, 4] or [4, 4]
            fx = cameras.fx.item() if hasattr(cameras.fx, 'item') else cameras.fx
            fy = cameras.fy.item() if hasattr(cameras.fy, 'item') else cameras.fy
            cx = cameras.cx.item() if hasattr(cameras.cx, 'item') else cameras.cx
            cy = cameras.cy.item() if hasattr(cameras.cy, 'item') else cameras.cy
            H = cameras.height.item() if hasattr(cameras.height, 'item') else cameras.height
            W = cameras.width.item() if hasattr(cameras.width, 'item') else cameras.width
            
            # Build intrinsics matrix
            K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=c2w.device, dtype=c2w.dtype)
            
            # Ensure proper shape
            if c2w.dim() == 3:
                c2w = c2w.squeeze(0)  # [4, 4]
                
            # Create dataclass camera
            cam = dataclass_camera(
                camtoworlds=c2w,
                camtoworlds_gt=c2w.clone(),
                Ks=K,
                H=int(H),
                W=int(W)
            )
            
            # Preserve metadata
            if hasattr(cameras, 'metadata') and cameras.metadata:
                cam.metadata = cameras.metadata
                
            return cam
            
        except Exception as e:
            print(f"Failed to convert camera: {e}")
            return None

class BADCameraOptimizerAdapter:
    """
    Adapter for BAD camera optimizer to work with drivestudio cameras.
    """
    
    def __init__(self, camera_optimizer):
        self.camera_optimizer = camera_optimizer
        
    def apply_to_camera(self, cam: dataclass_camera, mode: str) -> List[dataclass_camera]:
        """
        Apply camera optimization to drivestudio camera.
        
        Args:
            cam: drivestudio camera
            mode: sampling mode
            
        Returns:
            List of optimized drivestudio cameras
        """
        # Get camera index from metadata if available
        cam_idx = 0
        if hasattr(cam, 'metadata') and cam.metadata and 'cam_idx' in cam.metadata:
            cam_idx = cam.metadata['cam_idx']
        elif hasattr(cam, 'cam_idx'):
            cam_idx = cam.cam_idx
            
        # Convert to nerfstudio format
        ns_camera = CameraAdapter.dataclass_to_nerfstudio(cam, cam_idx)
        if ns_camera is None:
            return [cam]  # Fallback to original camera
            
        try:
            # Apply BAD camera optimization
            ns_cameras = self.camera_optimizer.apply_to_camera(ns_camera, mode)
            
            # Convert back to drivestudio format
            ds_cameras = []
            for ns_cam in ns_cameras:
                ds_cam = CameraAdapter.nerfstudio_to_dataclass(ns_cam)
                if ds_cam is not None:
                    # Preserve original metadata
                    if hasattr(cam, 'metadata'):
                        ds_cam.metadata = getattr(cam, 'metadata', {})
                    ds_cam.metadata = getattr(ds_cam, 'metadata', {})
                    ds_cam.metadata['cam_idx'] = cam_idx
                    ds_cameras.append(ds_cam)
                    
            return ds_cameras if ds_cameras else [cam]
            
        except Exception as e:
            print(f"Camera optimization failed: {e}")
            return [cam]
    
    def get_loss_dict(self, loss_dict: Dict[str, torch.Tensor]):
        """Get loss dictionary from camera optimizer"""
        if hasattr(self.camera_optimizer, 'get_loss_dict'):
            self.camera_optimizer.get_loss_dict(loss_dict)
            
    def get_metrics_dict(self, metrics_dict: Dict[str, torch.Tensor]):
        """Get metrics dictionary from camera optimizer"""
        if hasattr(self.camera_optimizer, 'get_metrics_dict'):
            self.camera_optimizer.get_metrics_dict(metrics_dict)
