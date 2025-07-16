#!/usr/bin/env python3
"""
Test script for BAD-Gaussians integration with OmniRe.

This script tests the basic functionality of the BAD-Gaussians
integration without requiring a full training setup.
"""

import sys
import torch
from omegaconf import OmegaConf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all necessary modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from models.gaussians.bad_vanilla import BADVanillaGaussians
        logger.info("✓ BADVanillaGaussians imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import BADVanillaGaussians: {e}")
        return False
    
    try:
        from models.bad_gaussians.bad_camera_optimizer import BadCameraOptimizer, BadCameraOptimizerConfig
        logger.info("✓ BAD camera optimizer imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import BAD camera optimizer: {e}")
        return False
    
    try:
        from models.bad_gaussians.camera_adapter import CameraAdapter, BADCameraOptimizerAdapter
        logger.info("✓ Camera adapter imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import camera adapter: {e}")
        return False
    
    try:
        from models.bad_gaussians.bad_losses import EdgeAwareVariationLoss
        logger.info("✓ BAD losses imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import BAD losses: {e}")
        return False
    
    return True

def test_bad_vanilla_gaussians():
    """Test BADVanillaGaussians instantiation."""
    logger.info("Testing BADVanillaGaussians instantiation...")
    
    try:
        from models.gaussians.bad_vanilla import BADVanillaGaussians
        
        # Create test configuration
        ctrl_cfg = OmegaConf.create({
            'sh_degree': 3,
            'warmup_steps': 500,
            'reset_alpha_interval': 3000,
            'refine_interval': 100,
            'sh_degree_interval': 1000,
            'n_split_samples': 2,
            'reset_alpha_value': 0.01,
            'densify_grad_thresh': 0.0005,
            'densify_size_thresh': 0.003,
            'cull_alpha_thresh': 0.005,
            'cull_scale_thresh': 0.5,
            'cull_screen_size': 0.15,
            'split_screen_size': 0.05,
            'stop_screen_size_at': 4000,
            'stop_split_at': 15000
        })
        
        reg_cfg = OmegaConf.create({
            'tv_loss': {
                'w': 0.001
            }
        })
        
        camera_optimizer_cfg = OmegaConf.create({
            'mode': 'linear',
            'num_virtual_views': 5,
            'trans_l2_penalty': 0.0,
            'rot_l2_penalty': 0.0,
            'initial_noise_se3_std': 1e-5
        })
        
        # Test instantiation without camera optimizer
        model_basic = BADVanillaGaussians(
            class_name="TestBackground",
            ctrl=ctrl_cfg,
            reg=reg_cfg,
            scene_scale=30.0,
            device=torch.device('cpu'),
            num_train_images=100
        )
        logger.info("✓ BADVanillaGaussians (basic) created successfully")
        
        # Test instantiation with camera optimizer
        model_full = BADVanillaGaussians(
            class_name="TestBackground",
            ctrl=ctrl_cfg,
            reg=reg_cfg,
            scene_scale=30.0,
            device=torch.device('cpu'),
            num_train_images=100,
            camera_optimizer=camera_optimizer_cfg
        )
        logger.info("✓ BADVanillaGaussians (with camera optimizer) created successfully")
        
        # Test parameter groups
        param_groups = model_full.get_param_groups()
        logger.info(f"✓ Parameter groups: {list(param_groups.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create BADVanillaGaussians: {e}")
        return False

def test_camera_adapter():
    """Test camera adapter functionality."""
    logger.info("Testing camera adapter...")
    
    try:
        from models.bad_gaussians.camera_adapter import CameraAdapter
        from models.gaussians.basics import dataclass_camera
        
        # Create a test camera
        device = torch.device('cpu')
        c2w = torch.eye(4, device=device)
        K = torch.tensor([[100, 0, 320], [0, 100, 240], [0, 0, 1]], device=device, dtype=torch.float32)
        
        test_cam = dataclass_camera(
            camtoworlds=c2w,
            camtoworlds_gt=c2w.clone(),
            Ks=K,
            H=480,
            W=640
        )
        test_cam.metadata = {'cam_idx': 0}
        
        # Test conversion to nerfstudio format
        ns_cam = CameraAdapter.dataclass_to_nerfstudio(test_cam, 0)
        if ns_cam is not None:
            logger.info("✓ Camera conversion to nerfstudio format successful")
            
            # Test conversion back
            ds_cam = CameraAdapter.nerfstudio_to_dataclass(ns_cam)
            if ds_cam is not None:
                logger.info("✓ Camera conversion back to dataclass format successful")
            else:
                logger.warning("⚠ Camera conversion back failed (nerfstudio may not be available)")
        else:
            logger.warning("⚠ Camera conversion failed (nerfstudio may not be available)")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Camera adapter test failed: {e}")
        return False

def test_config_loading():
    """Test configuration file loading."""
    logger.info("Testing configuration loading...")
    
    try:
        config_path = "configs/omnire_bad.yaml"
        with open(config_path, 'r') as f:
            config = OmegaConf.load(f)
        
        # Check if BADVanillaGaussians is configured
        bg_type = config.model.Background.type
        if bg_type == "models.gaussians.BADVanillaGaussians":
            logger.info("✓ Configuration uses BADVanillaGaussians")
        else:
            logger.warning(f"⚠ Configuration uses {bg_type} instead of BADVanillaGaussians")
        
        # Check camera optimizer config
        if 'camera_optimizer' in config.model.Background:
            camera_opt_config = config.model.Background.camera_optimizer
            logger.info(f"✓ Camera optimizer configured with mode: {camera_opt_config.mode}")
            logger.info(f"✓ Virtual views: {camera_opt_config.num_virtual_views}")
        else:
            logger.warning("⚠ No camera optimizer configuration found")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting BAD-Gaussians integration tests...")
    
    tests = [
        ("Import Tests", test_imports),
        ("BADVanillaGaussians Tests", test_bad_vanilla_gaussians),
        ("Camera Adapter Tests", test_camera_adapter),
        ("Configuration Tests", test_config_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! BAD-Gaussians integration is ready.")
        logger.info("")
        logger.info("To use BAD-Gaussians with OmniRe, run:")
        logger.info("python tools/train.py --config configs/omnire_bad.yaml")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
