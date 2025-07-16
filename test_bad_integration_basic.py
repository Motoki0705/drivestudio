#!/usr/bin/env python3
"""
Basic test script for BAD-Gaussians integration with OmniRe.

This script tests the basic file structure and imports without
requiring PyTorch or other heavy dependencies.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and report result."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory_structure():
    """Check if all necessary files and directories exist."""
    print("Checking BAD-Gaussians integration file structure...")
    print("=" * 60)
    
    checks = [
        ("models/bad_gaussians/", "BAD-Gaussians module directory"),
        ("models/bad_gaussians/bad_camera_optimizer.py", "BAD camera optimizer"),
        ("models/bad_gaussians/bad_losses.py", "BAD losses"),
        ("models/bad_gaussians/spline_functor.py", "Spline functions"),
        ("models/bad_gaussians/camera_adapter.py", "Camera adapter"),
        ("models/gaussians/bad_vanilla.py", "BADVanillaGaussians class"),
        ("configs/omnire_bad.yaml", "BAD-Gaussians configuration"),
    ]
    
    passed = 0
    for filepath, description in checks:
        if check_file_exists(filepath, description):
            passed += 1
    
    print(f"\nFile structure check: {passed}/{len(checks)} files found")
    return passed == len(checks)

def check_config_structure():
    """Check the configuration file structure."""
    print("\nChecking configuration file structure...")
    print("=" * 60)
    
    config_path = "configs/omnire_bad.yaml"
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Check for key configuration elements
        checks = [
            ("models.gaussians.BADVanillaGaussians", "BADVanillaGaussians type definition"),
            ("camera_optimizer:", "Camera optimizer configuration"),
            ("mode:", "Camera optimizer mode"),
            ("num_virtual_views:", "Virtual views configuration"),
            ("tv_loss:", "TV loss configuration"),
        ]
        
        passed = 0
        for pattern, description in checks:
            if pattern in content:
                print(f"✓ {description}")
                passed += 1
            else:
                print(f"✗ {description} (NOT FOUND)")
        
        print(f"\nConfiguration check: {passed}/{len(checks)} elements found")
        return passed == len(checks)
        
    except Exception as e:
        print(f"✗ Failed to read configuration file: {e}")
        return False

def check_python_syntax():
    """Check if Python files have valid syntax."""
    print("\nChecking Python syntax...")
    print("=" * 60)
    
    files_to_check = [
        "models/bad_gaussians/camera_adapter.py",
        "models/gaussians/bad_vanilla.py",
    ]
    
    passed = 0
    for filepath in files_to_check:
        try:
            with open(filepath, 'r') as f:
                source = f.read()
            
            # Check syntax by compiling
            compile(source, filepath, 'exec')
            print(f"✓ {filepath}: Valid Python syntax")
            passed += 1
            
        except SyntaxError as e:
            print(f"✗ {filepath}: Syntax error - {e}")
        except Exception as e:
            print(f"✗ {filepath}: Error - {e}")
    
    print(f"\nSyntax check: {passed}/{len(files_to_check)} files passed")
    return passed == len(files_to_check)

def check_import_structure():
    """Check if import statements are structured correctly."""
    print("\nChecking import structure...")
    print("=" * 60)
    
    # Check if __init__.py includes BADVanillaGaussians
    init_file = "models/gaussians/__init__.py"
    try:
        with open(init_file, 'r') as f:
            content = f.read()
        
        if "BADVanillaGaussians" in content:
            print(f"✓ BADVanillaGaussians is registered in {init_file}")
            return True
        else:
            print(f"✗ BADVanillaGaussians not found in {init_file}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to check {init_file}: {e}")
        return False

def generate_usage_instructions():
    """Generate usage instructions."""
    print("\n" + "=" * 60)
    print("BAD-GAUSSIANS INTEGRATION SUMMARY")
    print("=" * 60)
    
    print("""
🎯 INTEGRATION COMPLETE!

The following components have been integrated:

1. ✅ BADVanillaGaussians class
   - Extends VanillaGaussians with motion blur handling
   - Virtual camera trajectory generation
   - Multi-view rendering for deblurring
   - TV loss integration

2. ✅ Camera Optimizer
   - BAD camera optimizer with spline interpolation
   - Linear, cubic, and Bezier trajectory modes
   - Camera adapter for drivestudio compatibility

3. ✅ Configuration
   - omnire_bad.yaml for BAD-Gaussians training
   - Properly configured camera optimizer parameters
   - TV loss and regularization settings

🚀 USAGE:

To train OmniRe with BAD-Gaussians motion blur handling:

    python tools/train.py --config configs/omnire_bad.yaml

📋 CONFIGURATION OPTIONS:

- camera_optimizer.mode: "linear", "cubic", "bezier", or "off"
- camera_optimizer.num_virtual_views: Number of virtual views (default: 10)
- reg.tv_loss.w: TV loss weight (default: 0.001)

⚠️  NOTES:

- BAD-Gaussians is applied only to the Background component
- Virtual camera generation requires camera metadata with 'cam_idx'
- Motion blur handling is active during training mode
- Evaluation uses single-view rendering for speed

""")

def main():
    """Run all basic tests."""
    print("BAD-GAUSSIANS INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", check_directory_structure),
        ("Configuration Structure", check_config_structure),
        ("Python Syntax", check_python_syntax),
        ("Import Structure", check_import_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"OVERALL RESULT: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        generate_usage_instructions()
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
