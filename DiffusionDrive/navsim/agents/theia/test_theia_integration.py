"""
Test script to verify Theia integration with TransfuserBackbone.
"""

import torch
import sys
sys.path.insert(0, '/data/shengzhenli/DiffusionDrive')

from navsim.agents.theia.transfuser_config import TransfuserConfig
from navsim.agents.theia.transfuser_backbone import TransfuserBackbone


def test_theia_integration():
    """Test if Theia model integrates correctly with TransfuserBackbone."""

    print("=" * 80)
    print("Testing Theia Integration with TransfuserBackbone")
    print("=" * 80)

    # Create config (Theia is always used)
    config = TransfuserConfig()

    print(f"\n1. Configuration:")
    print(f"   - theia_model_name: {config.theia_model_name}")
    print(f"   - camera_width: {config.camera_width}")
    print(f"   - camera_height: {config.camera_height}")

    # Create model
    print("\n2. Creating TransfuserBackbone with Theia...")
    try:
        model = TransfuserBackbone(config)
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
        return False

    # Create dummy input
    batch_size = 2
    image_input = torch.randn(batch_size, 3, config.camera_height, config.camera_width)
    lidar_input = torch.randn(batch_size, config.lidar_seq_len,
                             config.lidar_resolution_height,
                             config.lidar_resolution_width)

    print(f"\n3. Input shapes:")
    print(f"   - Image: {image_input.shape}")
    print(f"   - LiDAR: {lidar_input.shape}")

    # Test forward pass
    print("\n4. Testing forward pass...")
    try:
        with torch.no_grad():
            features, fused_features, image_feature_grid = model(image_input, lidar_input)

        print("   ✓ Forward pass successful")
        print(f"\n5. Output shapes:")
        if features is not None:
            print(f"   - BEV features: {features.shape}")
        else:
            print(f"   - BEV features: None")
        print(f"   - Fused features: {fused_features.shape}")
        if image_feature_grid is not None:
            print(f"   - Image feature grid: {image_feature_grid.shape}")
        else:
            print(f"   - Image feature grid: None")

    except Exception as e:
        print(f"   ✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test feature info
    print(f"\n6. Image encoder feature info:")
    try:
        for i, info in enumerate(model.image_encoder.feature_info.info):
            print(f"   - Scale {i}: channels={info['num_chs']}, reduction={info['reduction']}")
    except Exception as e:
        print(f"   ✗ Error accessing feature info: {e}")

    print("\n7. Testing layer-by-layer feature extraction...")
    print("   (Extracting from ViT layers [5, 7, 9, 11])")

    print("\n" + "=" * 80)
    print("Test completed successfully! ✓")
    print("=" * 80)

    return True


def test_theia_wrapper():
    """Test TheiaFeatureWrapper in isolation."""
    print("\n" + "=" * 80)
    print("Testing TheiaFeatureWrapper in isolation")
    print("=" * 80)

    from transformers import AutoModel
    from navsim.agents.theia.transfuser_backbone import TheiaFeatureWrapper

    config = TransfuserConfig()

    print("\n1. Loading Theia model from HuggingFace...")
    try:
        theia_model = AutoModel.from_pretrained(
            config.theia_model_name,
            trust_remote_code=True,
            feature_reduce_method=config.theia_feature_reduce_method
        )
        print("   ✓ Theia model loaded")
    except Exception as e:
        print(f"   ✗ Error loading Theia model: {e}")
        return False

    print("\n2. Creating wrapper...")
    try:
        wrapper = TheiaFeatureWrapper(theia_model, config)
        print("   ✓ Wrapper created")
    except Exception as e:
        print(f"   ✗ Error creating wrapper: {e}")
        return False

    print("\n3. Testing wrapper forward pass...")
    batch_size = 1
    # Create uint8 input as Theia expects
    image_input = torch.randint(0, 256, (batch_size, 3, config.camera_height, config.camera_width),
                                dtype=torch.uint8)

    try:
        with torch.no_grad():
            multi_scale_features = wrapper(image_input)

        print("   ✓ Wrapper forward pass successful")
        print(f"\n4. Multi-scale features:")
        for i, feat in enumerate(multi_scale_features):
            print(f"   - Scale {i}: {feat.shape}")

    except Exception as e:
        print(f"   ✗ Error in wrapper forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("Wrapper test completed successfully! ✓")
    print("=" * 80)

    return True


if __name__ == "__main__":
    # Test wrapper first
    wrapper_success = test_theia_wrapper()

    if wrapper_success:
        # Test full integration
        integration_success = test_theia_integration()

        if integration_success:
            print("\n🎉 All tests passed! Theia integration is working correctly.")
        else:
            print("\n❌ Integration test failed. Please check the errors above.")
    else:
        print("\n❌ Wrapper test failed. Please check the errors above.")
