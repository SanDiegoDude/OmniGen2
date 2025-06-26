#!/usr/bin/env python3
"""
Test script for invisible watermarking functionality.
This script tests the watermarking and detection functions.
"""

import os
import sys
from PIL import Image
import numpy as np

# Add the current directory to the path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import apply_watermark, detect_watermark, WATERMARK_AVAILABLE
    print(f"Watermark system available: {WATERMARK_AVAILABLE}")
    
    if not WATERMARK_AVAILABLE:
        print("Watermarking not available. Please install invisible-watermark:")
        print("pip install invisible-watermark")
        sys.exit(1)
    
    # Create a test image
    test_image = Image.new('RGB', (512, 512), color='red')
    print("Created test image (512x512 red)")
    
    # Apply watermark
    watermarked_image = apply_watermark(test_image, "OmniGen2-AI-Test")
    print("Applied watermark: 'OmniGen2-AI-Test'")
    
    # Save the watermarked image
    watermarked_image.save("test_watermarked.png")
    print("Saved watermarked image as 'test_watermarked.png'")
    
    # Try to detect the watermark
    detected = detect_watermark(watermarked_image, len("OmniGen2-AI-Test"))
    print(f"Detected watermark: '{detected}'")
    
    if detected and "OmniGen2-AI-Test" in detected:
        print("✅ Watermark test PASSED!")
    else:
        print("❌ Watermark test FAILED!")
        print(f"Expected: 'OmniGen2-AI-Test', Got: '{detected}'")
    
    # Clean up
    if os.path.exists("test_watermarked.png"):
        os.remove("test_watermarked.png")
        print("Cleaned up test file")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc() 