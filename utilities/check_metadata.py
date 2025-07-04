#!/usr/bin/env python3
"""
Script to check PNG metadata in generated images.
This helps verify that metadata is being properly embedded.
"""

import sys
import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json

def check_metadata(image_path):
    """Check and display metadata from a PNG image."""
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return False
    
    try:
        with Image.open(image_path) as img:
            print(f"📁 File: {image_path}")
            print(f"📐 Size: {img.size[0]}x{img.size[1]}")
            print(f"🎨 Mode: {img.mode}")
            print(f"📊 Format: {img.format}")
            
            # Check if it's a PNG
            if img.format != 'PNG':
                print(f"⚠️  Warning: File is {img.format}, not PNG. Metadata may not be preserved.")
            
            # Get PNG info
            if hasattr(img, 'text') and img.text:
                print(f"\n🔍 Found {len(img.text)} metadata fields:")
                print("=" * 50)
                
                # Show individual fields
                for key, value in img.text.items():
                    if key == 'omnigen2_params':
                        print(f"📋 {key}:")
                        try:
                            parsed = json.loads(value)
                            for param_key, param_value in parsed.items():
                                print(f"   {param_key}: {param_value}")
                        except json.JSONDecodeError:
                            print(f"   {value}")
                    else:
                        print(f"📝 {key}: {value}")
                
                print("=" * 50)
                return True
            else:
                print("\n❌ No metadata found in this image!")
                print("Possible causes:")
                print("1. Image was saved without metadata")
                print("2. Image was processed by software that strips metadata")
                print("3. Image format doesn't support metadata")
                print("4. Metadata was corrupted")
                return False
                
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_metadata.py <image_path>")
        print("\nExample:")
        print("python check_metadata.py outputs_gradio/20250625/2025_06_25-12_34_56.png")
        
        # Try to find recent images automatically
        print("\n🔍 Looking for recent images...")
        outputs_dir = "outputs_gradio"
        if os.path.exists(outputs_dir):
            # Find the most recent date directory
            date_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
            if date_dirs:
                latest_date = sorted(date_dirs)[-1]
                latest_dir = os.path.join(outputs_dir, latest_date)
                png_files = [f for f in os.listdir(latest_dir) if f.endswith('.png')]
                if png_files:
                    print(f"Found {len(png_files)} PNG files in {latest_dir}")
                    print("Recent files:")
                    for i, f in enumerate(sorted(png_files)[-5:]):  # Show last 5
                        print(f"  {i+1}. {f}")
                    print(f"\nTry: python check_metadata.py \"{os.path.join(latest_dir, png_files[-1])}\"")
        return
    
    image_path = sys.argv[1]
    success = check_metadata(image_path)
    
    if success:
        print("\n✅ Metadata check completed successfully!")
    else:
        print("\n❌ Metadata check failed!")
        print("\nDebugging tips:")
        print("1. Make sure the image was generated by OmniGen2 (not downloaded/copied)")
        print("2. Check that the image is saved as PNG format")
        print("3. Verify the image wasn't processed by other software")
        print("4. Try generating a new image and check it immediately")

if __name__ == "__main__":
    main() 