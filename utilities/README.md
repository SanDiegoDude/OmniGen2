# 🔧 OmniGen2 Utilities

This folder contains utility scripts for working with OmniGen2's invisible watermarking system.

## 📋 Available Tools

### `check_metadata.py`
**Check PNG metadata and watermarks in generated images**

```bash
# Check a specific image
python check_metadata.py path/to/your/image.png

# Run without arguments to see recent images
python check_metadata.py
```

**Features:**
- Displays all PNG metadata fields
- Shows generation parameters
- Verifies watermark presence in metadata
- Automatically finds recent images if no path provided

### `test_watermark.py`
**Test watermarking functionality**

```bash
python test_watermark.py
```

**Features:**
- Creates a test image with watermark
- Attempts to detect the watermark
- Verifies the watermarking system is working correctly
- Useful for troubleshooting watermark issues

## 🔍 Why These Tools?

**Windows Limitation**: Windows File Explorer doesn't show PNG text metadata in the "Details" tab, so these utilities help you verify that:
- Your images have proper metadata
- Watermarks are being applied correctly
- Generation parameters are being saved

## 🎯 Usage Examples

```bash
# Check if your generated image has metadata
python check_metadata.py "outputs_gradio/20250625/your-image.png"

# Test if watermarking is working
python test_watermark.py

# Quick check of recent images
python check_metadata.py
```

## 🔐 About Watermarking

OmniGen2 uses invisible watermarking to identify AI-generated content:
- **Completely invisible** to humans
- **Robust** against common image manipulations
- **Industry standard** technology (same as Stable Diffusion)
- **Responsible AI** practice for content authenticity

The watermarking operates silently - no UI notifications or status indicators appear during normal use. 