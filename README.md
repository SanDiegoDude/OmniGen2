# ⚙️ OmniGen2: Advanced UI

A professional-grade Gradio interface for OmniGen2, the unified multimodal image generation model. This Advanced UI provides comprehensive controls, intelligent parameter management, and workflow optimization tools for serious image generation work.

![image](https://github.com/user-attachments/assets/3342a85e-1e79-4d6a-8dd7-9e0c46f73687)


**🔗 Original Project**: This UI is a fork of [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2). Visit the original repository for detailed information about OmniGen2's development, training, research papers, and core capabilities.

## 🎯 About This UI

This is an **Advanced Gradio Interface** for OmniGen2, featuring a comprehensive set of tools and controls for professional image generation workflows. Built on top of the powerful OmniGen2 model, this UI provides an intuitive yet feature-rich experience for:

- **Text-to-Image Generation** with advanced parameter control
- **Instruction-guided Image Editing** with precision tools
- **In-context Generation** with multi-image input support
- **Professional Workflow Features** including seed management, aspect ratio controls, and batch generation

## ✨ Advanced UI Features

### 🎛️ **Professional Controls**
- **Smart Aspect Ratio System**: Pre-configured ratios (1:1, 4:3, 16:9, etc.) with custom multipliers
- **Intelligent Megapixel Management**: Lock dimensions to megapixel values with real-time calculations
- **Advanced Seed Control**: Proper randomization with seed reproduction capabilities
- **CFG Range Control**: Fine-tune guidance application across generation steps
- **Dual Guidance Scales**: Separate text and image guidance with extended ranges (up to 8.0)

### 🖼️ **Image Management**
- **Multi-Image Input**: Support for up to 3 reference images
- **Smart Image Processing**: Automatic resizing with aspect ratio preservation
- **Generation History**: Detailed parameter tracking for each generation
- **Clickable Seed Links**: One-click seed copying for easy reproduction

### ⚡ **Performance & Efficiency**
- **Load-on-Demand**: Models load only when needed to save memory
- **Automatic Memory Management**: Pipeline automatically unloads when not in use
- **Progress Tracking**: Real-time generation progress with step counts
- **Command-line CPU Offloading**: Available via launch arguments for lower VRAM systems

### 🔐 **AI Content Identification**
- **Invisible Watermarking**: Silently applies invisible watermarks to identify AI-generated content
- **Industry Standard**: Uses the same watermarking technology as major AI image generators
- **Robust Detection**: Watermarks survive common image manipulations like compression and resizing
- **Truly Invisible**: No UI notifications or visual indicators - completely transparent to users

### 🎨 **User Experience**
- **Intuitive Layout**: Organized controls with logical grouping
- **Real-time Feedback**: Live parameter calculations and previews
- **Professional Tooltips**: Detailed guidance for each parameter
- **Responsive Design**: Optimized for various screen sizes

## 🚀 Quick Start

### 🛠️ Environment Setup

#### ✅ Recommended Setup

```bash
# 1. Clone the repo
git clone git@github.com:VectorSpaceLab/OmniGen2.git
cd OmniGen2

# 2. (Optional) Create a clean Python environment
conda create -n omnigen2 python=3.11
conda activate omnigen2

# 3. Install dependencies
# 3.1 Install PyTorch (choose correct CUDA version)
pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# 3.2 Install other required packages
pip install -r requirements.txt

# Note: Version 2.7.4.post1 is specified for compatibility with CUDA 12.4.
# Feel free to use a newer version if you use CUDA 12.6 or they fixed this compatibility issue.
# OmniGen2 runs even without flash-attn, though we recommend install it for best performance.
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

#### 🌏 For users in Mainland China

```bash
# Install PyTorch from a domestic mirror
pip install torch==2.6.0 torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu124

# Install other dependencies from Tsinghua mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Note: Version 2.7.4.post1 is specified for compatibility with CUDA 12.4.
# Feel free to use a newer version if you use CUDA 12.6 or they fixed this compatibility issue.
# OmniGen2 runs even without flash-attn, though we recommend install it for best performance.
pip install flash-attn==2.7.4.post1 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 🎨 Launch the Advanced UI

```bash
# Basic launch
python app.py

# With load-on-demand (recommended for limited VRAM)
python app.py --lod

# With CPU offloading for lower VRAM systems
python app.py --lod --enable_model_cpu_offload

# With aggressive CPU offloading (minimal VRAM usage)
python app.py --lod --enable_sequential_cpu_offload

# With API server (for programmatic access)
python app.py --api

# With public sharing
python app.py --share

# Custom port
python app.py --port 8080

# Disable invisible watermarking (not recommended)
python app.py --disable_watermark

# All options combined
python app.py --lod --api --port 8080
```

## 💡 Advanced Usage Tips

### 🎯 **Parameter Optimization**

- **Text Guidance Scale (1.0-8.0)**: Higher values make the model follow your prompt more strictly
  - *Recommended*: 3.0-6.0 for most use cases
  - *Lower (1.0-3.0)*: More creative, less prompt adherence
  - *Higher (6.0-8.0)*: Strict prompt following, may reduce creativity

- **Image Guidance Scale (1.0-8.0)**: Controls reference image influence
  - *For Editing*: 1.2-2.0 (allows text prompt to modify the image)
  - *For Style Transfer*: 2.5-4.0 (maintains image structure)
  - *For Exact Reproduction*: 5.0-8.0 (minimal changes to reference)

- **CFG Range (0.0-1.0)**: Define when guidance is applied during generation
  - *Start*: When to begin applying guidance (0.0 = from beginning)
  - *End*: When to stop applying guidance (1.0 = until end)
  - *Tip*: Reducing end value (e.g., 0.8) can speed up generation with minimal quality loss

### 🖼️ **Multi-Image Workflows**

1. **Reference + Edit**: Use first image as base, describe changes in prompt
2. **Style Transfer**: Reference image for style, prompt for content modifications
3. **Object Insertion**: Multiple references for different elements to combine
4. **Scene Composition**: Build complex scenes using multiple reference images

### ⚙️ **Memory Management**

- **Load-on-Demand (`--lod`)**: Essential for systems with <20GB VRAM
- **CPU Offloading (Command-line)**: Enable with `--enable_model_cpu_offload` or `--enable_sequential_cpu_offload`
- **Batch Generation**: Use multiple images per prompt efficiently
- **Resolution Management**: Use megapixel limits to control memory usage

### 🔐 **AI Content Watermarking**

This Advanced UI includes **invisible watermarking** to help identify AI-generated content, promoting responsible AI use and content authenticity.

#### **What is Invisible Watermarking?**
- **Completely Invisible**: Watermarks are imperceptible to humans but detectable by algorithms
- **Robust**: Survives common image manipulations (compression, resizing, cropping)
- **Industry Standard**: Uses the same technology as Stable Diffusion and other major AI generators
- **Privacy-Safe**: No tracking or personal information embedded

#### **How It Works**
- Silently embeds "OmniGen2-AI" watermark in all generated images
- Watermark is added to image frequency domain (invisible to human eye)
- Can be detected using compatible detection tools (see `utilities/` folder)
- Adds negligible processing time (~50ms per image)

#### **Disabling Watermarks**
While we recommend keeping watermarks enabled for responsible AI use, you can disable them:

```bash
# Disable watermarking
python app.py --disable_watermark

# API requests can also disable watermarking per request
curl -X POST "http://localhost:7551/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat", "disable_watermark": true}'
```

#### **Technical Details**
- **Library**: Uses `invisible-watermark` (MIT licensed)
- **Method**: DWT-DCT frequency domain embedding
- **Watermark Text**: "OmniGen2-AI" (11 characters)
- **Detection**: Requires compatible decoder with correct text length

#### **Utilities for Watermark Detection**
The `utilities/` folder contains helpful scripts:

```bash
# Check if an image has metadata and watermarks
python utilities/check_metadata.py path/to/your/image.png

# Test watermarking functionality
python utilities/test_watermark.py
```

**Note**: Windows doesn't show PNG text metadata in file properties - use the utilities above to verify watermarks.

## 💻 System Requirements

- **GPU**: NVIDIA RTX 3090 or equivalent (17GB+ VRAM recommended)
- **RAM**: 32GB+ system RAM (especially with CPU offloading)
- **Storage**: 50GB+ free space for models
- **Python**: 3.11+ recommended

### 🔧 **Performance Optimization**

| VRAM Available | Recommended Settings |
|----------------|---------------------|
| 24GB+ | `python app.py --lod` |
| 16-24GB | `python app.py --lod --enable_model_cpu_offload` |
| 12-16GB | `python app.py --lod --enable_sequential_cpu_offload` |
| <12GB | Use demo scripts in `/demo` folder |

## 📁 Project Structure

```
OmniGen2/
├── app.py                 # 🎯 Main Advanced UI (THIS FILE)
├── requirements.txt       # 📦 Dependencies
├── LICENSE               # 📄 License information
├── omnigen2/            # 🧠 Core model code
├── assets/              # 🖼️ UI assets and examples
├── utilities/           # 🔧 Utility scripts
│   ├── check_metadata.py # Check PNG metadata in generated images
│   └── test_watermark.py # Test watermarking functionality
└── demo/               # 📚 Example scripts and basic demos
    ├── app_basic.py    # Simple UI version
    ├── app_chat.py     # Chat-based interface
    ├── inference.py    # Command-line inference
    └── example_*.sh    # Usage examples
```

## ❤️ Citing the Original Work

This UI is built on top of OmniGen2. If you use this work, please cite the original paper:

```bibtex
@article{xiao2024omnigen,
  title={Omnigen: Unified image generation},
  author={Xiao, Shitao and Wang, Yueze and Zhou, Junjie and Yuan, Huaying and Xing, Xingrun and Yan, Ruiran and Wang, Shuting and Huang, Tiejun and Liu, Zheng},
  journal={arXiv preprint arXiv:2409.11340},
  year={2024}
}
```

## 📄 License

This work is licensed under Apache 2.0 license. The original OmniGen2 model and code are also licensed under Apache 2.0.

---

<p align="center">
  <em>Built with ❤️ for the AI community</em>
</p>
