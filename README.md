# ⚙️ OmniGen2: Advanced UI

A Gradio interface for OmniGen2 with comprehensive controls and workflow optimization features for image generation tasks.

![image](https://github.com/user-attachments/assets/3342a85e-1e79-4d6a-8dd7-9e0c46f73687)

**🔗 Original Project**: This UI is a fork of [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2). Visit the original repository for detailed information about OmniGen2's development, training, research papers, and core capabilities.

## 🎯 About This UI

This is an **Advanced Gradio Interface** for OmniGen2, providing tools and controls for image generation workflows. Built on top of the OmniGen2 model, this UI offers:

- **Text-to-Image Generation** with parameter control
- **Instruction-guided Image Editing** 
- **In-context Generation** with multi-image input support
- **Workflow Features** including seed management, aspect ratio controls, and batch generation

## ✨ Features

### 🎛️ **Generation Controls**
- **Aspect Ratio System**: Pre-configured ratios (1:1, 4:3, 16:9, etc.) with custom multipliers
- **Megapixel Management**: Lock dimensions to megapixel values with real-time calculations
- **Seed Control**: Randomization with seed reproduction capabilities
- **CFG Range Control**: Fine-tune guidance application across generation steps
- **Dual Guidance Scales**: Separate text and image guidance (up to 8.0)

### 🔧 **Advanced Settings**
- **DPM Solver Parameters**: Algorithm type (dpmsolver++, sde-dpmsolver++), solver type (midpoint, heun), and solver order (1-3)
- **Flow Scheduler Options**: Dynamic time shift for euler scheduler
- **Sequence Length Control**: Adjustable max sequence length for complex prompts
- **Use Karras Sigmas**: Alternative noise scheduling (when supported)

### 🛑 **Generation Control**
- **Hard Cancellation**: Cancel button that immediately stops generation and frees VRAM
- **Thread Interruption**: True process interruption, not just UI cancellation
- **Pipeline Management**: Automatic unloading (--lod mode) or reloading after cancellation
- **OOM Recovery**: Automatic out-of-memory error handling with VRAM cleanup

### 🖼️ **Image Management**
- **Multi-Image Input**: Support for up to 3 reference images
- **Image Processing**: Automatic resizing with aspect ratio preservation
- **Generation History**: Parameter tracking for each generation
- **Clickable Seed Links**: One-click seed copying for reproduction

### ⚡ **Performance & Memory**
- **Load-on-Demand**: Models load only when needed to save memory
- **Automatic Memory Management**: Pipeline automatically unloads when not in use
- **Progress Tracking**: Real-time generation progress
- **CPU Offloading**: Available via launch arguments for lower VRAM systems
- **Error Display**: Errors shown in UI instead of requiring console monitoring

### 🔐 **AI Content Identification**
- **Invisible Watermarking**: Applies invisible watermarks to identify AI-generated content
- **Detection Tools**: Built-in utilities for watermark verification
- **Metadata Storage**: Comprehensive PNG metadata with generation parameters

### 🎨 **User Interface**
- **Organized Layout**: Controls grouped logically with accordions
- **Real-time Feedback**: Live parameter calculations and previews
- **Error Handling**: Clear error messages displayed in the generation info box
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

# 3.3 Install flash attention for optimal performance
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

#### 🌏 For users in Mainland China

```bash
# Install PyTorch from a domestic mirror
pip install torch==2.6.0 torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu124

# Install other dependencies from Tsinghua mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install flash attention for optimal performance
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

# Disable invisible watermarking
python app.py --disable_watermark

# All options combined
python app.py --lod --api --port 8080
```

## 💡 Usage Tips

### 🎯 **Parameter Guidelines**

- **Text Guidance Scale (1.0-8.0)**: Higher values increase prompt adherence
  - *Lower (1.0-3.5)*: More creative interpretation
  - *Recommended*: 3.5-5.0 for most use cases, good balance of creativity and prompt adherenace
  - *Higher (5.0-8.0)*: CFG Burn zone, strict prompt following

- **Image Guidance Scale (1.0-8.0)**: Controls reference image influence
  - *For Editing*: 1.2-2.0 (Primarily text prompt focused, only minor image influence)
  - *For Style Transfer*: 2.5-5.0 (maintains better image structure with more reference back to input images but with less prompt coherence)
  - *For Close Reproduction*: 5.0-8.0 (Images will dominate output, will start to get rough past about 5.5 or so)

- **CFG Range (0.0-1.0)**: Define when guidance is applied during generation
  - *VERY CREATIVE CHANGES*: changing the start and/or end guidance can have massive changes on the style and visual layout of the output.
  - *Start*: When to begin applying guidance (0.0 = from beginning)
  - *End*: When to stop applying guidance (1.0 = until end)
  - *Tip*: Reducing end value (e.g., 0.8) can speed up generation

### 🔧 **Advanced Settings**

- **DPM Algorithm Type**: 
  - *dpmsolver++*: Standard algorithm
  - *sde-dpmsolver++*: Adds controlled randomness for varied outputs
  
- **DPM Solver Type**:
  - *midpoint*: Standard numerical integration
  - *heun*: Alternative integration method
  
- **DPM Solver Order**: Higher order (3) = more accurate but slower

### 🖼️ **Multi-Image Workflows**

1. **Reference + Edit**: Use first image as base, describe changes in prompt
2. **Style Transfer**: Reference image for style, prompt for content modifications
3. **Object Insertion**: Multiple references for different elements to combine
4. **Scene Composition**: Build complex scenes using multiple reference images

### ⚙️ **Memory Management**

- **Load-on-Demand (`--lod`)**: Recommended for systems with <20GB VRAM
- **CPU Offloading**: Enable with `--enable_model_cpu_offload` or `--enable_sequential_cpu_offload`
- **Cancellation**: Use Cancel button to immediately stop generation and free VRAM
- **Resolution Management**: Use megapixel limits to control memory usage

### 🛑 **Generation Control**

- **Cancel Button**: Immediately stops generation using thread interruption
- **Hard Interruption**: Actually stops the generation process, not just the UI
- **Automatic Recovery**: Pipeline automatically reloads after cancellation (non-lod mode)
- **Error Handling**: OOM and other errors are displayed in the generation info box

### 🔐 **AI Content Watermarking**

This UI includes **invisible watermarking** to help identify AI-generated content.

#### **What is Invisible Watermarking?**
- **Invisible**: Watermarks are imperceptible to humans but detectable by algorithms
- **Robust**: Survives common image manipulations (compression, resizing, cropping)
- **Standard**: Uses the same technology as other AI generators

#### **How It Works**
- Embeds "OmniGen2-AI" watermark in all generated images
- Watermark is added to image frequency domain 
- Can be detected using compatible detection tools (see `utilities/` folder)

#### **Disabling Watermarks**
```bash
# Disable watermarking
python app.py --disable_watermark

# API requests can also disable watermarking per request
curl -X POST "http://localhost:7551/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat", "disable_watermark": true}'
```

#### **Utilities for Watermark Detection**
The `utilities/` folder contains scripts:

```bash
# Check if an image has metadata and watermarks
python utilities/check_metadata.py path/to/your/image.png

# Test watermarking functionality
python utilities/test_watermark.py
```

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
├── app.py                 # 🎯 Main Advanced UI
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

## 🔧 API Usage

When launched with `--api`, the UI provides a REST API:

```bash
# Start with API
python app.py --api

# Basic generation request
curl -X POST "http://localhost:7551/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful landscape",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "text_guidance_scale": 5.0
  }'
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The invisible watermarking functionality uses the `invisible-watermark` library, which is also MIT licensed.

---

<p align="center">
  <em>Built with ❤️ for the AI community</em>
</p>
