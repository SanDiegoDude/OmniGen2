import dotenv

dotenv.load_dotenv(override=True)

import gradio as gr

import os
import argparse
import random
from datetime import datetime
import threading
import time
import logging
import json

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from omnigen2.utils.img_util import create_collage

# Watermarking imports
try:
    from imwatermark import WatermarkEncoder
    import cv2
    import numpy as np
    from PIL import Image
    WATERMARK_AVAILABLE = True
except ImportError:
    WATERMARK_AVAILABLE = False
    print("⚠️  Invisible watermarking not available")
    print("   Install with: pip install invisible-watermark")

# Configure logging to suppress noisy warnings
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

def create_png_metadata(
    instruction, width, height, scheduler, num_inference_steps, negative_prompt, 
    text_guidance_scale, image_guidance_scale, cfg_range_start, cfg_range_end, 
    num_images_per_prompt, max_input_image_side_length, max_pixels, seed, 
    has_input_images=False, watermarked=False
):
    """Create rich PNG metadata with all generation parameters."""
    metadata = {
        "Software": "OmniGen2",
        "prompt": instruction,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "scheduler": scheduler,
        "num_inference_steps": num_inference_steps,
        "text_guidance_scale": text_guidance_scale,
        "image_guidance_scale": image_guidance_scale,
        "cfg_range_start": cfg_range_start,
        "cfg_range_end": cfg_range_end,
        "num_images_per_prompt": num_images_per_prompt,
        "max_input_image_side_length": max_input_image_side_length,
        "max_pixels": max_pixels,
        "seed": seed,
        "has_input_images": has_input_images,
        "watermarked": watermarked,
        "generation_timestamp": datetime.now().isoformat(),
        "model": "OmniGen2"
    }
    
    # Convert to PNG info format
    from PIL.PngImagePlugin import PngInfo
    png_info = PngInfo()
    
    # Add each parameter as a separate field
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
    
    # Also add a comprehensive JSON field for easy parsing
    png_info.add_text("omnigen2_params", json.dumps(metadata, indent=2))
    
    return png_info

# Custom logging filter to suppress specific warnings
class CustomLogFilter(logging.Filter):
    def filter(self, record):
        # Suppress specific warnings
        suppressed_messages = [
            "Keyword arguments {'trust_remote_code': True} are not expected",
            "Expected types for transformer:",
            "Using a slow image processor as",
            "use_fast` is unset and a slow processor was saved"
        ]
        return not any(msg in record.getMessage() for msg in suppressed_messages)

# Apply the filter to relevant loggers
for logger_name in ["diffusers", "transformers", "omnigen2"]:
    logger = logging.getLogger(logger_name)
    logger.addFilter(CustomLogFilter())

# Check for flash attention installation
def check_flash_attention():
    try:
        import flash_attn
        print("✅ Flash Attention detected - optimal performance enabled")
        return True
    except ImportError:
        print("⚠️  Flash Attention not detected")
        print("   For best performance, install with:")
        print("   pip install flash-attn==2.7.4.post1 --no-build-isolation")
        print("   (OmniGen2 will still work without it)")
        return False

# Run the check on startup
check_flash_attention()

# Watermarking functions
def apply_watermark(image, watermark_text="OmniGen2-AI", method="dwtDct"):
    """Apply invisible watermark to PIL image."""
    if not WATERMARK_AVAILABLE:
        return image
    
    try:
        # Convert PIL to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize watermark encoder
        encoder = WatermarkEncoder()
        encoder.set_watermark('bytes', watermark_text.encode('utf-8'))
        
        # Apply watermark
        watermarked_cv = encoder.encode(image_cv, method)
        
        # Convert back to PIL format (RGB)
        watermarked_pil = Image.fromarray(cv2.cvtColor(watermarked_cv, cv2.COLOR_BGR2RGB))
        
        return watermarked_pil
    except Exception as e:
        print(f"Warning: Failed to apply watermark: {e}")
        return image

def detect_watermark(image, watermark_length=32, method="dwtDct"):
    """Detect watermark in PIL image (for testing purposes)."""
    if not WATERMARK_AVAILABLE:
        return None
    
    try:
        from imwatermark import WatermarkDecoder
        
        # Convert PIL to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize watermark decoder
        decoder = WatermarkDecoder('bytes', watermark_length)
        
        # Extract watermark
        watermark = decoder.decode(image_cv, method)
        
        return watermark.decode('utf-8')
    except Exception as e:
        print(f"Warning: Failed to detect watermark: {e}")
        return None

# API imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import Optional, List
    import uvicorn
    import base64
    import io
    from PIL import Image
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")

NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

pipeline = None
accelerator = None
save_images = False
load_on_demand = False

def load_pipeline(accelerator, weight_dtype, args):
    print("Loading pipeline...")
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(accelerator.device)
    print("Pipeline loaded successfully.")
    return pipeline

def unload_pipeline():
    global pipeline
    if pipeline is not None:
        print("Unloading pipeline...")
        # Move to CPU and clear cache
        if hasattr(pipeline, 'to'):
            pipeline = pipeline.to('cpu')
        del pipeline
        pipeline = None
        torch.cuda.empty_cache()
        print("Pipeline unloaded.")

def ensure_pipeline_loaded(args):
    global pipeline, accelerator
    if pipeline is None:
        bf16 = True
        if accelerator is None:
            accelerator = Accelerator(mixed_precision="bf16" if bf16 else "no")
        weight_dtype = torch.bfloat16 if bf16 else torch.float32
        pipeline = load_pipeline(accelerator, weight_dtype, args)
    return pipeline

# API Models
if API_AVAILABLE:
    class GenerationRequest(BaseModel):
        prompt: str
        negative_prompt: Optional[str] = NEGATIVE_PROMPT
        width: int = 1024
        height: int = 1024
        num_inference_steps: int = 30
        text_guidance_scale: float = 5.0
        image_guidance_scale: float = 5.0
        cfg_range_start: float = 0.1
        cfg_range_end: float = 0.5
        num_images_per_prompt: int = 1
        max_input_image_side_length: int = 1024
        max_pixels_mp: float = 1.05
        seed: int = 0
        scheduler: str = "dpmsolver"
        align_res: bool = False
        input_images_b64: Optional[List[str]] = None
        disable_watermark: bool = False

    class GenerationResponse(BaseModel):
        success: bool
        images_b64: Optional[List[str]] = None
        error: Optional[str] = None
        seed_used: Optional[int] = None

def run(
    instruction,
    width_input,
    height_input,
    scheduler,
    num_inference_steps,
    image_input_1,
    image_input_2,
    image_input_3,
    negative_prompt,
    guidance_scale_input,
    img_guidance_scale_input,
    cfg_range_start,
    cfg_range_end,
    num_images_per_prompt,
    max_input_image_side_length,
    max_pixels,
    seed_input,
    save_images_enabled=True,
    progress=gr.Progress(),
    align_res=False,
    args=None,
):
    global load_on_demand
    
    try:
        # Load pipeline if needed
        if load_on_demand and args:
            ensure_pipeline_loaded(args)
        
        input_images = [image_input_1, image_input_2, image_input_3]
        input_images = [img for img in input_images if img is not None]

        if len(input_images) == 0:
            input_images = None

        # Handle seed randomization properly
        actual_seed = seed_input
        if seed_input == -1:
            actual_seed = random.randint(0, 2**31 - 1)

        generator = torch.Generator(device=accelerator.device).manual_seed(actual_seed)

        def progress_callback(cur_step, timesteps):
            if progress:
                frac = (cur_step + 1) / float(timesteps)
                progress(frac)

        if scheduler == 'euler':
            pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
        elif scheduler == 'dpmsolver':
            pipeline.scheduler = DPMSolverMultistepScheduler(
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                solver_order=2,
                prediction_type="flow_prediction",
            )

        results = pipeline(
            prompt=instruction,
            input_images=input_images,
            width=width_input,
            height=height_input,
            max_input_image_side_length=max_input_image_side_length,
            max_pixels=max_pixels,
            num_inference_steps=num_inference_steps,
            max_sequence_length=1024,
            text_guidance_scale=guidance_scale_input,
            image_guidance_scale=img_guidance_scale_input,
            cfg_range=(cfg_range_start, cfg_range_end),
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            output_type="pil",
            step_func=progress_callback,
            align_res=align_res,
        )

        if progress:
            progress(1.0)

        # Apply watermarking to generated images (unless disabled)
        watermarked_images = []
        if args and not args.disable_watermark and WATERMARK_AVAILABLE:
            for image in results.images:
                watermarked_image = apply_watermark(image, "OmniGen2-AI")
                watermarked_images.append(watermarked_image)
        else:
            watermarked_images = results.images
            if args and args.disable_watermark:
                print("🚫 Watermarking disabled by --disable_watermark flag")

        vis_images = [to_tensor(image) * 2 - 1 for image in watermarked_images]
        output_image = create_collage(vis_images)

        if save_images_enabled:
            # Create date-based directory structure: ./outputs_gradio/YYYYMMDD/
            date_str = datetime.now().strftime("%Y%m%d")
            output_dir = os.path.join(ROOT_DIR, "outputs_gradio", date_str)
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename: YYMMDD-SEED-###.png
            date_prefix = datetime.now().strftime("%y%m%d")
            base_filename = f"{date_prefix}-{actual_seed}"
            
            # Find next available number for this seed
            counter = 1
            while True:
                filename = f"{base_filename}-{counter:03d}.png"
                output_path = os.path.join(output_dir, filename)
                if not os.path.exists(output_path):
                    break
                counter += 1

            # Create PNG metadata with all generation parameters
            has_input_images = any([image_input_1, image_input_2, image_input_3])
            is_watermarked = args and not args.disable_watermark and WATERMARK_AVAILABLE
            png_info = create_png_metadata(
                instruction, width_input, height_input, scheduler, num_inference_steps, 
                negative_prompt, guidance_scale_input, img_guidance_scale_input, 
                cfg_range_start, cfg_range_end, num_images_per_prompt, 
                max_input_image_side_length, max_pixels, actual_seed, has_input_images, is_watermarked
            )

            # Save the collage image with metadata
            output_image.save(output_path, pnginfo=png_info)
            print(f"💾 Saved with metadata: {output_path}")

            # Save individual images if multiple generated
            if len(watermarked_images) > 1:
                for i, image in enumerate(watermarked_images):
                    individual_filename = f"{base_filename}-{counter:03d}_{i+1}.png"
                    individual_path = os.path.join(output_dir, individual_filename)
                    image.save(individual_path, pnginfo=png_info)
                    print(f"💾 Saved with metadata: {individual_path}")
        
        return output_image, watermarked_images, actual_seed
        
    finally:
        # Unload pipeline if in load-on-demand mode
        if load_on_demand:
            # Add a small delay to allow any pending operations to complete
            threading.Timer(2.0, unload_pipeline).start()

def main(args):
    # Gradio
    with gr.Blocks(title="⚙️ OmniGen2 UI", css="""
        /* Ensure uniform gray backgrounds */
        .gradio-group {
            background-color: var(--block-background-fill) !important;
            border: 1px solid var(--block-border-color) !important;
            border-radius: var(--block-radius) !important;
            padding: var(--block-padding) !important;
        }
        
        /* Remove any internal borders within groups */
        .gradio-group .gradio-container,
        .gradio-group .gradio-row,
        .gradio-group .gradio-column {
            border: none !important;
            background: transparent !important;
        }
        
        /* Remove borders from HTML elements within groups */
        .gradio-group .gradio-html,
        .gradio-group .gradio-html > div,
        .gradio-group .gradio-html > div > div {
            border: none !important;
            background: transparent !important;
            background-color: transparent !important;
        }
        
        /* Ensure MP display matches background */
        #max-pixels-display {
            border: none !important;
            background: transparent !important;
        }
        
        #max-pixels-display > div,
        #max-pixels-display div {
            border: none !important;
            background: transparent !important;
            background-color: transparent !important;
        }
        
        /* Force HTML content to fill parent and remove any container styling */
        .gradio-group .gradio-html {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Ensure consistent spacing */
        .gradio-group > * {
            margin-bottom: 0 !important;
        }
        
        /* Output image expansion after first generation */
        .output-image-expanded {
            height: 1024px !important;
            max-height: 1024px !important;
        }
        
        /* Gallery styling for better grid display */
        #output-display {
            margin-top: 16px;
        }
        
        #output-display .gallery-container {
            gap: 8px;
        }
        
        /* Cap individual gallery image sizes at 320px */
        #output-display .gallery-item {
            max-width: 320px !important;
            max-height: 320px !important;
        }
        
        #output-display .gallery-item img {
            max-width: 320px !important;
            max-height: 320px !important;
            object-fit: contain !important;
        }
        
        /* Single image in gallery - center and make larger */
        #output-display[data-testid="gallery"]:has(.gallery-item:only-child) {
            justify-content: center;
        }
        
        #output-display .gallery-item:only-child {
            max-width: 100%;
            width: 100%;
        }
        
        /* Hide gallery when no individual images */
        #output-display:empty {
            display: none;
        }
        
        /* Fix fullscreen image overlay - fully opaque black background */
        .gradio-container .image-container .image-frame .image-button {
            background-color: rgba(0, 0, 0, 1) !important;
        }
        
        /* Ensure fullscreen overlay is completely opaque */
        .gradio-container [data-testid="image"] .image-container .image-frame {
            background-color: rgba(0, 0, 0, 1) !important;
        }
        
        /* Target fullscreen modal overlay */
        .gradio-container .modal-backdrop,
        .gradio-container .image-modal,
        .gradio-container .image-overlay {
            background-color: rgba(0, 0, 0, 1) !important;
            backdrop-filter: none !important;
        }
        
        /* More specific targeting for image fullscreen */
        [data-testid="image"] .image-container .image-frame,
        [data-testid="image"] .image-button,
        .image-container .image-frame,
        .image-button {
            background-color: rgba(0, 0, 0, 1) !important;
        }
        
        /* Hide scrollbars in gallery */
        #output-display {
            overflow: hidden !important;
        }
        
        #output-display .gallery-container {
            overflow: hidden !important;
        }
        
        /* Remove scrollbars from gallery items */
        #output-display .gallery-item {
            overflow: hidden !important;
        }
        
        /* Hide scrollbars more comprehensively */
        #output-display::-webkit-scrollbar,
        #output-display .gallery-container::-webkit-scrollbar,
        #output-display .gallery-item::-webkit-scrollbar {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
        }
        
        /* Firefox scrollbar hiding */
        #output-display,
        #output-display .gallery-container,
        #output-display .gallery-item {
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }
    """) as demo:
        gr.Markdown(
            "# ⚙️ OmniGen2: Unified Image Generation Advanced UI"
        )
        with gr.Row():
            with gr.Column():
                # Move Generate button to the top
                generate_button = gr.Button("Generate Image")

                # text prompt
                instruction = gr.Textbox(
                    label='Enter your prompt. Input images are optional - use "first/second image" as reference if provided.',
                    placeholder="Type your prompt here...",
                    lines=2,
                    elem_id="prompt-box",
                )

                with gr.Row(equal_height=True):
                    # input images
                    image_input_1 = gr.Image(label="Input Image 1", type="pil", height=320, width=320, show_label=True, elem_id="input-image-1")
                    image_input_2 = gr.Image(label="Input Image 2", type="pil", height=320, width=320, show_label=True, elem_id="input-image-2")
                    image_input_3 = gr.Image(label="Input Image 3", type="pil", height=320, width=320, show_label=True, elem_id="input-image-3")

                # Negative prompt section - unified block
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=10):
                            gr.HTML("")  # Empty space for alignment
                        with gr.Column(scale=1, min_width=40):
                            reset_negative_btn = gr.Button("🔄", size="sm", elem_id="reset-negative-btn", variant="secondary")
                    negative_prompt = gr.Textbox(
                        label="Enter your negative prompt",
                        placeholder="Type your negative prompt here...",
                        value=NEGATIVE_PROMPT,
                        elem_id="negative-prompt-input"
                    )

                # Aspect ratio and dimensions
                with gr.Row(equal_height=True):
                    aspect_ratio_multiplier = gr.Slider(
                        label="Aspect Ratio Multiplier",
                        minimum=1.0,
                        maximum=4.0,
                        value=1.0,
                        step=0.1,
                        info="Multiply the default aspect ratio size by this factor."
                    )

                    aspect_ratio = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=["Use Image1 Aspect Ratio", "1:1 (Square)", "4:3 (Landscape)", "3:4 (Portrait)", "16:9 (Widescreen)", "9:16 (Portrait)", "21:9 (Ultrawide)", "9:21 (Portrait)", "Custom"],
                        value="1:1 (Square)",
                        info="Select aspect ratio or choose Custom to set dimensions manually."
                    )

                # Width/Height row (moved up above Max Pixels)
                with gr.Row(equal_height=True):
                    width_input = gr.Slider(
                        label="Width", minimum=256, maximum=4096, value=1024, step=128
                    )
                    height_input = gr.Slider(
                        label="Height", minimum=256, maximum=4096, value=1024, step=128
                    )

                # Max pixels section - unified block
                with gr.Group():
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=4):
                            max_pixels_mp = gr.Slider(
                                label="Max Pixels (Megapixels)",
                                minimum=0.1,
                                maximum=16.8,
                                value=1.05,
                                step=0.01,
                                elem_id="max-pixels-mp"
                            )
                            max_pixels_display = gr.HTML(
                                value="<div style='background: none; color: var(--body-text-color); padding: 8px; font-size: 14px; margin-top: 8px; border: none;'>1.05 MP (≈1024×1024 square)</div>",
                                elem_id="max-pixels-display",
                                show_label=False
                            )
                        
                        lock_to_wh = gr.Checkbox(
                            label="Lock to W/H",
                            value=True,
                            info="Lock megapixel value to current width/height dimensions",
                            scale=1
                        )

                with gr.Row(equal_height=True):
                    text_guidance_scale_input = gr.Slider(
                        label="Text Guidance Scale",
                        minimum=1.0,
                        maximum=8.0,
                        value=5.0,
                        step=0.1,
                    )

                    image_guidance_scale_input = gr.Slider(
                        label="Image Guidance Scale",
                        minimum=1.0,
                        maximum=8.0,
                        value=5.0,
                        step=0.1,
                    )
                with gr.Row(equal_height=True):
                    cfg_range_start = gr.Slider(
                        label="CFG Range Start",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.1,
                    )

                    cfg_range_end = gr.Slider(
                        label="CFG Range End",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                    )
                
                def adjust_end_slider(start_val, end_val):
                    return max(start_val, end_val)

                def adjust_start_slider(end_val, start_val):
                    return min(end_val, start_val)
                
                cfg_range_start.input(
                    fn=adjust_end_slider,
                    inputs=[cfg_range_start, cfg_range_end],
                    outputs=[cfg_range_end]
                )

                cfg_range_end.input(
                    fn=adjust_start_slider,
                    inputs=[cfg_range_end, cfg_range_start],
                    outputs=[cfg_range_start]
                )

                # Aspect ratio change handler
                def update_dimensions_from_aspect_and_multiplier(aspect_choice, multiplier_choice):
                    aspect_ratios = {
                        "1:1 (Square)": (1024, 1024),
                        "4:3 (Landscape)": (1024, 768),
                        "3:4 (Portrait)": (768, 1024),
                        "16:9 (Widescreen)": (1024, 576),
                        "9:16 (Portrait)": (576, 1024),
                        "21:9 (Ultrawide)": (1024, 439),
                        "9:21 (Portrait)": (439, 1024),
                        "Custom": (1024, 1024),
                        "Use Image1 Aspect Ratio": (1024, 1024),
                    }
                    multiplier = multiplier_choice
                    if aspect_choice in aspect_ratios and aspect_choice not in ["Custom", "Use Image1 Aspect Ratio"]:
                        base_width, base_height = aspect_ratios[aspect_choice]
                        width = min(int(base_width * multiplier), 4096)
                        height = min(int(base_height * multiplier), 4096)
                        return width, height
                    return 1024, 1024

                def calculate_megapixels_from_dimensions(width, height):
                    """Calculate megapixels from width and height."""
                    return (width * height) / 1_000_000

                def update_max_pixels_display(mp_value, width=None, height=None):
                    """Update the megapixel display with both square estimate and actual dimensions if provided."""
                    pixels = int(mp_value * 1_000_000)
                    square_side = int((pixels) ** 0.5)
                    
                    if width is not None and height is not None:
                        return f"<div style='background: none; color: var(--body-text-color); padding: 8px; font-size: 14px; margin-top: 8px; border: none;'>{mp_value:.2f} MP (≈{square_side}×{square_side} square) | Current: {width}×{height} = {calculate_megapixels_from_dimensions(width, height):.2f} MP</div>"
                    else:
                        return f"<div style='background: none; color: var(--body-text-color); padding: 8px; font-size: 14px; margin-top: 8px; border: none;'>{mp_value:.2f} MP (≈{square_side}×{square_side} square)</div>"

                def update_dimensions_and_megapixels(aspect_choice, multiplier_choice, lock_enabled, current_mp, current_width, current_height):
                    """Update dimensions and handle megapixel locking."""
                    # Get new dimensions from aspect ratio
                    new_width, new_height = update_dimensions_from_aspect_and_multiplier(aspect_choice, multiplier_choice)
                    
                    # If lock is enabled, update megapixels to match new dimensions
                    if lock_enabled:
                        new_mp = calculate_megapixels_from_dimensions(new_width, new_height)
                        new_mp = max(0.1, min(new_mp, 16.8))  # Clamp to slider bounds
                        return new_width, new_height, new_mp, update_max_pixels_display(new_mp, new_width, new_height)
                    else:
                        return new_width, new_height, current_mp, update_max_pixels_display(current_mp, new_width, new_height)

                def update_megapixels_from_manual_dimensions(width, height, lock_enabled, current_mp):
                    """Update megapixels when dimensions are manually changed."""
                    if lock_enabled:
                        new_mp = calculate_megapixels_from_dimensions(width, height)
                        new_mp = max(0.1, min(new_mp, 16.8))  # Clamp to slider bounds
                        return new_mp, update_max_pixels_display(new_mp, width, height)
                    else:
                        return current_mp, update_max_pixels_display(current_mp, width, height)

                # Update aspect ratio handlers
                aspect_ratio.change(
                    fn=update_dimensions_and_megapixels,
                    inputs=[aspect_ratio, aspect_ratio_multiplier, lock_to_wh, max_pixels_mp, width_input, height_input],
                    outputs=[width_input, height_input, max_pixels_mp, max_pixels_display]
                )
                aspect_ratio_multiplier.change(
                    fn=update_dimensions_and_megapixels,
                    inputs=[aspect_ratio, aspect_ratio_multiplier, lock_to_wh, max_pixels_mp, width_input, height_input],
                    outputs=[width_input, height_input, max_pixels_mp, max_pixels_display]
                )

                # Handle manual dimension changes
                width_input.change(
                    fn=update_megapixels_from_manual_dimensions,
                    inputs=[width_input, height_input, lock_to_wh, max_pixels_mp],
                    outputs=[max_pixels_mp, max_pixels_display]
                )
                height_input.change(
                    fn=update_megapixels_from_manual_dimensions,
                    inputs=[width_input, height_input, lock_to_wh, max_pixels_mp],
                    outputs=[max_pixels_mp, max_pixels_display]
                )

                # Handle megapixel slider changes
                max_pixels_mp.change(
                    fn=lambda mp_val, w, h: update_max_pixels_display(mp_val, w, h),
                    inputs=[max_pixels_mp, width_input, height_input],
                    outputs=[max_pixels_display]
                )

                # Handle lock checkbox changes
                lock_to_wh.change(
                    fn=lambda lock_enabled, w, h, current_mp: (
                        calculate_megapixels_from_dimensions(w, h) if lock_enabled else current_mp,
                        update_max_pixels_display(
                            calculate_megapixels_from_dimensions(w, h) if lock_enabled else current_mp, 
                            w, h
                        )
                    ),
                    inputs=[lock_to_wh, width_input, height_input, max_pixels_mp],
                    outputs=[max_pixels_mp, max_pixels_display]
                )

                # Reset negative prompt function
                def reset_negative_prompt():
                    return NEGATIVE_PROMPT
                
                reset_negative_btn.click(
                    fn=reset_negative_prompt,
                    outputs=[negative_prompt]
                )

                with gr.Row(equal_height=True):
                    scheduler_input = gr.Dropdown(
                        label="Scheduler",
                        choices=["euler", "dpmsolver"],
                        value="dpmsolver",
                        info="The scheduler to use for the model.",
                    )

                    num_inference_steps = gr.Slider(
                        label="Inference Steps", minimum=20, maximum=100, value=30, step=1
                    )
                with gr.Row(equal_height=True):
                    num_images_per_prompt = gr.Slider(
                        label="Number of images per prompt",
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1,
                    )

                    seed_input = gr.Slider(
                        label="Seed", minimum=-1, maximum=2147483647, value=-1, step=1,
                        elem_id="seed-slider"
                    )
                with gr.Row(equal_height=True):
                    max_input_image_side_length = gr.Slider(
                        label="max_input_image_side_length",
                        minimum=256,
                        maximum=2048,
                        value=1024,
                        step=256,
                    )

            with gr.Column():
                with gr.Column():
                    # Adaptive output - single image or gallery based on batch size
                    output_display = gr.Gallery(
                        label="Output Image", 
                        show_label=True, 
                        elem_id="output-display",
                        columns=2,
                        rows=2,
                        height=768,
                        show_download_button=True,
                        allow_preview=True
                    )
                    
                    # Generation info box
                    generation_info = gr.HTML(
                        label="Generation Details",
                        value="""
                        <div style='background-color: #222; color: #eee; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 13px; min-height: 60px; display: flex; align-items: center;'>No generation yet</div>
                        """
                    )
                    
                    global save_images
                    save_images = gr.Checkbox(label="Save generated images", value=True)

        global accelerator
        global pipeline
        global load_on_demand

        load_on_demand = args.lod

        if not load_on_demand:
            # Load pipeline immediately if not in load-on-demand mode
            bf16 = True
            accelerator = Accelerator(mixed_precision="bf16" if bf16 else "no")
            weight_dtype = torch.bfloat16 if bf16 else torch.float32
            pipeline = load_pipeline(accelerator, weight_dtype, args)
        else:
            print("Load-on-demand mode enabled. Models will be loaded when needed.")

        def update_generation_info(instruction, width, height, scheduler, steps, negative_prompt, 
                                 text_guidance, image_guidance, cfg_start, cfg_end, num_images, 
                                 max_side, max_pixels, seed, output_img):
            if output_img is None:
                return """
                <div style='background-color: #222; color: #eee; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 13px;'>No generation yet</div>
                """
            
            info_html = f"""
            <div style='background-color: #222; color: #eee; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 13px;'>
                <strong>Generation Parameters:</strong><br>
                <strong>Prompt:</strong> {instruction}<br>
                <strong>Negative Prompt:</strong> {negative_prompt}<br>
                <strong>Dimensions:</strong> {width} × {height}<br>
                <strong>Scheduler:</strong> {scheduler}<br>
                <strong>Steps:</strong> {steps}<br>
                <strong>Text Guidance:</strong> {text_guidance}<br>
                <strong>Image Guidance:</strong> {image_guidance}<br>
                <strong>CFG Range:</strong> {cfg_start} - {cfg_end}<br>
                <strong>Images:</strong> {num_images}<br>
                <strong>Max Side Length:</strong> {max_side}<br>
                <strong>Max Pixels:</strong> {max_pixels:,}<br>
                <strong>Seed:</strong> <span style='color: #4ea1ff; cursor: pointer; text-decoration: underline;' onclick='
                    const seedElement = document.getElementById("seed-slider");
                    if (seedElement) {{
                        const input = seedElement.querySelector("input[type=\\"range\\"]") || seedElement.querySelector("input");
                        if (input) {{
                            input.value = "{seed}";
                            input.dispatchEvent(new Event("input", {{ bubbles: true }}));
                            input.dispatchEvent(new Event("change", {{ bubbles: true }}));
                        }}
                    }}
                '>{seed}</span>
            </div>
            """
            return info_html

        def run_with_align_res(
            instruction,
            width_input,
            height_input,
            scheduler,
            num_inference_steps,
            image_input_1,
            image_input_2,
            image_input_3,
            negative_prompt,
            guidance_scale_input,
            img_guidance_scale_input,
            cfg_range_start,
            cfg_range_end,
            num_images_per_prompt,
            max_input_image_side_length,
            max_pixels_mp,
            seed_input,
            aspect_ratio_choice,
            save_images_enabled,
            progress=gr.Progress(),
        ):
            align_res = aspect_ratio_choice == "Use Image1 Aspect Ratio"
            max_pixels = int(max_pixels_mp * 1_000_000)
            result = run(
                instruction,
                width_input,
                height_input,
                scheduler,
                num_inference_steps,
                image_input_1,
                image_input_2,
                image_input_3,
                negative_prompt,
                guidance_scale_input,
                img_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels,
                seed_input,
                save_images_enabled,
                progress,
                align_res,
                args,
            )
            return result

        # Function to handle both generation and info update
        def generate_and_update_info(
            instruction,
            width_input,
            height_input,
            scheduler_input,
            num_inference_steps,
            image_input_1,
            image_input_2,
            image_input_3,
            negative_prompt,
            text_guidance_scale_input,
            image_guidance_scale_input,
            cfg_range_start,
            cfg_range_end,
            num_images_per_prompt,
            max_input_image_side_length,
            max_pixels_mp,
            seed_input,
            aspect_ratio,
            save_images_enabled,
            progress=gr.Progress(),
        ):
            # Generate the image and get the actual seed used
            output_image_result, all_images, actual_seed = run_with_align_res(
                instruction,
                width_input,
                height_input,
                scheduler_input,
                num_inference_steps,
                image_input_1,
                image_input_2,
                image_input_3,
                negative_prompt,
                text_guidance_scale_input,
                image_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels_mp,
                seed_input,
                aspect_ratio,
                save_images_enabled,
                progress,
            )
            
            # Generate the info using the actual seed
            info_html = update_generation_info(
                instruction, width_input, height_input, scheduler_input, num_inference_steps, negative_prompt, 
                text_guidance_scale_input, image_guidance_scale_input, cfg_range_start, cfg_range_end, num_images_per_prompt, 
                max_input_image_side_length, int(max_pixels_mp * 1_000_000), actual_seed, output_image_result
            )
            
            # Return individual images for gallery display
            return all_images, info_html

        # click
        generate_event = generate_button.click(
            generate_and_update_info,
            inputs=[
                instruction,
                width_input,
                height_input,
                scheduler_input,
                num_inference_steps,
                image_input_1,
                image_input_2,
                image_input_3,
                negative_prompt,
                text_guidance_scale_input,
                image_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels_mp,
                seed_input,
                aspect_ratio,
                save_images,
            ],
            outputs=[output_display, generation_info],
        )

        # Add Enter key handler for prompt box
        instruction.submit(
            generate_and_update_info,
            inputs=[
                instruction,
                width_input,
                height_input,
                scheduler_input,
                num_inference_steps,
                image_input_1,
                image_input_2,
                image_input_3,
                negative_prompt,
                text_guidance_scale_input,
                image_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels_mp,
                seed_input,
                aspect_ratio,
                save_images,
            ],
            outputs=[output_display, generation_info],
        )

    # launch
    demo.launch(share=args.share, server_port=args.port, server_name="0.0.0.0", allowed_paths=[ROOT_DIR])

# API Server Implementation
if API_AVAILABLE:
    def create_api_server(args):
        app = FastAPI(title="OmniGen2 API", version="1.0.0")
        
        def image_to_base64(image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        def base64_to_image(b64_string):
            image_data = base64.b64decode(b64_string)
            return Image.open(io.BytesIO(image_data))
        
        @app.post("/generate", response_model=GenerationResponse)
        async def generate_image(request: GenerationRequest):
            try:
                # Convert input images from base64 if provided
                input_images = [None, None, None]
                if request.input_images_b64:
                    for i, b64_img in enumerate(request.input_images_b64[:3]):
                        if b64_img:
                            input_images[i] = base64_to_image(b64_img)
                
                # Convert MP to pixels
                max_pixels = int(request.max_pixels_mp * 1_000_000)
                
                # Create args object for API request with watermark setting
                api_args = type('obj', (object,), {
                    'disable_watermark': request.disable_watermark,
                    'lod': args.lod if args else False,
                    'model_path': args.model_path if args else "OmniGen2/OmniGen2",
                    'enable_model_cpu_offload': args.enable_model_cpu_offload if args else False,
                    'enable_sequential_cpu_offload': args.enable_sequential_cpu_offload if args else False,
                })
                
                # Generate image
                result = run(
                    instruction=request.prompt,
                    width_input=request.width,
                    height_input=request.height,
                    scheduler=request.scheduler,
                    num_inference_steps=request.num_inference_steps,
                    image_input_1=input_images[0],
                    image_input_2=input_images[1],
                    image_input_3=input_images[2],
                    negative_prompt=request.negative_prompt,
                    guidance_scale_input=request.text_guidance_scale,
                    img_guidance_scale_input=request.image_guidance_scale,
                    cfg_range_start=request.cfg_range_start,
                    cfg_range_end=request.cfg_range_end,
                    num_images_per_prompt=request.num_images_per_prompt,
                    max_input_image_side_length=request.max_input_image_side_length,
                    max_pixels=max_pixels,
                    seed_input=request.seed,
                    save_images_enabled=False,  # Don't save for API requests
                    progress=None,
                    align_res=request.align_res,
                    args=api_args,
                )
                
                # Handle return values
                if isinstance(result, tuple):
                    output_image, all_images, seed_used = result
                    images_b64 = [image_to_base64(img) for img in all_images]
                else:
                    # Fallback for single image
                    images_b64 = [image_to_base64(result)]
                    seed_used = request.seed
                
                return GenerationResponse(
                    success=True,
                    images_b64=images_b64,
                    seed_used=seed_used
                )
                
            except Exception as e:
                return GenerationResponse(
                    success=False,
                    error=str(e)
                )
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "load_on_demand": load_on_demand}
        
        return app

def parse_args():
    parser = argparse.ArgumentParser(description="Run the OmniGen2")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app")
    parser.add_argument(
        "--port", type=int, default=7550, help="Port to use for the Gradio app"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="OmniGen2/OmniGen2",
        help="Path or HuggingFace name of the model to load."
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload."
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload."
    )
    parser.add_argument(
        "--lod",
        action="store_true",
        help="Load on demand - only load models when generating, then unload to save VRAM."
    )
    parser.add_argument(
        "--api",
        nargs="?",
        const=7551,
        type=int,
        help="Enable API server on specified port (default: 7551)"
    )
    parser.add_argument(
        "--disable_watermark",
        action="store_true",
        help="Disable invisible watermarking of generated images"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Start API server if requested
    if args.api is not None:
        if not API_AVAILABLE:
            print("ERROR: FastAPI not available. Install with: pip install fastapi uvicorn")
            exit(1)
        
        api_app = create_api_server(args)
        api_thread = threading.Thread(
            target=lambda: uvicorn.run(api_app, host="0.0.0.0", port=args.api, log_level="info"),
            daemon=True
        )
        api_thread.start()
        print(f"API server starting on http://0.0.0.0:{args.api}")
        time.sleep(1)  # Give API server time to start
    
    # Start Gradio interface
    main(args)
