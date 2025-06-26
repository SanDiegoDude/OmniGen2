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
import gc
import traceback
import signal
import ctypes
import sys

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

# Global variables for cancellation control
generation_cancelled = False
generation_thread = None
cancel_event = threading.Event()
generation_result = None
generation_exception = None

class ThreadKilledException(Exception):
    """Exception raised when a thread is forcibly killed"""
    pass

def terminate_thread(thread):
    """Forcibly terminate a thread using ctypes (Windows/Linux compatible)"""
    if thread is None or not thread.is_alive():
        return False
    
    thread_id = thread.ident
    try:
        # This is a low-level way to raise an exception in another thread
        # It's not guaranteed to work in all cases, but it's the best we can do in Python
        if sys.platform == "win32":
            # Windows
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id), 
                ctypes.py_object(ThreadKilledException)
            )
        else:
            # Unix/Linux - use signal if possible
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id), 
                ctypes.py_object(ThreadKilledException)
            )
        
        if res == 0:
            print("⚠️ Thread termination failed - invalid thread ID")
            return False
        elif res != 1:
            # Clean up if we accidentally raised exception in multiple threads
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
            print("⚠️ Thread termination failed - multiple threads affected")
            return False
        
        print("🛑 Thread forcibly terminated")
        return True
    except Exception as e:
        print(f"⚠️ Failed to terminate thread: {e}")
        return False

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
    """Ensure pipeline is loaded for load-on-demand mode or after cancellation"""
    global pipeline, accelerator
    if pipeline is None:
        print("Loading pipeline for generation...")
        bf16 = True
        if accelerator is None:
            accelerator = Accelerator(mixed_precision="bf16" if bf16 else "no")
        weight_dtype = torch.bfloat16 if bf16 else torch.float32
        pipeline = load_pipeline(accelerator, weight_dtype, args)
        print("Pipeline loaded successfully.")

def reload_pipeline_if_needed(args):
    """Reload pipeline if it was unloaded due to cancellation"""
    global pipeline, accelerator, load_on_demand
    
    if not load_on_demand and pipeline is None:
        print("🔄 Reloading pipeline after cancellation...")
        try:
            bf16 = True
            if accelerator is None:
                accelerator = Accelerator(mixed_precision="bf16" if bf16 else "no")
            weight_dtype = torch.bfloat16 if bf16 else torch.float32
            pipeline = load_pipeline(accelerator, weight_dtype, args)
            print("✅ Pipeline reloaded successfully")
        except Exception as e:
            print(f"❌ Failed to reload pipeline: {e}")
            raise e
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
    # Advanced parameters
    rotary_theta=10000,
    max_sequence_length=256,
    dpm_algorithm_type="sde-dpmsolver++",
    dpm_solver_type="heun",
    dpm_solver_order=3,
    use_karras_sigmas=False,
    enable_dynamic_thresholding=False,
    dynamic_thresholding_ratio=0.95,
    enable_dynamic_time_shift=True,
    save_images_enabled=True,
    progress=gr.Progress(),
    align_res=False,
    args=None,
):
    global load_on_demand, generation_cancelled, cancel_event
    
    try:
        # Check for cancellation before starting
        if generation_cancelled or cancel_event.is_set():
            raise ThreadKilledException("Generation cancelled before starting")
        
        # Load pipeline if needed or reload if unloaded
        if load_on_demand and args:
            ensure_pipeline_loaded(args)
        elif args:
            # Try to reload pipeline if it was unloaded due to cancellation
            reload_pipeline_if_needed(args)
        
        input_images = [image_input_1, image_input_2, image_input_3]
        input_images = [img for img in input_images if img is not None]

        if len(input_images) == 0:
            input_images = None

        # Handle seed randomization properly
        actual_seed = seed_input
        if seed_input == -1:
            actual_seed = random.randint(0, 2**31 - 1)

        # Check for cancellation before generating
        if generation_cancelled or cancel_event.is_set():
            raise ThreadKilledException("Generation cancelled before torch.Generator creation")
        
        generator = torch.Generator(device=accelerator.device).manual_seed(actual_seed)

        def progress_callback(cur_step, timesteps):
            global generation_cancelled, cancel_event
            # Check for cancellation
            if cancel_event.is_set() or generation_cancelled:
                print("🛑 Generation cancelled during progress callback")
                # Raise exception to stop generation
                raise ThreadKilledException("Generation cancelled during progress")
            if progress:
                frac = (cur_step + 1) / float(timesteps)
                progress(frac)

        if scheduler == 'euler':
            # Try to use advanced parameters if supported
            try:
                pipeline.scheduler = FlowMatchEulerDiscreteScheduler(
                    dynamic_time_shift=enable_dynamic_time_shift
                )
            except TypeError:
                # Fallback to basic initialization if parameter not supported
                pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
                if enable_dynamic_time_shift:
                    print("Note: dynamic_time_shift not supported in this version of FlowMatchEulerDiscreteScheduler")
        elif scheduler == 'dpmsolver':
            # Build scheduler kwargs based on what's supported
            scheduler_kwargs = {
                "algorithm_type": dpm_algorithm_type,
                "solver_type": dpm_solver_type,
                "solver_order": dpm_solver_order,
                "prediction_type": "flow_prediction",
            }
            
            # Try to add advanced parameters if supported
            try:
                # Test if use_karras_sigmas is supported
                test_scheduler = DPMSolverMultistepScheduler(**scheduler_kwargs, use_karras_sigmas=False)
                scheduler_kwargs["use_karras_sigmas"] = use_karras_sigmas
            except TypeError:
                if use_karras_sigmas:
                    print("Note: use_karras_sigmas not supported in this version of DPMSolverMultistepScheduler")
            
            # Handle dynamic thresholding parameters
            if enable_dynamic_thresholding:
                try:
                    # In newer versions it's called 'thresholding'
                    scheduler_kwargs["thresholding"] = True
                    scheduler_kwargs["dynamic_thresholding_ratio"] = dynamic_thresholding_ratio
                except:
                    print("Note: dynamic thresholding not supported in this version of DPMSolverMultistepScheduler")
            
            pipeline.scheduler = DPMSolverMultistepScheduler(**scheduler_kwargs)

        # Final cancellation check before pipeline execution
        if generation_cancelled or cancel_event.is_set():
            raise ThreadKilledException("Generation cancelled before pipeline execution")

        results = pipeline(
            prompt=instruction,
            input_images=input_images,
            width=width_input,
            height=height_input,
            max_input_image_side_length=max_input_image_side_length,
            max_pixels=max_pixels,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            text_guidance_scale=guidance_scale_input,
            image_guidance_scale=img_guidance_scale_input,
            cfg_range=(cfg_range_start, cfg_range_end),
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            output_type="pil",
            step_func=progress_callback,
            align_res=align_res,
            # Note: rotary_theta would need to be set on the transformer model
            # during initialization, not passed to the pipeline call
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
        
    except ThreadKilledException as e:
        print(f"🛑 Generation forcibly cancelled: {e}")
        # Re-raise to be handled by the calling function
        raise e
    except Exception as e:
        print(f"❌ Unexpected error in generation: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise e
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
                with gr.Row():
                    generate_button = gr.Button("Generate Image", scale=4, variant="primary")
                    cancel_button = gr.Button("Cancel", scale=1, variant="secondary")

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

                # Generation Settings accordion (open by default)
                with gr.Accordion("Generation Settings", open=True):
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
                            label="Max Input Image Side Length",
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=256,
                            info="Maximum side length for input images before resizing"
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

                # Advanced Settings accordion (closed by default)
                with gr.Accordion("Advanced Settings", open=False):
                    gr.Markdown("*Note: Some parameters may not be supported depending on your diffusers version. Unsupported parameters will be ignored with a console message.*")
                    
                    with gr.Row(equal_height=True):
                        # Hide rotary_theta - keep as hidden component for compatibility
                        rotary_theta = gr.Slider(
                            label="Rotary Position Embedding Theta",
                            minimum=1000,
                            maximum=50000,
                            value=10000,
                            step=1000,
                            visible=False  # Hidden
                        )
                        
                        max_sequence_length = gr.Slider(
                            label="Max Sequence Length",
                            minimum=256,
                            maximum=1024,
                            value=256,
                            step=64,
                            info="Maximum sequence length for text encoding - longer for complex prompts"
                        )
                    
                    # DPMSolver specific settings
                    gr.Markdown("#### DPMSolver Advanced Parameters (when using dpmsolver scheduler)")
                    with gr.Row(equal_height=True):
                        dpm_algorithm_type = gr.Dropdown(
                            label="DPM Algorithm Type",
                            choices=["dpmsolver++", "sde-dpmsolver++"],
                            value="sde-dpmsolver++",
                            info="sde-dpmsolver++ adds controlled randomness for more varied outputs"
                        )
                        
                        dpm_solver_type = gr.Dropdown(
                            label="DPM Solver Type",
                            choices=["midpoint", "heun"],
                            value="heun",
                            info="Different numerical integration methods"
                        )
                    
                    with gr.Row(equal_height=True):
                        dpm_solver_order = gr.Slider(
                            label="DPM Solver Order",
                            minimum=1,
                            maximum=3,
                            value=3,
                            step=1,
                            info="Higher order = more accurate but slower"
                        )
                        
                        use_karras_sigmas = gr.Checkbox(
                            label="Use Karras Sigmas",
                            value=False,
                            info="Alternative noise scheduling approach"
                        )
                    
                    # Hide dynamic thresholding - keep as hidden components for compatibility
                    with gr.Row(equal_height=True, visible=False):
                        enable_dynamic_thresholding = gr.Checkbox(
                            label="Enable Dynamic Thresholding",
                            value=False,
                            visible=False  # Hidden
                        )
                        
                        dynamic_thresholding_ratio = gr.Slider(
                            label="Dynamic Thresholding Ratio",
                            minimum=0.5,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            visible=False  # Hidden
                        )
                    
                    # Flow Scheduler specific settings
                    gr.Markdown("#### Flow Scheduler Parameters (when using euler scheduler)")
                    with gr.Row(equal_height=True):
                        enable_dynamic_time_shift = gr.Checkbox(
                            label="Enable Dynamic Time Shift",
                            value=True,
                            info="Enables adaptive time shifting for better quality"
                        )

                # Reset negative prompt function
                def reset_negative_prompt():
                    return NEGATIVE_PROMPT
                
                reset_negative_btn.click(
                    fn=reset_negative_prompt,
                    outputs=[negative_prompt]
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

        def update_generation_info_with_error(error_message):
            """Create error display for the generation info box"""
            return f"""
            <div style='background-color: #422; color: #fcc; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 13px; border-left: 4px solid #f44;'>
                <strong>❌ Generation Error:</strong><br>
                {error_message}
            </div>
            """

        def handle_oom_error():
            """Handle Out of Memory errors gracefully"""
            global pipeline, accelerator, load_on_demand
            print("🚨 Out of Memory detected. Attempting recovery...")
            
            try:
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Force garbage collection
                gc.collect()
                
                # Unload pipeline to free VRAM
                if 'pipeline' in globals() and pipeline is not None:
                    print("🔄 Unloading pipeline to free VRAM...")
                    unload_pipeline()
                
                print("✅ Recovery attempt completed. VRAM should be freed.")
                return "Out of Memory error occurred. Pipeline unloaded to free VRAM. Try reducing image dimensions, batch size, or inference steps."
                
            except Exception as e:
                print(f"⚠️ Error during OOM recovery: {e}")
                return f"Out of Memory error occurred. Recovery attempt failed: {str(e)}"

        def cancel_generation():
            """Hard cancel the current generation"""
            global generation_cancelled, cancel_event, generation_thread, load_on_demand, pipeline
            
            print("🛑 Hard cancellation requested by user")
            generation_cancelled = True
            cancel_event.set()
            
            # Try to forcibly terminate the generation thread
            if generation_thread and generation_thread.is_alive():
                print("🔥 Attempting to forcibly terminate generation thread...")
                
                # Give it a moment to check the cancel flag naturally
                time.sleep(0.1)
                
                if generation_thread.is_alive():
                    # Force terminate the thread
                    success = terminate_thread(generation_thread)
                    if success:
                        print("✅ Generation thread terminated")
                    else:
                        print("⚠️ Could not terminate thread cleanly")
                
                # Wait a bit for cleanup
                time.sleep(0.2)
            
            # Force cleanup regardless of thread termination success
            try:
                # Clear CUDA cache aggressively
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Force garbage collection
                gc.collect()
                
                # Unload/reload pipeline depending on mode
                if pipeline is not None:
                    if load_on_demand:
                        print("🔄 Unloading pipeline due to cancellation in load-on-demand mode...")
                        unload_pipeline()
                    else:
                        print("🔄 Reloading pipeline due to hard cancellation...")
                        try:
                            # Try to reinitialize the pipeline
                            global accelerator
                            if accelerator is not None:
                                # Get the args from somewhere - we'll need to pass them
                                # For now, just unload and let it reload on next generation
                                unload_pipeline()
                                print("⚠️ Pipeline unloaded. Will reload on next generation.")
                        except Exception as e:
                            print(f"⚠️ Error during pipeline reload: {e}")
                            unload_pipeline()
                
                print("✅ Cancellation cleanup completed")
                return "Generation forcibly cancelled and cleaned up"
                
            except Exception as e:
                print(f"⚠️ Error during cancellation cleanup: {e}")
                return f"Generation cancelled but cleanup had errors: {str(e)}"

        def update_generation_info(instruction, width, height, scheduler, steps, negative_prompt, 
                                 text_guidance, image_guidance, cfg_start, cfg_end, num_images, 
                                 max_side, max_pixels, seed, output_img, error_message=None):
            if error_message:
                return update_generation_info_with_error(error_message)
            
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

        def threaded_generation_wrapper(
            instruction, width_input, height_input, scheduler, num_inference_steps,
            image_input_1, image_input_2, image_input_3, negative_prompt,
            guidance_scale_input, img_guidance_scale_input, cfg_range_start, cfg_range_end,
            num_images_per_prompt, max_input_image_side_length, max_pixels,
            seed_input, rotary_theta, max_sequence_length, dpm_algorithm_type,
            dpm_solver_type, dpm_solver_order, use_karras_sigmas,
            enable_dynamic_thresholding, dynamic_thresholding_ratio,
            enable_dynamic_time_shift, save_images_enabled, align_res, args
        ):
            """Wrapper function to run generation in a separate thread"""
            global generation_result, generation_exception
            
            try:
                result = run(
                    instruction, width_input, height_input, scheduler, num_inference_steps,
                    image_input_1, image_input_2, image_input_3, negative_prompt,
                    guidance_scale_input, img_guidance_scale_input, cfg_range_start, cfg_range_end,
                    num_images_per_prompt, max_input_image_side_length, max_pixels,
                    seed_input, rotary_theta, max_sequence_length, dpm_algorithm_type,
                    dpm_solver_type, dpm_solver_order, use_karras_sigmas,
                    enable_dynamic_thresholding, dynamic_thresholding_ratio,
                    enable_dynamic_time_shift, save_images_enabled, None, align_res, args
                )
                generation_result = result
                generation_exception = None
            except ThreadKilledException:
                print("🛑 Generation thread was forcibly terminated")
                generation_result = None
                generation_exception = ThreadKilledException("Generation was forcibly cancelled")
            except Exception as e:
                print(f"❌ Generation thread failed: {e}")
                generation_result = None
                generation_exception = e

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
            # Advanced parameters
            rotary_theta,
            max_sequence_length,
            dpm_algorithm_type,
            dpm_solver_type,
            dpm_solver_order,
            use_karras_sigmas,
            enable_dynamic_thresholding,
            dynamic_thresholding_ratio,
            enable_dynamic_time_shift,
            save_images_enabled,
            progress=gr.Progress(),
        ):
            global generation_thread, generation_result, generation_exception, generation_cancelled
            
            align_res = aspect_ratio_choice == "Use Image1 Aspect Ratio"
            max_pixels = int(max_pixels_mp * 1_000_000)
            
            # Reset global state
            generation_result = None
            generation_exception = None
            generation_cancelled = False
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=threaded_generation_wrapper,
                args=(
                    instruction, width_input, height_input, scheduler, num_inference_steps,
                    image_input_1, image_input_2, image_input_3, negative_prompt,
                    guidance_scale_input, img_guidance_scale_input, cfg_range_start, cfg_range_end,
                    num_images_per_prompt, max_input_image_side_length, max_pixels,
                    seed_input, rotary_theta, max_sequence_length, dpm_algorithm_type,
                    dpm_solver_type, dpm_solver_order, use_karras_sigmas,
                    enable_dynamic_thresholding, dynamic_thresholding_ratio,
                    enable_dynamic_time_shift, save_images_enabled, align_res, args
                ),
                daemon=True
            )
            
            generation_thread.start()
            
            # Wait for completion or cancellation with progress updates
            start_time = time.time()
            while generation_thread.is_alive():
                if generation_cancelled:
                    # Cancellation was requested, thread should be terminated by cancel_generation()
                    break
                
                # Update progress if we can estimate it
                elapsed = time.time() - start_time
                if progress and elapsed > 1:
                    # Rough progress estimation based on time (very crude)
                    estimated_total = 60  # Assume 60 seconds for generation
                    progress_frac = min(elapsed / estimated_total, 0.9)
                    progress(progress_frac)
                
                time.sleep(0.1)
            
            # Wait for thread to finish (or timeout)
            generation_thread.join(timeout=1.0)
            
            # Check results
            if generation_cancelled:
                raise Exception("Generation was cancelled by user")
            
            if generation_exception:
                if isinstance(generation_exception, ThreadKilledException):
                    raise Exception("Generation was forcibly cancelled")
                else:
                    raise generation_exception
            
            if generation_result is None:
                raise Exception("Generation failed - no result returned")
            
            return generation_result

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
            # Advanced parameters
            rotary_theta,
            max_sequence_length,
            dpm_algorithm_type,
            dpm_solver_type,
            dpm_solver_order,
            use_karras_sigmas,
            enable_dynamic_thresholding,
            dynamic_thresholding_ratio,
            enable_dynamic_time_shift,
            save_images_enabled,
            progress=gr.Progress(),
        ):
            global generation_cancelled, cancel_event
            
            # Reset cancellation state
            generation_cancelled = False
            cancel_event.clear()
            
            try:
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
                    # Advanced parameters
                    rotary_theta,
                    max_sequence_length,
                    dpm_algorithm_type,
                    dpm_solver_type,
                    dpm_solver_order,
                    use_karras_sigmas,
                    enable_dynamic_thresholding,
                    dynamic_thresholding_ratio,
                    enable_dynamic_time_shift,
                    save_images_enabled,
                    progress,
                )
                
                if generation_cancelled:
                    error_message = "Generation was cancelled by user"
                    generation_info_result = update_generation_info(
                        instruction, width_input, height_input, scheduler_input, num_inference_steps,
                        negative_prompt, text_guidance_scale_input, image_guidance_scale_input,
                        cfg_range_start, cfg_range_end, num_images_per_prompt, max_input_image_side_length,
                        int(max_pixels_mp * 1_000_000), actual_seed, None, error_message
                    )
                    return None, generation_info_result
                
                # Update the generation info with successful parameters
                generation_info_result = update_generation_info(
                    instruction, width_input, height_input, scheduler_input, num_inference_steps,
                    negative_prompt, text_guidance_scale_input, image_guidance_scale_input,
                    cfg_range_start, cfg_range_end, num_images_per_prompt, max_input_image_side_length,
                    int(max_pixels_mp * 1_000_000), actual_seed, output_image_result
                )
                
                return all_images, generation_info_result
                
            except torch.cuda.OutOfMemoryError as e:
                error_message = handle_oom_error()
                generation_info_result = update_generation_info(
                    instruction, width_input, height_input, scheduler_input, num_inference_steps,
                    negative_prompt, text_guidance_scale_input, image_guidance_scale_input,
                    cfg_range_start, cfg_range_end, num_images_per_prompt, max_input_image_side_length,
                    int(max_pixels_mp * 1_000_000), seed_input, None, error_message
                )
                return None, generation_info_result
                
            except Exception as e:
                error_message = f"Generation failed: {str(e)}"
                print(f"❌ Generation error: {error_message}")
                print(f"Traceback: {traceback.format_exc()}")
                
                generation_info_result = update_generation_info(
                    instruction, width_input, height_input, scheduler_input, num_inference_steps,
                    negative_prompt, text_guidance_scale_input, image_guidance_scale_input,
                    cfg_range_start, cfg_range_end, num_images_per_prompt, max_input_image_side_length,
                    int(max_pixels_mp * 1_000_000), seed_input, None, error_message
                )
                return None, generation_info_result

        # Connect the cancel button
        cancel_button.click(
            fn=lambda: cancel_generation(),
            outputs=[generation_info]
        )

        # Connect the generate button
        generate_button.click(
            fn=generate_and_update_info,
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
                # Advanced parameters
                rotary_theta,
                max_sequence_length,
                dpm_algorithm_type,
                dpm_solver_type,
                dpm_solver_order,
                use_karras_sigmas,
                enable_dynamic_thresholding,
                dynamic_thresholding_ratio,
                enable_dynamic_time_shift,
                save_images,
            ],
            outputs=[output_display, generation_info],
        )

        # Add Enter key handler for prompt box
        instruction.submit(
            fn=generate_and_update_info,
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
                # Advanced parameters
                rotary_theta,
                max_sequence_length,
                dpm_algorithm_type,
                dpm_solver_type,
                dpm_solver_order,
                use_karras_sigmas,
                enable_dynamic_thresholding,
                dynamic_thresholding_ratio,
                enable_dynamic_time_shift,
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
