import dotenv

dotenv.load_dotenv(override=True)

import gradio as gr

import os
import argparse
import random
from datetime import datetime

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from omnigen2.utils.img_util import create_collage

NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

pipeline = None
accelerator = None
save_images = False

def load_pipeline(accelerator, weight_dtype, args):
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
    return pipeline


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
    progress=gr.Progress(),
    align_res=False,
):
    input_images = [image_input_1, image_input_2, image_input_3]
    input_images = [img for img in input_images if img is not None]

    if len(input_images) == 0:
        input_images = None

    if seed_input == -1:
        seed_input = random.randint(0, 2**16 - 1)

    generator = torch.Generator(device=accelerator.device).manual_seed(seed_input)

    def progress_callback(cur_step, timesteps):
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

    progress(1.0)

    vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
    output_image = create_collage(vis_images)

    if save_images:
        # Create outputs directory if it doesn't exist
        output_dir = os.path.join(ROOT_DIR, "outputs_gradio")
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Generate unique filename with timestamp
        output_path = os.path.join(output_dir, f"{timestamp}.png")
        # Save the image
        output_image.save(output_path)

        # Save All Generated Images
        if len(results.images) > 1:
            for i, image in enumerate(results.images):
                image_name, ext = os.path.splitext(output_path)
                image.save(f"{image_name}_{i}{ext}")
    return output_image


def get_example():
    case = [
        [
            "The sun rises slightly, the dew on the rose petals in the garden is clear, a crystal ladybug is crawling to the dew, the background is the early morning garden, macro lens.",
            1024,
            1024,
            "euler",
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            3.5,
            1.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "A snow maiden with pale translucent skin, frosty white lashes, and a soft expression of longing",
            1024,
            1024,
            "euler",
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            3.5,
            1.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Add a fisherman hat to the woman's head",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/flux5.png"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            " replace the sword with a hammer.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/d8f8f44c64106e7715c61b5dfa9d9ca0974314c5d4a4a50418acf7ff373432bb.png",
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Extract the character from the picture and fill the rest of the background with white.",
            # "Transform the sculpture into jade",
            1024,
            1024,
            "euler",
            50,
            os.path.join(
                ROOT_DIR, "example_images/46e79704-c88e-4e68-97b4-b4c40cd29826.png"
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Make he smile",
            1024,
            1024,
            "euler",
            50,
            os.path.join(
                ROOT_DIR, "example_images/vicky-hladynets-C8Ta0gwPbQg-unsplash.jpg"
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Change the background to classroom",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/ComfyUI_temp_mllvz_00071_.png"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Raise his hand",
            1024,
            1024,
            "euler",
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/289089159-a6d7abc142419e63cab0a566eb38e0fb6acb217b340f054c6172139b316f6596.png",
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Generate a photo of an anime-style figurine placed on a desk. The figurine model should be based on the character photo provided in the attachment, accurately replicating the full-body pose, facial expression, and clothing style of the character in the photo, ensuring the entire figurine is fully presented. The overall design should be exquisite and detailed, soft gradient colors and a delicate texture, leaning towards a Japanese anime style, rich in details, with a realistic quality and beautiful visual appeal.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/RAL_0315.JPG"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Change the dress to blue.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/1.png"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Remove the cat",
            1024,
            1024,
            "euler",
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/386724677-589d19050d4ea0603aee6831459aede29a24f4d8668c62c049f413db31508a54.png",
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "In a cozy café, the anime figure is sitting in front of a laptop, smiling confidently.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/ComfyUI_00254_.png"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Create a wedding figure based on the girl in the first image and the man in the second image. Set the background as a wedding hall, with the man dressed in a suit and the girl in a white wedding dress. Ensure that the original faces remain unchanged and are accurately preserved. The man should adopt a realistic style, whereas the girl should maintain their classic anime style.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/1_20241127203215.png"),
            os.path.join(ROOT_DIR, "example_images/000050281.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            3.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Let the girl  and the boy get married in the church. ",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/8FtFUxRzXqaguVRGzkHvN.png"),
            os.path.join(ROOT_DIR, "example_images/01194-20240127001056_1024x1536.png"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            3.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Let the man from image1 and the woman from image2 kiss and hug",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/1280X1280.png"),
            os.path.join(ROOT_DIR, "example_images/000077066.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Please let the person in image 2 hold the toy from the first image in a parking lot.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/04.jpg"),
            os.path.join(ROOT_DIR, "example_images/000365954.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Make the girl pray in the second image.",
            1024,
            682,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/000440817.jpg"),
            os.path.join(ROOT_DIR, "example_images/000119733.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Add the bird from image 1 to the desk in image 2",
            1024,
            682,
            "euler",
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/996e2cf6-daa5-48c4-9ad7-0719af640c17_1748848108409.png",
            ),
            os.path.join(ROOT_DIR, "example_images/00066-10350085.png"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Replace the apple in the first image with the cat from the second image",
            1024,
            780,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/apple.png"),
            os.path.join(
                ROOT_DIR,
                "example_images/468404374-d52ec1a44aa7e0dc9c2807ce09d303a111c78f34da3da2401b83ce10815ff872.png",
            ),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Replace the woman in the second image with the woman from the first image",
            1024,
            747,
            "euler",
            50,
            os.path.join(
                ROOT_DIR, "example_images/byward-outfitters-B97YFrsITyo-unsplash.jpg"
            ),
            os.path.join(
                ROOT_DIR, "example_images/6652baf6-4096-40ef-a475-425e4c072daf.png"
            ),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "The cat is sitting on the table. The bird is perching on the edge of the table.",
            800,
            512,
            "euler",
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/996e2cf6-daa5-48c4-9ad7-0719af640c17_1748848108409.png",
            ),
            os.path.join(
                ROOT_DIR,
                "example_images/468404374-d52ec1a44aa7e0dc9c2807ce09d303a111c78f34da3da2401b83ce10815ff872.png",
            ),
            os.path.join(ROOT_DIR, "example_images/00066-10350085.png"),
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
    ]
    return case


def run_for_examples(
    instruction,
    width_input,
    height_input,
    scheduler,
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
    max_pixels,
    seed_input,
):
    return run(
        instruction,
        width_input,
        height_input,
        scheduler,
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
        max_pixels,
        seed_input,
    )

def main(args):
    # Gradio
    with gr.Blocks() as demo:
        gr.Markdown(
            "# OmniGen2: Unified Image Generation"
        )
        with gr.Row():
            with gr.Column():
                # Move Generate button to the top
                generate_button = gr.Button("Generate Image")

                # text prompt
                instruction = gr.Textbox(
                    label='Enter your prompt. Use "first/second image" as reference.',
                    placeholder="Type your prompt here...",
                    lines=2,
                    elem_id="prompt-box",
                )

                with gr.Row(equal_height=True):
                    # input images
                    image_input_1 = gr.Image(label="Input Image 1", type="pil", height=320, width=320, show_label=True, elem_id="input-image-1")
                    image_input_2 = gr.Image(label="Input Image 2", type="pil", height=320, width=320, show_label=True, elem_id="input-image-2")
                    image_input_3 = gr.Image(label="Input Image 3", type="pil", height=320, width=320, show_label=True, elem_id="input-image-3")

                negative_prompt = gr.Textbox(
                    label="Enter your negative prompt",
                    placeholder="Type your negative prompt here...",
                    value=NEGATIVE_PROMPT,
                )

                # Aspect ratio and dimensions
                aspect_ratio_multiplier = gr.Dropdown(
                    label="Aspect Ratio Multiplier",
                    choices=["x1", "x2", "x3", "x4"],
                    value="x1",
                    info="Multiply the default aspect ratio size by this factor."
                )

                aspect_ratio = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=["Use Image1 Aspect Ratio", "1:1 (Square)", "4:3 (Landscape)", "3:4 (Portrait)", "16:9 (Widescreen)", "9:16 (Portrait)", "21:9 (Ultrawide)", "9:21 (Portrait)", "Custom"],
                    value="1:1 (Square)",
                    info="Select aspect ratio or choose Custom to set dimensions manually."
                )

                # slider
                with gr.Row(equal_height=True):
                    height_input = gr.Slider(
                        label="Height", minimum=256, maximum=4096, value=1024, step=128
                    )
                    width_input = gr.Slider(
                        label="Width", minimum=256, maximum=4096, value=1024, step=128
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
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                    )
                with gr.Row(equal_height=True):
                    cfg_range_start = gr.Slider(
                        label="CFG Range Start",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.1,
                    )

                    cfg_range_end = gr.Slider(
                        label="CFG Range End",
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
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
                    multiplier = int(multiplier_choice[1]) if multiplier_choice and multiplier_choice.startswith('x') else 1
                    if aspect_choice in aspect_ratios and aspect_choice not in ["Custom", "Use Image1 Aspect Ratio"]:
                        base_width, base_height = aspect_ratios[aspect_choice]
                        width = min(base_width * multiplier, 4096)
                        height = min(base_height * multiplier, 4096)
                        return width, height
                    return 1024, 1024

                aspect_ratio.change(
                    fn=update_dimensions_from_aspect_and_multiplier,
                    inputs=[aspect_ratio, aspect_ratio_multiplier],
                    outputs=[width_input, height_input]
                )
                aspect_ratio_multiplier.change(
                    fn=update_dimensions_from_aspect_and_multiplier,
                    inputs=[aspect_ratio, aspect_ratio_multiplier],
                    outputs=[width_input, height_input]
                )

                with gr.Row(equal_height=True):
                    scheduler_input = gr.Dropdown(
                        label="Scheduler",
                        choices=["euler", "dpmsolver"],
                        value="euler",
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
                        label="Seed", minimum=-1, maximum=2147483647, value=0, step=1,
                        elem_id="seed-slider"
                    )
                with gr.Row(equal_height=True):
                    max_input_image_side_length = gr.Slider(
                        label="max_input_image_side_length",
                        minimum=256,
                        maximum=2048,
                        value=2048,
                        step=256,
                    )
                    max_pixels = gr.Slider(
                        label="max_pixels",
                        minimum=256 * 256,
                        maximum=1536 * 1536,
                        value=1024 * 1024,
                        step=256 * 256,
                    )

            with gr.Column():
                with gr.Column():
                    # output image
                    output_image = gr.Image(label="Output Image")
                    
                    # Generation info box
                    generation_info = gr.HTML(
                        label="Generation Details",
                        value="""
                        <div style='background-color: #222; color: #eee; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 13px;'>No generation yet</div>
                        """
                    )
                    
                    global save_images
                    save_images = gr.Checkbox(label="Save generated images", value=False)

        global accelerator
        global pipeline

        bf16 = True
        accelerator = Accelerator(mixed_precision="bf16" if bf16 else "no")
        weight_dtype = torch.bfloat16 if bf16 else torch.float32

        pipeline = load_pipeline(accelerator, weight_dtype, args)

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
                <strong>Seed:</strong> <span style='color: #4ea1ff; cursor: pointer; text-decoration: underline;' onclick='document.getElementById("seed-slider").value = "{seed}"; document.getElementById("seed-slider").dispatchEvent(new Event("input", {{ bubbles: true }}));'>{seed}</span>
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
            max_pixels,
            seed_input,
            aspect_ratio_choice,
            progress=gr.Progress(),
        ):
            align_res = aspect_ratio_choice == "Use Image1 Aspect Ratio"
            return run(
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
                progress=progress,
                align_res=align_res,
            )

        # click
        generate_event = generate_button.click(
            run_with_align_res,
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
                max_pixels,
                seed_input,
                aspect_ratio,
            ],
            outputs=output_image,
        ).then(
            update_generation_info,
            inputs=[
                instruction,
                width_input,
                height_input,
                scheduler_input,
                num_inference_steps,
                negative_prompt,
                text_guidance_scale_input,
                image_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels,
                seed_input,
                output_image,
            ],
            outputs=generation_info,
        )

        # Add Enter key handler for prompt box
        instruction.submit(
            run_with_align_res,
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
                max_pixels,
                seed_input,
                aspect_ratio,
            ],
            outputs=output_image,
        ).then(
            update_generation_info,
            inputs=[
                instruction,
                width_input,
                height_input,
                scheduler_input,
                num_inference_steps,
                negative_prompt,
                text_guidance_scale_input,
                image_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels,
                seed_input,
                output_image,
            ],
            outputs=generation_info,
        )

    # launch
    demo.launch(share=args.share, server_port=args.port, server_name="0.0.0.0", allowed_paths=[ROOT_DIR])

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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
