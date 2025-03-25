import os
import re
import time
import gradio as gr
from openai import OpenAI
import base64
from PIL import Image
import io
import uuid

from cogkit.api.settings import APISettings

settings = APISettings()


def convert_prompt(
    prompt: str,
    retry_times: int = 5,
) -> str:
    def clean_string(s):
        s = s.replace("\n", " ")
        s = s.strip()
        s = re.sub(r"\s{2,}", " ", s)
        return s

    client = OpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
    prompt = clean_string(prompt)
    for i in range(retry_times):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": 'You are a bilingual image description assistant that works with an image generation bot.  You work with an assistant bot that will draw anything you say . \n    For example ,For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" or "阳光透过树林的美丽清晨" will trigger your partner bot to output an image of a forest morning, as described . \n    You will be prompted by people looking to create detailed , amazing images . The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive . \n    There are a few rules to follow : \n    - Input can be in Chinese or English. If input is in English, prompt should be written in English. If input is in Chinese, prompt should be written in Chinese.\n    - You will only ever output a single image description per user request .\n    - Image descriptions must be detailed and specific, including keyword categories such as subject, medium, style, additional details, color, and lighting. \n    - When generating descriptions, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.\n    - Do not provide the process and explanation, just return the modified description . \n    ',
                    },
                    {
                        "role": "user",
                        "content": f"Create an imaginative image descriptive caption for the user input : {prompt}",
                    },
                ],
                model="glm-4-flash",  # You can change to other models like "glm-4-plus" if needed
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=300,
            )
            enhanced_prompt = response.choices[0].message.content
            if enhanced_prompt:
                enhanced_prompt = clean_string(enhanced_prompt)
                return enhanced_prompt
        except Exception as e:
            print(f"Error enhancing prompt (attempt {i + 1}): {e}")
            time.sleep(1)
    return prompt


def get_lora_paths():
    lora_dir = settings.lora_dir
    if not os.path.exists(lora_dir):
        os.makedirs(lora_dir, exist_ok=True)
        return ["None"]
    checkpoint_dirs = [
        d
        for d in os.listdir(lora_dir)
        if os.path.isdir(os.path.join(lora_dir, d)) and d.startswith("checkpoint")
    ]

    if not checkpoint_dirs:
        return ["None"]
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]) if x != "None" and "-" in x else 0)
    return ["None"] + checkpoint_dirs


def generate_images(
    prompt,
    num_images=1,
    width=768,
    height=1024,
    num_inference_steps=1,
    guidance_scale=3.5,
    lora_path=None,
):
    client = OpenAI(base_url="http://127.0.0.1:8000/v1/", api_key="EMPTY")

    extra_body = {
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }

    if lora_path and lora_path != "None":
        full_lora_path = os.path.join(settings.lora_dir, lora_path)
        if os.path.exists(full_lora_path):
            extra_body["lora_path"] = full_lora_path
    response = client.images.generate(
        model="cogview-4",
        prompt=prompt,
        n=num_images,
        size=f"{width}x{height}",
        extra_body=extra_body,
    )
    return response.data


def infer(
    prompt,
    width,
    height,
    num_images,
    num_inference_steps,
    guidance_scale,
    lora_path,
    progress=gr.Progress(track_tqdm=True),
):
    images_data = generate_images(
        prompt=prompt,
        num_images=int(num_images),
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        lora_path=lora_path,
    )

    image_results = []
    for img in images_data:
        if hasattr(img, "url") and img.url:
            image_results.append(img.url)
        elif hasattr(img, "b64_json") and img.b64_json:
            img_data = base64.b64decode(img.b64_json)
            img_obj = Image.open(io.BytesIO(img_data))
            os.makedirs("./gradio_tmp", exist_ok=True)
            file_path = f"./gradio_tmp/img_{uuid.uuid4()}.png"
            img_obj.save(file_path)
            image_results.append(file_path)

    return image_results


def update_max_height(width):
    max_height = MAX_PIXELS // width
    return gr.update(maximum=max_height)


def update_max_width(height):
    max_width = MAX_PIXELS // height
    return gr.update(maximum=max_width)


def refresh_lora_paths():
    paths = get_lora_paths()
    return gr.update(choices=paths, value=paths[0])


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">CogKit Inference</h2>
            </div>
        """)

    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        max_lines=15,
                        placeholder="Enter your prompt",
                        container=False,
                    )
                with gr.Row():
                    enhance = gr.Button("Enhance Prompt (Strongly Suggest)", scale=1)
                    run_button = gr.Button("Generate Images", scale=1)
                with gr.Row():
                    num_images = gr.Number(
                        label="Number of Images",
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1,
                    )
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=2048,
                        step=32,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=2048,
                        step=32,
                        value=1024,
                    )
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=1,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0,
                        maximum=10,
                        step=0.1,
                        value=3.5,
                    )

                with gr.Row():
                    lora_path = gr.Dropdown(
                        label="Lora Path", choices=get_lora_paths(), value=get_lora_paths()[0]
                    )
                    refresh_button = gr.Button("Refresh Lora Folder", size="sm")
            with gr.Column():
                result = gr.Gallery(label="Results", show_label=True)

        MAX_PIXELS = 2**21
        enhance.click(convert_prompt, inputs=[prompt], outputs=[prompt])
        width.change(update_max_height, inputs=[width], outputs=[height])
        height.change(update_max_width, inputs=[height], outputs=[width])
        refresh_button.click(refresh_lora_paths, inputs=[], outputs=[lora_path])
    run_button.click(
        fn=infer,
        inputs=[prompt, width, height, num_images, num_inference_steps, guidance_scale, lora_path],
        outputs=[result],
    )
    prompt.submit(
        fn=infer,
        inputs=[prompt, width, height, num_images, num_inference_steps, guidance_scale, lora_path],
        outputs=[result],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1")
