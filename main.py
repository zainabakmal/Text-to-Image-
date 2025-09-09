# Run this cell first. It installs diffusers + helpers.
# If Colab already has torch preinstalled you don't need to force a heavy torch reinstall.
!pip install -q diffusers transformers accelerate safetensors huggingface_hub


# Run this cell to login; it will prompt for your token in the notebook.
from huggingface_hub import login
# paste your Hugging Face token when prompted
login()


# Load model + small optimizations. This selects GPU automatically if available.
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from IPython.display import display

model_id = "runwayml/stable-diffusion-v1-5"  # change if you want another model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Choose dtype: fp16 on GPU, fp32 on CPU
dtype = torch.float16 if device == "cuda" else torch.float32

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)

# Memory-saving helpers (GPU)
if device == "cuda":
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    # If xformers installed you can also try to enable it for extra memory savings:
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

print("Pipeline ready.")


# Example generation cell. Change prompt to whatever you want.
prompt = "A magical forest with bioluminescent trees, ultra-detailed, cinematic, 4k"
seed = 42               # set to any int for reproducible outputs, or None for random
num_inference_steps = 50
guidance_scale = 7.5    # higher = follows prompt more strictly

# width and height must be multiples of 8
width = 512
height = 512

generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

result = pipe(
    prompt,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator
)

image = result.images[0]
display(image)

# Save image to workspace
image_path = "sd_generated.png"
image.save(image_path)
print("Saved to", image_path)


from google.colab import files
files.download("sd_generated.png")
