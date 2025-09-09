# Text-to-Image AI Model

This project generates images from text prompts using a diffusion-based AI model.

## Setup

Install Python libraries: torch, torchvision, diffusers, transformers, accelerate, Pillow, and tqdm.

## Dataset

Create a folder called "data".
Inside it, add a folder "images" that contains your images (like 0001.jpg, 0002.jpg, etc.).
Also add a file "annotations.csv" with two columns: file and caption.
### Example:
0001.jpg, A red car on the road
0002.jpg, A dog sitting on a sofa

## Training

Run the training script (train.py).
It will read your dataset, train the diffusion model, and save checkpoints.

## Inference

Run the inference script (inference.py) with a text prompt.
It will generate an image based on the prompt and save it as a PNG file.

## Configuration

Training settings like batch size, epochs, learning rate, and image size can be edited inside the config file (configs/default.yaml).

## Tips

Use pretrained models for text encoder and VAE to save time.

If you get out-of-memory errors, reduce image size or batch size.

Clean and detailed captions give better results.
