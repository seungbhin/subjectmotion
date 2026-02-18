import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import random
import os

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(path="./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
        ModelConfig(path="./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    ],
)
lora_spatial_path = "./outputs/train/subject/dog/XXXX-XXXX/loras/epoch-3.safetensors"
lora_temporal_path = "./outputs/train/motion/skateboarding/XXXX-XXXX/loras/temporal/epoch-4.safetensors"
pipe.load_lora(pipe.dit, lora_spatial_path, alpha=0.6)
print("-"*100)
pipe.load_lora(pipe.dit, lora_temporal_path, alpha=1.0)
pipe.enable_vram_management()
seed = random.randint(0, 10000000)
video = pipe(
    prompt="A sks dog is vvt skateboarding through an open-air market, dodging fruit stands and weaving between busy shoppers.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=seed, 
    tiled=True,
    inference_type = "STAIS",
    num_frames = 49,
    lora_spatial_path = lora_spatial_path,
    lora_temporal_path = lora_temporal_path,
    sigma_shift = 1.0
)
name = lora_spatial_path.split("/")[-4]
os.makedirs(f"./outputs/inference/{name}", exist_ok=True)
save_video(video, f"./outputs/inference/{name}/{seed}.mp4", fps=15, quality=5)
