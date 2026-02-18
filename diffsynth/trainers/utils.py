import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from diffsynth.utils.utils_encoder import SEARAFT_FlowProcessor
import os, json

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, adapter_name="default"):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model, adapter_name=adapter_name)
        return model


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.loss_values = []  # 总损失
        self.loss_main_values = []  # 主损失 (MSE loss)
        self.loss_proj_values = []  # 投影损失 (projection loss)

    def on_step_end(self, loss_total, loss_main=None, loss_proj=None, accelerator=None):
        # 仅主进程记录，避免多进程重复
        if accelerator is not None and not accelerator.is_main_process:
            return
        try:
            total_value = loss_total.detach().float().item()
            main_value = loss_main.detach().float().item() if loss_main is not None else 0.0
            proj_value = loss_proj.detach().float().item() if loss_proj is not None else 0.0
        except Exception:
            total_value = float(loss_total)
            main_value = float(loss_main) if loss_main is not None else 0.0
            proj_value = float(loss_proj) if loss_proj is not None else 0.0
            
        self.loss_values.append(total_value)
        self.loss_main_values.append(main_value)
        self.loss_proj_values.append(proj_value)

    def _plot_and_save_losses(self):
        if len(self.loss_values) == 0:
            return
        # 将图保存到"运行根目录"（即 loras 的上一层），例如 ./outputs/train/dog2/loss.png
        run_root = os.path.dirname(self.output_path)
        os.makedirs(run_root, exist_ok=True)
        out_path = os.path.join(run_root, "loss.png")

        x = list(range(1, len(self.loss_values) + 1))
        
        # 计算EMA
        ema_total = calculate_ema(self.loss_values, alpha=0.9)
        ema_main = calculate_ema(self.loss_main_values, alpha=0.9) if self.loss_main_values else []
        ema_proj = calculate_ema(self.loss_proj_values, alpha=0.9) if self.loss_proj_values else []

        plt.figure(figsize=(12, 8))
        
        # 创建子图
        plt.subplot(2, 2, 1)
        plt.plot(x, self.loss_values, label="Total Loss", alpha=0.35, color='blue')
        plt.plot(x, ema_total, label="Total Loss EMA(0.9)", linewidth=2, color='darkblue')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Total Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        if self.loss_main_values:
            plt.plot(x, self.loss_main_values, label="Main Loss", alpha=0.35, color='green')
            plt.plot(x, ema_main, label="Main Loss EMA(0.9)", linewidth=2, color='darkgreen')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Main Loss (MSE)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        if self.loss_proj_values:
            plt.plot(x, self.loss_proj_values, label="Projection Loss", alpha=0.35, color='red')
            plt.plot(x, ema_proj, label="Proj Loss EMA(0.9)", linewidth=2, color='darkred')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Projection Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        # 综合对比图
        plt.plot(x, ema_total, label="Total Loss EMA", linewidth=2, color='blue')
        if self.loss_main_values:
            plt.plot(x, ema_main, label="Main Loss EMA", linewidth=2, color='green')
        if self.loss_proj_values:
            plt.plot(x, ema_proj, label="Proj Loss EMA", linewidth=2, color='red')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    def on_epoch_end(self, accelerator, model, epoch_id, train_type=None):
        if train_type == "Motion":
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # 临时启用所有LoRA参数的梯度以确保完整导出
                print("Temporarily enabling gradients for all LoRA parameters...")
                original_grad_states = {}
                for name, param in model.named_parameters():
                    if "lora" in name.lower():
                        original_grad_states[name] = param.requires_grad
                        param.requires_grad = True
                
                # 获取完整的状态字典
                full_state_dict = accelerator.get_state_dict(model)
                full_state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                    full_state_dict, remove_prefix=self.remove_prefix_in_ckpt
                )
                full_state_dict = self.state_dict_converter(full_state_dict)
                
                # 恢复原来的梯度状态
                print("Restoring original gradient states...")
                for name, param in model.named_parameters():
                    if name in original_grad_states:
                        param.requires_grad = original_grad_states[name]
                
                # 分离spatial和temporal的LoRA权重
                spatial_state_dict = {}
                temporal_state_dict = {}
                other_state_dict = {}
                
                # 分类LoRA参数
                for key, value in full_state_dict.items():
                    if "spatial_lora" in key.lower():
                        spatial_state_dict[key] = value
                    elif "temporal_lora" in key.lower():
                        temporal_state_dict[key] = value
                    else:
                        other_state_dict[key] = value
                
                # 统计权重数量
                spatial_count = len(spatial_state_dict)
                temporal_count = len(temporal_state_dict)
                other_count = len(other_state_dict)
                total_count = len(full_state_dict)
                
                print(f"\nEpoch {epoch_id} - LoRA weights statistics:")
                print(f"  Total parameters: {total_count}")
                print(f"  Spatial LoRA parameters: {spatial_count}")
                print(f"  Temporal LoRA parameters: {temporal_count}")
                print(f"  Other parameters: {other_count}")
                
                # 创建保存目录并保存分离的权重
                os.makedirs(os.path.join(self.output_path, "spatial"), exist_ok=True)
                os.makedirs(os.path.join(self.output_path, "temporal"), exist_ok=True)
                
                spatial_path = os.path.join(self.output_path, "spatial", f"epoch-{epoch_id}.safetensors")
                temporal_path = os.path.join(self.output_path, "temporal", f"epoch-{epoch_id}.safetensors")
                
                accelerator.save(spatial_state_dict, spatial_path, safe_serialization=True)
                accelerator.save(temporal_state_dict, temporal_path, safe_serialization=True)
                
                print(f"Saved spatial LoRA weights ({spatial_count} parameters) to: {spatial_path}")
                print(f"Saved temporal LoRA weights ({temporal_count} parameters) to: {temporal_path}")
                
                # 新增：绘图（覆盖更新）
                self._plot_and_save_losses()
        else:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                state_dict = accelerator.get_state_dict(model)
                state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                    state_dict, remove_prefix=self.remove_prefix_in_ckpt
                )
                state_dict = self.state_dict_converter(state_dict)
                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
                accelerator.save(state_dict, path, safe_serialization=True)

                # 新增：绘图（覆盖更新）
                self._plot_and_save_losses()



def calculate_ema(values, alpha=0.9):
    if not values:
        return []
    ema_values = [values[0]]
    for i in range(1, len(values)):
        ema = alpha * ema_values[-1] + (1 - alpha) * values[i]
        ema_values.append(ema)
    return ema_values

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
def preprocess_raw_image(x, pil_img, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255. #0~1
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x) #正态分布
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic') #兼容DINO原始的训练逻辑，还原DINO本身输入的预处理
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov3' in enc_type:
        x = pil_img
    elif 'dino' in enc_type:
        x = pil_img

    return x


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    test_prompts: list = None,
    test_output_path: str = None,
    test_height: int = None,
    test_width: int = None,
    enc_type: str = None,
    visuliaze_path: str = None,
    flow_path: str = None,
    proj_diff: float = None,
    train_type: str = None,
):
    import os
    
    # 添加CUDA错误检查环境变量（用于Blackwell GPU兼容性）
    if "CUDA_LAUNCH_BLOCKING" not in os.environ:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # 禁用NCCL的某些优化以解决Blackwell GPU兼容性问题
    if "NCCL_IB_DISABLE" not in os.environ:
        os.environ["NCCL_IB_DISABLE"] = "1"
    if "NCCL_P2P_DISABLE" not in os.environ:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    
    # 添加更多NCCL设置以解决Blackwell GPU的兼容性问题
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    if "NCCL_BUFFSIZE" not in os.environ:
        os.environ["NCCL_BUFFSIZE"] = "2097152"
    if "NCCL_P2P_LEVEL" not in os.environ:
        os.environ["NCCL_P2P_LEVEL"] = "0"
    if "NCCL_SHM_DISABLE" not in os.environ:
        os.environ["NCCL_SHM_DISABLE"] = "1"
    
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0])
    
    # 为Motion训练类型设置特殊的DDP配置
    if train_type == "Motion":
        # 设置环境变量以处理未使用参数的问题
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        # Motion训练需要find_unused_parameters=True，因为spatial_lora在某些阶段不会被使用
        
        # 创建支持未使用参数的accelerator配置
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        
        # 在accelerator创建后，使用正确的设备初始化flow_processor
        flow_processor = SEARAFT_FlowProcessor(device=str(accelerator.device))
        # 光流模型保持float32，损失计算也使用float32
        # 将光流模型设置为评估模式，避免BatchNorm的就地操作
        flow_processor.model.eval()
        print(f"[INFO] Flow processor initialized on device: {flow_processor.device}")
    else:
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if train_type == "Subject":
        # 设置所有后缀为 .projectors 的参数需要梯度更新
        print("Setting .projectors parameters to require gradients...")
        projector_param_count = 0
        for name, param in model.named_parameters():
            if name.endswith('.projectors') or '.projectors.' in name:
                param.requires_grad_(True)
                projector_param_count += param.numel()
                print(f"  {name}: requires_grad=True, shape={tuple(param.shape)}")
        print(f"Total projector parameters set to require gradients: {projector_param_count:,}")
        print("-" * 100)
        
        # Load encoders
        if enc_type != None:    
            print("enc_type:", enc_type)
            from diffsynth.utils.utils_encoder import load_encoders
            encoders, encoder_types, architectures = load_encoders(
                    enc_type, accelerator.device, resolution=512
                )

            z_dims = [encoder.embed_dim for encoder in encoders] if enc_type != 'None' else [0]
            print(f"z_dims: {z_dims}") #z_dims: [1536]->dinov2-vit-g

        latents_scale = torch.tensor(
            [0.18215, 0.18215, 0.18215, 0.18215]
            ).view(1, 4, 1, 1).to(accelerator.device)
        
        latents_bias = torch.tensor(
            [0., 0., 0., 0.]
            ).view(1, 4, 1, 1).to(accelerator.device)

        from diffsynth.utils.utils_loss import SILoss
        loss_fn = SILoss(
            prediction="v",
            path_type="linear", 
            encoders=encoders,
            accelerator=accelerator,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            weighting="uniform"
        )


        for epoch_id in range(num_epochs):
            for data in tqdm(dataloader):
                zs = []  # 每次迭代都重新初始化，避免计算图重用
                mask_img = data['mask'][0] #mask_image python type: <class 'PIL.Image.Image'>
                
                if enc_type != None:
                    #data.size:512*512
                    pil_img = data['video'][0] #raw_image python type: <class 'PIL.Image.Image'>
                    # 将 PIL 图片转为 Tensor（保持0-255范围，dtype=uint8，形状[C,H,W]）
                    raw_image = torchvision.transforms.functional.pil_to_tensor(pil_img) #3,512,512
                    raw_image = torch.nn.functional.interpolate(raw_image.unsqueeze(0).float(), size=(512, 512), mode='bilinear', align_corners=False).byte() #1,3,512,512
                    # 将张量移动到与编码器相同的设备上
                    raw_image = raw_image.to(accelerator.device) #1,3,512,512

                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        if 'clip' in encoder_type:
                            raw_image = torch.nn.functional.interpolate(raw_image, size=(256, 256), mode='bilinear', align_corners=False)  # (1)*3*256*256
                        if 'mae' in encoder_type:
                            raw_image = torch.nn.functional.interpolate(raw_image, size=(256, 256), mode='bilinear', align_corners=False)  # (1)*3*256*256
                        
                        raw_image_ = preprocess_raw_image(raw_image, pil_img, encoder_type) #(b*f)*c*h*w 16*3*448*448 正态分布(DINOv2)  (b*f)*c*h*w 1*3*224*224(CLIP)        
                        z = encoder.forward_features(raw_image_) #(b*f)*tokens*z_dims 1*1024*1536 (DINOv2)  (b*f)*tokens*z_dims 1*256*1536 (DINOv2)
                        if 'mocov3' in encoder_type: 
                            z = z[:, 1:]
                        elif 'dinov2' in encoder_type: 
                            z = z['x_norm_patchtokens'] #1,1024,1536
                        print("z_size:", z.shape)
                        zs.append(z) #zs is the list of image features of DINO-V2 zs shape: torch.Size([1, 1024, 1536])
                    

                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    loss, loss_proj, zs_tilde = model(data, zs=zs, loss_fn=loss_fn, z_dims=z_dims, enc_type=enc_type, proj_diff=proj_diff, mask_img=mask_img, train_type=train_type)
                    loss_total = loss + loss_proj * proj_diff
                    accelerator.backward(loss_total)
                    optimizer.step()
                    # 传入分解的损失值用于绘图
                    model_logger.on_step_end(loss_total, loss_main=loss, loss_proj=loss_proj, accelerator=accelerator)
                    scheduler.step()
                    
            model_logger.on_epoch_end(accelerator, model, epoch_id, train_type)
            print("Here start to sample with test prompts!")
            if test_prompts is not None and test_output_path is not None:
                from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
                pipe = WanVideoPipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device=accelerator.device,
                    model_configs=[
                        ModelConfig(path="./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
                        ModelConfig(path="./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
                        ModelConfig(path="./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
                    ],
                )
                from diffsynth import save_video
                 
                import os
                print("model_logger.output_path:", model_logger.output_path)
                lora_path = os.path.join(model_logger.output_path, f"epoch-{epoch_id}.safetensors")
                print("lora_path:", lora_path)
                if os.path.exists(lora_path):
                    pipe.load_lora(pipe.dit, lora_path, alpha=1)
                pipe.enable_vram_management()
                for test_prompt in test_prompts:
                    video = pipe(
                        prompt=test_prompt,
                        height=test_height if test_height is not None else 480,
                        width=test_width if test_width is not None else 832,
                        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，, still, no movement.",
                        seed=1, 
                        num_frames = 33,
                        tiled=True
                    )
                    sanitized_prompt = test_prompt.replace(" ", "_").replace("*", "_").replace(".", "_").replace(",", "_").replace("'", "_")[:20]
                    os.makedirs(os.path.join(test_output_path, f"epoch-{epoch_id}"), exist_ok=True)
                    save_video(video, os.path.join(test_output_path, f"epoch-{epoch_id}", f"{sanitized_prompt}.mp4"), fps=15, quality=5)
                
    elif train_type == "Motion":
        spatial_lora_para = []
        temporal_lora_para = []
        for name, param in model.named_parameters():
            if "spatial_lora" in name.lower():
                spatial_lora_para.append(param)
            elif "temporal_lora" in name.lower():
                temporal_lora_para.append(param)
        
        dataloader_list = list(dataloader)
        for data in tqdm(dataloader):
            video = data["video"]
            f_list, flowviz_list = flow_processor.video_flow(video, accelerator.device)  # 假设flow_processor是CSCVProcessor的实例
            tot_flow_list = []
            for f in f_list:
                flow = f.unsqueeze(0) #1*2*h*w
                tot_flow_list.append(flow)
            tot_flow_list = torch.cat(tot_flow_list, dim=0) #f*2*h*w  #flow_list: torch.Size([48, 2, h, w])
            tot_flow_list = tot_flow_list.to(accelerator.device)
            # 保存光流可视化图像
            if flow_path is None and test_output_path is not None:
                flow_path = os.path.join(test_output_path, "flow_visualization")
            if flow_path is not None:
                import cv2
                os.makedirs(flow_path, exist_ok=True)
                for i, flowviz in enumerate(flowviz_list):
                    flow_filename = f"flow_frame_{i:04d}.png"
                    flow_filepath = os.path.join(flow_path, flow_filename)
                    cv2.imwrite(flow_filepath, flowviz)

        print("FLow has been successfully generated!")
        print("flow_processor.device:",flow_processor.device)
                    
        for epoch_id in range(num_epochs):
            # Motion_0 阶段：只训练 spatial_lora
            if hasattr(model, 'module'):
                model.module.adjust_lora_alpha("spatial_lora", 1.0)
                model.module.adjust_lora_alpha("temporal_lora", 0.0)
            else:
                model.adjust_lora_alpha("spatial_lora", 1.0)
                model.adjust_lora_alpha("temporal_lora", 0.0)
            
            for name, param in model.named_parameters():
                if "spatial_lora" in name.lower():
                    param.requires_grad_(True)
            
            for step_idx, data in enumerate(tqdm(dataloader_list, desc="Motion_0 (Spatial)")):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    loss_spatial = model(data, train_type="Motion_0")
                    accelerator.backward(loss_spatial)
                    
                    # 清零temporal_lora的梯度（使用非就地操作）
                    for param in temporal_lora_para:
                        if param.grad is not None:
                            param.grad = None  # 改为非inplace操作
                    
                    optimizer.step()
                    scheduler.step()
                    
                    # 每个step记录loss
                    model_logger.on_step_end(
                        loss_total=loss_spatial, 
                        loss_main=loss_spatial, 
                        loss_proj=None,  # Motion_0阶段没有projection loss
                        accelerator=accelerator
                    )
            
            # Motion_1 阶段：训练 temporal_lora
            if hasattr(model, 'module'):
                model.module.adjust_lora_alpha("spatial_lora", 1.0)
                model.module.adjust_lora_alpha("temporal_lora", 1.0)
            else:
                model.adjust_lora_alpha("spatial_lora", 1.0)
                model.adjust_lora_alpha("temporal_lora", 1.0)
            
            for name, param in model.named_parameters():
                if "spatial_lora" in name.lower():
                    param.requires_grad_(False)
                
            for step_idx, data in enumerate(tqdm(dataloader_list, desc="Motion_1 (Temporal)")):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    loss_temporal, loss_flow = model(data, train_type="Motion_1", flow_list=tot_flow_list, flow_processor=flow_processor)
                    print("Motion1 loss_temporal:",loss_temporal)
                    print("Motion1 loss_flow:",loss_flow)
                    
                    loss_temporal = loss_temporal.float()
                    loss_flow = loss_flow
                    loss_flow = loss_flow * proj_diff
                    loss_tot = loss_temporal + loss_flow
                    accelerator.backward(loss_tot)            
                    for param in spatial_lora_para:
                        if param.grad is not None:
                            param.grad = None
                    
                    optimizer.step()
                    scheduler.step()
                    
                    # 每个step记录loss
                    model_logger.on_step_end(
                        loss_total=loss_tot, 
                        loss_main=loss_temporal, 
                        loss_proj= loss_flow ,  # Motion_1阶段也没有projection loss
                        accelerator=accelerator
                    )
                    
            model_logger.on_epoch_end(accelerator, model, epoch_id, train_type)
            if test_prompts is not None and test_output_path is not None:
                from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
                pipe = WanVideoPipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device=accelerator.device,
                    model_configs=[
                        ModelConfig(path="./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
                        ModelConfig(path="./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
                        ModelConfig(path="./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
                    ],
                )
                from diffsynth import save_video
                lora_path = os.path.join(model_logger.output_path,"temporal", f"epoch-{epoch_id}.safetensors")
                if os.path.exists(lora_path):
                    pipe.load_lora(pipe.dit, lora_path, alpha=1)
                pipe.enable_vram_management()
                for test_prompt in test_prompts:
                    video = pipe(
                        prompt=test_prompt,
                        height=test_height if test_height is not None else 480,
                        width=test_width if test_width is not None else 832,
                        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕作品，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        seed=1, 
                        num_frames = 49,
                        tiled=True
                    )
                    sanitized_prompt = test_prompt.replace(" ", "_").replace("*", "_").replace(".", "_").replace(",", "_").replace("'", "_")[:20]
                    
                    os.makedirs(os.path.join(test_output_path, f"epoch-{epoch_id}"), exist_ok=True)
                    save_video(video, os.path.join(test_output_path, f"epoch-{epoch_id}", f"{sanitized_prompt}.mp4"), fps=15, quality=5)
    


def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--spatial_lora_target_modules", type=str, default=None, help="Which layers LoRA is added to.")
    parser.add_argument("--temporal_lora_target_modules", type=str, default=None, help="Which layers LoRA is added to.")
    parser.add_argument("--train_type", type=str, default=None, help="Train type.")
    parser.add_argument("--spatial_lora_rank", type=int, default=None, help="Rank of LoRA.")
    parser.add_argument("--temporal_lora_rank", type=int, default=None, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--test_prompts", type=str, default=None, help="JSON list of test prompts to render at epoch end.")
    parser.add_argument("--test_output_path", type=str, default=None, help="Output directory to save test videos.")
    parser.add_argument("--sample_height", type=int, default=512, help="Height for test sample generation.")
    parser.add_argument("--sample_width", type=int, default=512, help="Width for test sample generation.")
    parser.add_argument("--encoder_type", type=str, default=None, help="Encoder type.")
    parser.add_argument("--proj_diff", type=float, default=0.15, help="Projection loss weight.")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser
