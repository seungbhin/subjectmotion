import os
from torchvision.datasets.utils import download_url
import torch
import torchvision.models as torchvision_models
import timm
#from models import mocov3_vit
import math
import warnings
import sys
import cv2
from glob import glob
import os.path as osp
from PIL import Image
import argparse
import os
import numpy as np
from diffsynth.utils.core.utils.flow_viz import flow2rgb
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffsynth.utils.core.raft import RAFT
from diffsynth.utils.core.utils.flow_viz import flow_to_image
from diffsynth.utils.core.utils.utils import load_ckpt

# code from SiT repository
pretrained_models = {'last.pt'}

def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0'
        download_url(web_path, 'pretrained_models', filename=model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model

def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict

@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    #assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.

        architectures.append(architecture)
        encoder_types.append(encoder_type)

        
        if 'dinov2' in encoder_type:
            
            import timm
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
                #encoder = torch.hub.load('/home/user/.cache/torch/hub/facebookresearch_dinov2_main', f'dinov2_vit{model_config}14_reg', trust_repo=True, source='local')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
                #encoder = torch.hub.load('/home/user/.cache/torch/hub/facebookresearch_dinov2_main', f'dinov2_vit{model_config}14', trust_repo=True, source='local')

            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()

        encoders.append(encoder)
        
    return encoders, encoder_types, architectures

class SEARAFT_FlowProcessor:
    
    def __init__(self, checkpoint_path=None, checkpoint_url=None, device='cpu'):
        self.device = torch.device(device)
        self._init_config()
        self.model = self._load_model(checkpoint_path, checkpoint_url)

        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        
    def _init_config(self):
        """初始化配置参数"""
        # 从配置文件 spring-M.json 中读取的参数
        self.name = "spring-M"
        self.dataset = "spring"
        self.gpus = [0, 1, 2, 3, 4, 5, 6, 7]
        self.use_var = True
        self.var_min = 0
        self.var_max = 10
        self.pretrain = "resnet34"
        self.initial_dim = 64
        self.block_dims = [64, 128, 256]
        self.radius = 4
        self.dim = 128
        self.num_blocks = 2
        self.iters = 4
        self.image_size = [540, 960]
        self.scale = -1
        self.batch_size = 32
        self.epsilon = 1e-8
        self.lr = 4e-4
        self.wdecay = 1e-5
        self.dropout = 0
        self.clip = 1.0
        self.gamma = 0.85
        self.num_steps = 120000
        self.restore_ckpt = None
        self.coarse_config = None
    
    def _load_model(self, checkpoint_path, checkpoint_url):
        """加载模型"""
        if checkpoint_path is None and checkpoint_url is None:
            # 使用默认路径
            checkpoint_path = "./ckpts/SEA-RAFT/Tartan-C-T-TSKH-spring540x960-M/Tartan-C-T-TSKH-spring540x960-M.pth"
        
        if checkpoint_path is not None:
            model = RAFT(self)  # 直接传入self而不是args
            load_ckpt(model, checkpoint_path)
        else:
            model = RAFT.from_pretrained(checkpoint_url, args=self)  # 传入self
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def create_color_bar(self, height, width, color_map):
        """创建颜色条"""
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)
        color_bar = cv2.applyColorMap(gradient, color_map)
        return color_bar
    
    def add_color_bar_to_image(self, image, color_bar, orientation='vertical'):
        """为图像添加颜色条"""
        if orientation == 'vertical':
            return cv2.vconcat([image, color_bar])
        else:
            return cv2.hconcat([image, color_bar])
    
    def vis_heatmap(self, name, image, heatmap):
        """可视化热图"""
        heatmap = heatmap[:, :, 0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = (heatmap * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = image * 0.3 + colored_heatmap * 0.7
        
        height, width = image.shape[:2]
        color_bar = self.create_color_bar(50, width, cv2.COLORMAP_JET)
        overlay = overlay.astype(np.uint8)
        combined_image = self.add_color_bar_to_image(overlay, color_bar, 'vertical')
        cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
    
    def get_heatmap(self, info):
        """从信息中生成热图"""
        raw_b = info[:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = info[:, :2].softmax(dim=1)              
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.var_max)
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.var_min, max=0)
        heatmap = (log_b * weight).sum(dim=1, keepdim=True)
        return heatmap
    
    def forward_flow(self, image1, image2):
        """前向计算光流"""
        output = self.model(image1, image2, iters=self.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final
    
    def calc_flow(self, image1, image2):
        """计算光流，包含尺度变换"""
        img1 = F.interpolate(image1, scale_factor=2 ** self.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** self.scale, mode='bilinear', align_corners=False)
        H, W = img1.shape[2:]
        flow, info = self.forward_flow(img1, img2)
        flow_down = F.interpolate(flow, scale_factor=0.5 ** self.scale, mode='bilinear', align_corners=False) * (0.5 ** self.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** self.scale, mode='area')
        return flow_down, info_down
    
    def process_images(self, image1, image2, output_path='./output/'):
        os.makedirs(output_path, exist_ok=True)
        
        # 如果输入是numpy数组，转换为tensor
        if isinstance(image1, np.ndarray):
            image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)[None].to(self.device)
        if isinstance(image2, np.ndarray):
            image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)[None].to(self.device)
        
        # 计算光流
        flow, info = self.calc_flow(image1, image2)
        
        # 生成光流可视化
        flow_vis = flow_to_image(flow[0].detach().permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
        cv2.imwrite(f"{output_path}/flow.jpg", flow_vis)
        
        # 生成热图
        heatmap = self.get_heatmap(info)
        self.vis_heatmap(f"{output_path}/heatmap.jpg", 
                        image1[0].detach().permute(1, 2, 0).cpu().numpy(), 
                        heatmap[0].detach().permute(1, 2, 0).cpu().numpy())
        
        return {
            'flow': flow,
            'heatmap': heatmap,
            'flow_vis': flow_vis
        }
    
    def process_image_files(self, image1_path, image2_path, output_path='./output/'):
        # 读取图像
        image1 = cv2.imread(image1_path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.imread(image2_path)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        
        return self.process_images(image1, image2, output_path)

    def video_flow(self, video_frames, device):
        import torchvision.transforms as transforms
        
        # 将设备设置为传入的device
        self.device = device
        self.model = self.model.to(device)
        
        f_list = []
        flowviz_list = []
        
        # 定义图像预处理
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
        ])
        
        # 处理相邻帧对
        for i in range(len(video_frames) - 1):
            # 获取当前帧和下一帧
            frame1 = video_frames[i]
            frame2 = video_frames[i + 1]
            
            # 转换为tensor
            img1_tensor = transform(frame1).unsqueeze(0).to(device)  # [1, 3, H, W]
            img2_tensor = transform(frame2).unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # 将[0,1]范围转换为[0,255]范围，因为模型期望这个范围
            img1_tensor = img1_tensor * 255.0
            img2_tensor = img2_tensor * 255.0
            
            # 计算光流
            with torch.no_grad():
                flow, info = self.calc_flow(img1_tensor, img2_tensor)
            
            # 添加到列表
            f_list.append(flow.squeeze(0))  # 移除batch维度，形状变为[2, H, W]
            
            # 生成光流可视化
            flow_vis = flow_to_image(
                flow[0].detach().permute(1, 2, 0).cpu().numpy(), 
                convert_to_bgr=True
            )
            flowviz_list.append(flow_vis)
        
        return f_list, flowviz_list

    def tensor_flow(self, video_tensor):
        from einops import rearrange
        
        # 确保video_tensor在正确的设备上
        video_tensor = video_tensor.to(torch.float32)
        
        # 如果输入是[0,1]范围，转换为[0,255]范围，因为模型期望这个范围
        if video_tensor.max() <= 1.0:
            video_tensor = video_tensor * 255.0
        
        f_list = []
        flowviz_list = []
        
        # 处理相邻帧对
        for i in range(video_tensor.shape[0] - 1):
            # 获取当前帧和下一帧
            frame1 = video_tensor[i:i+1]  # [1, C, H, W]
            frame2 = video_tensor[i+1:i+2]  # [1, C, H, W]
            
            # 计算光流 - 移除torch.no_grad()以保持梯度
            flow, info = self.calc_flow(frame1, frame2)
            
            # 添加到列表
            f_list.append(flow.squeeze(0))  # 移除batch维度，形状变为[2, H, W]
            
            # 生成可视化（使用detach避免影响梯度）
            flow_vis = flow_to_image(
                flow[0].detach().permute(1, 2, 0).cpu().numpy(), 
                convert_to_bgr=True
            )
            flowviz_list.append(flow_vis)
        
        return f_list, flowviz_list
