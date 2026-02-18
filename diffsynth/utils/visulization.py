import os
import torch
import matplotlib.pyplot as plt
import math
from PIL import Image
import numpy as np
from einops import rearrange
import cv2

def visulize_feature(z, tokensize, output_dir, output_keyword):
    # z shape: (16,1024,768)
    z = torch.nn.functional.normalize(z, dim=-1)  # 在特征维度上归一化 (16,1024,768)
    z = z.max(dim=-1)[0]  # 在特征维度上取最大值 (16,1024)
    z = z[0]  # 取第一个时间步 (1024)
    z = z.reshape(tokensize, tokensize)  # 重塑为方形特征图 (32,32)
    
    # 对特征图进行归一化到[0,1]范围
    z_min = z.min()
    z_max = z.max()
    z_normalized = (z - z_min) / (z_max - z_min + 1e-8)  # 避免除零
    
    plt.figure(figsize=(8, 8))
    plt.imshow(z_normalized.detach().cpu().float().numpy(), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()  # 添加颜色条以显示数值范围
    plt.axis('off')
    plt.title('Normalized Feature Map')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'feature_vis_{output_keyword}.png'))
    plt.close()
    
def visulize_feature_video(z, tokensize, output_dir, output_keyword):
    """
    将视频特征可视化并拼接成长图，所有通道取均值
    Args:
        z: 特征张量，shape为(16,1024,768)
        tokensize: token的大小，用于重塑特征图
        output_dir: 输出目录
        output_keyword: 输出文件的关键字
    """
    import torch
    import numpy as np
    from einops import rearrange
    import cv2
    import os
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备存储所有帧的列表
    all_frames = []
    
    # 处理每一帧
    for frame_idx in range(z.shape[0]):  # 遍历16帧
        # 获取当前帧的特征
        z_frame = z[frame_idx]  # (1024,768)
        
        # 将特征重塑为方形图像
        z_frame = rearrange(z_frame, '(h w) c -> c h w', h=int(np.sqrt(z_frame.shape[0])))  # (768,32,32)
        
        # 对所有通道取最大值
        feature_map = torch.max(z_frame, dim=0)[0]  # (32,32)
        
        # 归一化到0-255范围
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255
        feature_map = feature_map.cpu().float().numpy().astype(np.uint8)
        
        # 转换为3通道图像（灰度图复制到3个通道）
        feature_map_rgb = np.stack([feature_map] * 3, axis=-1)  # (32,32,3)
        
        # 应用热力图颜色映射（可选）
        feature_map_color = cv2.applyColorMap(feature_map, cv2.COLORMAP_VIRIDIS)
        
        # 调整大小到指定的token size
        feature_map_color = cv2.resize(feature_map_color, (tokensize, tokensize), interpolation=cv2.INTER_LINEAR)
        
        # 添加到帧列表
        all_frames.append(feature_map_color)
    
    # 水平拼接所有帧
    concat_image = np.hstack(all_frames)
    
    # 保存拼接后的图像
    output_path = os.path.join(output_dir, f'{output_keyword}_concat.png')
    cv2.imwrite(output_path, concat_image)  # 已经是BGR格式，不需要转换
    
    # 可选：同时保存单独的帧
    for i, frame in enumerate(all_frames):
        frame_path = os.path.join(output_dir, f'{output_keyword}_frame_{i:02d}.png')
        cv2.imwrite(frame_path, frame)
    
    return concat_image

def save_mask(mask, save_dir, step):
    import matplotlib.pyplot as plt
    import os
    
    # 确保是CPU张量并且分离计算图
    mask = mask.detach().cpu()
    
    # 获取帧数
    num_frames = mask.shape[1]
    
    # 创建子图
    plt.figure(figsize=(2*num_frames, 2))
    
    # 为每一帧创建一个子图
    for f in range(1):
        plt.subplot(1, num_frames, f+1)
        # 取第一个batch和第一个通道
        mask_vis = mask[0, f, 0].numpy()
        plt.imshow(mask_vis, cmap='gray')
        plt.axis('off')
        plt.title(f'Frame {f}')
    
    plt.tight_layout()
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'mask_step_{step}.png'))
    plt.close() 

def save_frames_and_concat(raw_video, output_dir, output_keyword):
    import torch
    import cv2
    import numpy as np
    import os
    
    # 确保save_path目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换tensor到numpy并调整为正确的图像格式
    frames = []
    for i in range(16):
        # 获取单帧图像并转换为numpy
        frame = raw_video[i].permute(1,2,0)  # (3,128,128) -> (128,128,3)
        frame = frame.cpu().numpy()
        
        # 将值范围从[-1,1]转换到[0,255]
        frame = ((frame + 1.0) * 127.5).astype(np.uint8)
        
        # BGR转RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
    
    # 水平拼接所有帧
    concat_img = np.hstack(frames)
    
    # 保存拼接后的长图
    cv2.imwrite(os.path.join(output_dir, f'concat_frames_{output_keyword}.png'), concat_img)

def visualize_frames_concat(video_list, output_dir, step):
    import torch
    import torchvision.utils as vutils
    import os
    
    # 创建保存目录
    save_dir = os.path.join(output_dir, f'video_frames_step_{step}')
    frames_dir = os.path.join(save_dir, 'frames')  # 创建单独的帧目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)  # 创建保存单帧的目录
    
    all_frames = []
    
    # 遍历video_list中的每个视频
    for video_idx, video in enumerate(video_list):
        # 调整维度顺序 [1, 3, 1, 384, 384] -> [1, 1, 3, 384, 384]
        video = video.permute(0, 2, 1, 3, 4)
        
        # 去掉多余的维度 -> [1, 3, 384, 384]
        video = video.squeeze(1)
        
        # 将像素值范围从[-1, 1]调整到[0, 1]
        video = (video + 1.0) / 2.0
        
        # 保存单独的帧
        for frame_idx in range(video.size(0)):
            frame = video[frame_idx]  # [3, 384, 384]
            # 保存单帧
            vutils.save_image(
                frame,
                os.path.join(frames_dir, f'video_{video_idx}_frame_{frame_idx:04d}.png'),
                normalize=False
            )
        
        # 将当前视频的帧添加到列表中（用于创建长图）
        all_frames.append(video)
    
    # 在第一个维度（batch维度）上拼接所有帧
    all_frames = torch.cat(all_frames, dim=0)  # [N, 3, 384, 384] 其中N是总帧数
    
    # 创建水平排列的网格图像
    grid = vutils.make_grid(
        all_frames,
        nrow=all_frames.size(0),  # 所有帧在一行中显示
        padding=2,  # 帧之间的间隔
        normalize=False
    )
    
    # 保存拼接后的长图
    vutils.save_image(
        grid,
        os.path.join(save_dir, f'all_frames_concat_{step}.png')
    )
    
    return grid, all_frames

# 使用示例：
# video_list = decode_latents(pred_x_0)  # [[1, 3, 1, 384, 384]]
# grid, frames = visualize_frames_concat(video_list, output_dir, global_step)

