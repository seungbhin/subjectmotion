import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import math
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, zs_tilde=None,  zs=None, mask=None, type=None):
        #zs_tilde is the latents of diffsuion
        epsilon = 1e-8
        # projection loss
        if mask is not None:
                print("Use mask to do REPA!!!!")
                proj_loss = 0.
                bsz, token, d = zs[0].shape
                height = width = int(np.sqrt(token))
                mask = mask[0].unsqueeze(0)  # [1, C, H, W]
                mask = F.interpolate(mask, size=(height, width), mode='bilinear', align_corners=False) #1,3,32,32
                #mask = TF.gaussian_blur(mask, kernel_size=15, sigma=3)
                mask = mask.bool()
                mask = mask[:, 0, :, :] #1,32,32
                num_masked_elements = mask.sum()
                mask = mask.reshape(1, token) #1,1024
                for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                    for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)): #t,d: 1024,768
                        #print(f"z_tilde_j before normalize最大值: {torch.max(z_tilde_j).item():.4f}, z_tilde_j before normalize最小值: {torch.min(z_tilde_j).item():.4f}, z_tilde_j before normalize均值: {torch.mean(z_tilde_j).item():.4f}")
                        #print(f"z_j before normalize最大值: {torch.max(z_j).item():.4f}, z_j before normalize最小值: {torch.min(z_j).item():.4f}, z_j before normalize均值: {torch.mean(z_j).item():.4f}")
                        z_tilde_j = z_tilde_j + epsilon
                        z_j = z_j + epsilon
                        z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                        z_j = torch.nn.functional.normalize(z_j, dim=-1)
                        #out = tat(z_tilde_j = z_tilde_j, z_j = z_j)
                        #out = z_j
                        mask = mask[0] #1024
                        z_tilde_j = z_tilde_j[mask] #token,d
                        z_j = z_j[mask] #token,d
                        print("z_tilde_j:",z_tilde_j.size())
                        print("z_j:",z_j.size())
                        #out = tat(z_tilde_j = z_tilde_j, z_j = z_j)
                        loss = calculate_cos_proj_loss(z_tilde_j, z_j)
                        #loss = feature_sim + 1
                        proj_loss = proj_loss + loss

                proj_loss /= (len(zs))
                return proj_loss
        else:
                proj_loss = 0.
                bsz = zs[0].shape[0]
                for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                    for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)): #b,t,d: 1,1024,1536
                        #print(f"z_tilde_j before normalize最大值: {torch.max(z_tilde_j).item():.4f}, z_tilde_j before normalize最小值: {torch.min(z_tilde_j).item():.4f}, z_tilde_j before normalize均值: {torch.mean(z_tilde_j).item():.4f}")
                        #print(f"z_j before normalize最大值: {torch.max(z_j).item():.4f}, z_j before normalize最小值: {torch.min(z_j).item():.4f}, z_j before normalize均值: {torch.mean(z_j).item():.4f}")
                        print("z_tilde_j:",z_tilde_j.size())
                        print("z_j:",z_j.size())
                        z_tilde_j = z_tilde_j + epsilon
                        z_j = z_j + epsilon
                        z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                        z_j = torch.nn.functional.normalize(z_j, dim=-1)
                        
                        feature_sim = calculate_cos_proj_loss(z_tilde_j, z_j)
                        #print("feature_sim:",feature_sim)
                        #similarity_matrix_z_tilde = calculate_internal_cos_similarity(z_tilde_j)  # [token_num, token_num]
                        #similarity_matrix_z_j = calculate_internal_cos_similarity(z_j)  # [token_num, token_num]
                        #print(f"z_tilde_j相似度矩阵形状: {similarity_matrix_z_tilde.shape}")
                        #print(f"z_j相似度矩阵形状: {similarity_matrix_z_j.shape}")
                        #matrix_diff = F.mse_loss(similarity_matrix_z_tilde, similarity_matrix_z_j, reduction='mean')
                        #print("matrix_diff:",matrix_diff)
                        #loss = (feature_sim + 1) * 0.5 + (matrix_diff) * 0.5
                        loss = feature_sim
                        #loss = matrix_diff
                        proj_loss = proj_loss + loss
                        print("loss:",loss)
                        #proj_loss += calculate_cos_proj_loss(z_tilde_j, z_j)
                proj_loss /= (len(zs))
                return proj_loss
            
        
def calculate_cos_proj_loss(a, b):
    #a = torch.nn.functional.normalize(a, dim=-1) 
    #b = torch.nn.functional.normalize(b, dim=-1) 
    loss = - F.cosine_similarity(a, b, dim=-1).mean()
    return loss

def calculate_mse_proj_loss(a, b):
    loss = F.mse_loss(a, b, reduction='mean')
    return loss 

def tat(z_tilde_j, z_j):
    epsilon = 1e-8
    q = z_tilde_j  # (1024,768)
    k = z_j  # (1024,768)
    v = z_j  # (1024,768)
    
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (1024,1024)
    attn = attn + epsilon
    # Softmax
    attn = torch.softmax(attn, dim=-1)  # (1024,1024)
    out = torch.matmul(attn, v)  # (1024,768)
    return out

def masked_mse_loss(model_pred, target, mask):
    _, channel, _, height, width = model_pred.shape  #noise_pred: torch.Size([1, 16, 1, 64, 64]) b,c,f,h,w
    
    model_pred = rearrange(model_pred, "b c f h w -> (b f) c h w")
    target = rearrange(target, "b c f h w -> (b f) c h w")
    
    mask = F.interpolate(mask, size=(height, width), mode='bilinear', align_corners=False)
    #mask = TF.gaussian_blur(mask, kernel_size=15, sigma=3)

    mask = mask.bool() # (b, c, h, w)
    mask = mask[:, 0, :, :].detach() #1,384,384
    # Apply the mask by setting non-masked elements to zero
    #print(f"model_pred最大值: {torch.max(model_pred).item():.4f}, model_pred before mask最小值: {torch.min(model_pred).item():.4f}, model_pred before mask均值: {torch.mean(model_pred).item():.4f}")
    masked_model_pred = model_pred * mask
    masked_target = target * mask
    #print(f"masked_model_pred最大值: {torch.max(masked_model_pred).item():.4f}, masked_model_pred最小值: {torch.min(masked_model_pred).item():.4f}, masked_model_pred均值: {torch.mean(masked_model_pred).item():.4f}")
    #print(f"masked_target最大值: {torch.max(masked_target).item():.4f}, masked_target最小值: {torch.min(masked_target).item():.4f}, masked_target均值: {torch.mean(masked_target).item():.4f}")

    #mask2 = ~mask
    #unmasked_model_pred = model_pred * mask2
    #unmasked_target = target * mask2
    
    # Compute the MSE loss on the masked tensors
    # Note: Since mse_loss computes the mean over all elements and we want the mean only over masked elements,
    # we need to adjust the reduction manually.
    masked_loss = F.mse_loss(masked_model_pred, masked_target, reduction='sum')
    #unmasked_loss = F.mse_loss(unmasked_model_pred, unmasked_target, reduction='sum')
    
    # Count the number of masked elements
    num_masked_elements = mask.sum().detach().item()
    #num_unmasked_elements = mask2.sum().detach().item()
    #print("num_masked_elements:",num_masked_elements)
    #print("num_unmasked_elements:",num_unmasked_elements)

    # Calculate the average only over masked elements
    if num_masked_elements > 0:
        mse_loss = masked_loss / num_masked_elements
        mse_loss = mse_loss / channel
    else:
        mse_loss = torch.tensor(0.0).to(model_pred.device)
    
    '''
    if num_unmasked_elements > 0:
        mse_unmasked_loss = unmasked_loss / num_unmasked_elements
        mse_unmasked_loss = mse_unmasked_loss / channel
    else:
        mse_unmasked_loss = torch.tensor(0.0).to(model_pred.device)

    # 🔥 **按总像素数平均两个loss**
    total_pixels = num_masked_elements + num_unmasked_elements
    if total_pixels > 0:
        # 直接用原始的sum loss除以总像素数
        combined_loss = (masked_loss + unmasked_loss) / total_pixels
        combined_loss = combined_loss / channel
    else:
        combined_loss = torch.tensor(0.0).to(model_pred.device)
    '''
    
    return mse_loss

def calculate_l1_proj_loss(a, b):
    """计算L1损失 (平均绝对误差)"""
    loss = F.l1_loss(a, b, reduction='mean')
    return loss 

def flow_loss(flow_tilde_list, flow_list):
    # 使用float32进行损失计算
    print("flow_tilde_list.dtype:",flow_tilde_list.dtype)
    print("flow_list.dtype:",flow_list.dtype)
    loss = 0.
    f, C, H, W = flow_list.shape  # 6,2,384,384
    print("F,C,H,W:", f, C, H, W)
    for flow_tilde, flow in zip(flow_tilde_list, flow_list):
        for c in range(C):
            if c == 0:
                flow_tilde_norm = flow_tilde[c] / W
                flow_norm = flow[c] / W  
            elif c == 1:
                flow_tilde_norm = flow_tilde[c] / H
                flow_norm = flow[c] / H
            else:
                raise ValueError(f"Invalid channel index: {c}")
            #l = calculate_mse_proj_loss(flow_norm, flow_tilde_norm)
            l = calculate_l1_proj_loss(flow_norm, flow_tilde_norm)
            loss = loss + l
    
    loss = loss / len(flow_tilde_list)
    print("loss:", loss)
    return loss

def calculate_internal_cos_similarity(features):
    """
    计算特征内部的余弦相似度矩阵
    Args:
        features: shape [token_num, dim] 的特征张量
    Returns:
        相似度矩阵 [token_num, token_num]，表示所有patch之间的余弦相似度
    """
    # features shape: [token_num, dim]
    # 计算所有token之间的余弦相似度矩阵
    normalized_features = F.normalize(features, dim=-1)  # [token_num, dim]
    similarity_matrix = torch.matmul(normalized_features, normalized_features.transpose(-2, -1))  # [token_num, token_num]
    
    # 返回完整的相似度矩阵
    return similarity_matrix
