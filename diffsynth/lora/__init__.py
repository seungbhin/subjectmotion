import torch

# 全局变量：保存所有LoRA加载器共享的原始权重
_GLOBAL_ORIGINAL_WEIGHTS = {}



class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
        # 使用全局变量保存原始权重，所有实例共享
    
    
    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict


    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        updated_num = 0
        saved_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name, module in model.named_modules():
            if name in lora_name_dict:
                # 保存原始权重到全局变量（如果还没有保存的话）
                if name not in _GLOBAL_ORIGINAL_WEIGHTS:
                    state_dict = module.state_dict()
                    _GLOBAL_ORIGINAL_WEIGHTS[name] = state_dict["weight"].clone().detach()
                    saved_num += 1
                    print(f"Saved original weight for layer: {name}")
                
                weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=self.device, dtype=self.torch_dtype)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict)
                updated_num += 1
                print(f"Updated layer: {name}")
        
        print(f"{updated_num} tensors are updated by LoRA.")
        print(f"{saved_num} original weights are saved.")
    
    def restore(self, model: torch.nn.Module):
        restored_num = 0
        for name, module in model.named_modules():
            if name in _GLOBAL_ORIGINAL_WEIGHTS:
                # 还原原始权重
                state_dict = module.state_dict()
                original_weight = _GLOBAL_ORIGINAL_WEIGHTS[name].to(device=self.device, dtype=self.torch_dtype)
                state_dict["weight"] = original_weight
                module.load_state_dict(state_dict)
                restored_num += 1
                print(f"Restored original weight for layer: {name}")

        print(f"{restored_num} tensors are restored to original weights.")
    
    def clear_original_weights(self):
        """
        清除全局保存的原始权重，释放内存
        """
        _GLOBAL_ORIGINAL_WEIGHTS.clear()
        print("Cleared all saved original weights.")
    
    def save_original_weights(self, model: torch.nn.Module, target_layer_names=None):
        """
        手动保存指定层的原始权重
        
        Args:
            model: 模型
            target_layer_names: 要保存的层名列表，如果为None则保存所有有权重的层
        """
        saved_num = 0
        for name, module in model.named_modules():
            # 如果指定了目标层名，只保存指定的层
            if target_layer_names is not None and name not in target_layer_names:
                continue
                
            # 检查模块是否有weight参数
            if hasattr(module, 'weight') and module.weight is not None:
                if name not in _GLOBAL_ORIGINAL_WEIGHTS:
                    _GLOBAL_ORIGINAL_WEIGHTS[name] = module.weight.clone().detach()
                    saved_num += 1
                    print(f"Manually saved original weight for layer: {name}")
        
        print(f"Manually saved {saved_num} original weights.")


# 全局函数：管理全局原始权重
def get_global_original_weights_info():
    """
    获取全局原始权重的信息
    """
    print(f"Global original weights contains {len(_GLOBAL_ORIGINAL_WEIGHTS)} layers:")
    for name in _GLOBAL_ORIGINAL_WEIGHTS.keys():
        print(f"  - {name}")
    return list(_GLOBAL_ORIGINAL_WEIGHTS.keys())

def clear_global_original_weights():
    """
    清除全局原始权重
    """
    _GLOBAL_ORIGINAL_WEIGHTS.clear()
    print("Cleared all global original weights.")

def has_global_original_weights():
    """
    检查是否有保存的全局原始权重
    """
    return len(_GLOBAL_ORIGINAL_WEIGHTS) > 0
