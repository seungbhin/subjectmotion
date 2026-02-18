import torch, os, json
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, ModelLogger, launch_training_task, wan_parser
from datetime import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, 
        spatial_lora_target_modules=None, 
        temporal_lora_target_modules=None,
        spatial_lora_rank=None,
        temporal_lora_rank=None,
        use_gradient_checkpointing=False, #True
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        train_type=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu",  model_configs=model_configs)
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        # Add LoRA to the base models
        if lora_base_model is not None:
            if spatial_lora_target_modules is not None:
                model = self.add_lora_to_model(
                    getattr(self.pipe, lora_base_model),
                    target_modules=spatial_lora_target_modules.split(","),
                    lora_rank=spatial_lora_rank,
                    adapter_name="spatial_lora"
                )
                setattr(self.pipe, lora_base_model, model)
            
            if temporal_lora_target_modules is not None:
                model = self.add_lora_to_model(
                    getattr(self.pipe, lora_base_model),
                    target_modules=temporal_lora_target_modules.split(","),
                    lora_rank=temporal_lora_rank,
                    adapter_name="temporal_lora"
                )
                setattr(self.pipe, lora_base_model, model)

            # 最终统计
            self.temporal_lora_params = []
            self.spatial_lora_params = []
            print(f"\n[INFO] LoRA添加完成，最终统计:")
            total_lora_params = 0
            total_trainable_params = 0
            for name, param in getattr(self.pipe, lora_base_model).named_parameters():
                #if param.requires_grad:
                    total_trainable_params += param.numel()
                    if 'lora' in name.lower():
                        total_lora_params += param.numel()
                        param.requires_grad = True
                        if "temporal_lora" in name.lower():
                            self.temporal_lora_params.append(param)
                        elif "spatial_lora" in name.lower():
                            self.spatial_lora_params.append(param)
                        #print(f"  - LoRA参数: {name}, shape: {param.shape}, 参数量: {param.numel()}")
            
            total_lora_params = 0
            total_trainable_params = 0
            for name, param in getattr(self.pipe, lora_base_model).named_parameters():
                if param.requires_grad:
                    total_trainable_params += param.numel()
                    if 'lora' in name.lower():
                        total_lora_params += param.numel()
                        print(f"  - LoRA参数: {name}, shape: {param.shape}, 参数量: {param.numel()}")
            print(f"[INFO] 总LoRA参数量: {total_lora_params:,}")
            print(f"[INFO] 总可训练参数量: {total_trainable_params:,}")
            print(f"[INFO] LoRA参数占比: {total_lora_params/total_trainable_params*100:.2f}%")
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        # Store the base model name for later use
        self.lora_base_model = lora_base_model
        
    def adjust_lora_alpha(self, adapter_name, new_scale_factor):
        """
        调整指定LoRA adapter的scaling factor
        
        Args:
            adapter_name (str): adapter名称，如 "spatial_lora" 或 "temporal_lora"
            new_scale_factor (float): 新的缩放因子，相对于初始的 lora_alpha/r
        """
        if self.lora_base_model is None:
            print("[WARNING] 没有LoRA base model，无法调整alpha")
            return
            
        base_model = getattr(self.pipe, self.lora_base_model)
        adjusted_layers = 0
        
        # 遍历所有模块，找到LoRA层
        for name, module in base_model.named_modules():
            # 检查是否是LoRA层（具有scaling属性）
            if hasattr(module, 'scaling') and hasattr(module, 'set_scale'):
                if adapter_name in module.scaling:
                    module.set_scale(adapter_name, new_scale_factor)
                    adjusted_layers += 1
                    #print(f"[INFO] 调整LoRA层 {name} 的 {adapter_name} scaling为: {module.scaling[adapter_name]:.6f}")
        
        print(f"[INFO] 总共调整了 {adjusted_layers} 个 {adapter_name} 层的scaling")
    
    def get_lora_scaling_info(self):
        """获取当前所有LoRA adapter的scaling信息"""
        if self.lora_base_model is None:
            print("[WARNING] 没有LoRA base model")
            return
            
        base_model = getattr(self.pipe, self.lora_base_model)
        scaling_info = {}
        
        for name, module in base_model.named_modules():
            if hasattr(module, 'scaling'):
                for adapter_name, scaling_value in module.scaling.items():
                    if adapter_name not in scaling_info:
                        scaling_info[adapter_name] = []
                    scaling_info[adapter_name].append((name, scaling_value))
        
        # 打印信息
        for adapter_name, layer_info in scaling_info.items():
            print(f"\n[INFO] {adapter_name} scaling信息:")
            for layer_name, scaling_value in layer_info:
                print(f"  {layer_name}: {scaling_value:.6f}")
        
        return scaling_info
    
    def forward_preprocess(self, data, train_type=None):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        if train_type == "Subject":
            # CFG-unsensitive parameters
            inputs_shared = {
                # Assume you are using this pipeline for inference,
                # please fill in the input parameters.
                "input_video": data["video"],
                "mask": data["mask"],
                "height": data["video"][0].size[1],
                "width": data["video"][0].size[0],
                "num_frames": len(data["video"]),
                # Please do not modify the following parameters
                # unless you clearly know what this will cause.
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
            }
        elif train_type == "Motion_0":
            # 从视频帧中随机选择一帧的索引
            ran_idx = torch.randint(0, len(data["video"]), (1,)).item()
            print(f"ran_idx for spatial LoRA: {ran_idx}")
            inputs_shared = {
                # Assume you are using this pipeline for inference,
                # please fill in the input parameters.
                "input_video": [data["video"][ran_idx]],  # 必须包装成列表！
                "height": data["video"][0].size[1],
                "width": data["video"][0].size[0],
                "num_frames": 1,  # 设置为1帧
                # Please do not modify the following parameters
                # unless you clearly know what this will cause.
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
            }
        elif train_type == "Motion_1":            
            inputs_shared = {
                # Assume you are using this pipeline for inference,
                # please fill in the input parameters.
                "input_video": data["video"],
                "height": data["video"][0].size[1],
                "width": data["video"][0].size[0],
                "num_frames": len(data["video"]),
                # Please do not modify the following parameters
                # unless you clearly know what this will cause.
                "cfg_scale": 1,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
            }
            print("num_frames:",len(data["video"]))
            print("inputs_shared has been done for motion_1!")
        
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi} #input_video->latents新增了一些keys
    
    
    def forward(self, data, inputs=None, zs=None, loss_fn=None, z_dims=None, enc_type=None, proj_diff=None, mask_img=None, train_type=None, flow_list=None, flow_processor=None):
    
        if train_type == "Subject":
            if inputs is None: inputs = self.forward_preprocess(data, train_type=train_type)
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            loss, loss_proj, zs_tilde = self.pipe.training_loss(
                **models, **inputs, 
                zs=zs, 
                loss_fn=loss_fn,
                enc_type=enc_type,
                z_dims=z_dims,
                mask_img=mask_img,
                train_type=train_type,
            )
            return loss, loss_proj, zs_tilde

        elif train_type == "Motion_0":
            # Motion_0阶段：只使用spatial_lora，关闭temporal_lora
            '''
            self.adjust_lora_alpha("spatial_lora", 1.0)
            self.adjust_lora_alpha("temporal_lora", 0.0)
            '''
            inputs = self.forward_preprocess(data, train_type=train_type)
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
                    
            loss_spatial = self.pipe.training_loss(
                **models, **inputs,
                train_type=train_type,
            )
            return loss_spatial

        elif train_type == "Motion_1":    
            inputs = self.forward_preprocess(data, train_type=train_type)
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            loss_temporal, loss_flow = self.pipe.training_loss(
                **models, **inputs,
                train_type=train_type,
                flow_list=flow_list,
                flow_processor=flow_processor
                )
            return loss_temporal, loss_flow
                

if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    # 统一时间键
    time_key = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = os.path.join(args.output_path, time_key)

    # ===== GPU/CUDA 环境信息 =====
    print("=" * 60)
    print("[GPU INFO] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("[GPU INFO] torch.cuda.is_available:", torch.cuda.is_available())
    try:
        dc = torch.cuda.device_count()
    except Exception:
        dc = 0
    print("[GPU INFO] torch.cuda.device_count:", dc)
    if torch.cuda.is_available() and dc > 0:
        for i in range(dc):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = "<unknown>"
            print(f"[GPU INFO] CUDA:{i} -> {name}")
    print("=" * 60)
    dataset = VideoDataset(args=args) #[image:numpy],[mask:numpy],prompt:str
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        spatial_lora_target_modules=args.spatial_lora_target_modules,
        temporal_lora_target_modules=args.temporal_lora_target_modules,
        spatial_lora_rank=args.spatial_lora_rank,
        temporal_lora_rank=args.temporal_lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        train_type = args.train_type,
    )
    
    model_logger = ModelLogger(
        os.path.join(run_root, "loras"),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    # 可选的测试配置
    test_prompts = json.loads(args.test_prompts) if getattr(args, "test_prompts", None) else None
    test_output_path = os.path.join(run_root, "test_results") if test_prompts is not None else None
    visuliaze_path = os.path.join(run_root, "visuliaze_path")
    test_height = args.sample_height if getattr(args, "sample_height", None) else None
    test_width = args.sample_width if getattr(args, "sample_width", None) else None
    encoder_type = args.encoder_type if getattr(args, "encoder_type", None) else None
    proj_diff = args.proj_diff if getattr(args, "proj_diff", None) else None
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    train_type = args.train_type if getattr(args, "train_type", None) else None

    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        test_prompts=test_prompts,
        test_output_path=test_output_path,
        test_height=test_height,
        test_width=test_width,
        enc_type=encoder_type,
        visuliaze_path=visuliaze_path,
        proj_diff=proj_diff,
        train_type=train_type,
    )
