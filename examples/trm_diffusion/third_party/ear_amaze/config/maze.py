import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 1
    config.sample.num_batches_per_epoch = 1

    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 1

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


def maze_bagel_acc_w_llm_2gpu():
    # 1. 自动检测 GPU 数量 (如果没有传入)
    import torch
    gpu_number = torch.cuda.device_count()
    config = compressibility()
    
    # 动态获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 使用传入的参数，如果没有则使用默认值
    config.dataset_debug = "/media/raid/workspace/zhaoyanpeng/model/huggingface/hub/datasets--piekenius123--maze-dataset/snapshots/17fb8a102a779cec8e4beab0f93d8f490d06ae85"
    config.dataset = "/mnt/data/zhaoyanpeng/model/huggingface/hub/datasets--piekenius123--maze_dataset_debug/snapshots/f977fd2b667a180f8237b9741caa61ef157f0d52"
    #config.dataset = "/media/raid/workspace/zhaoyanpeng/model/huggingface/hub/datasets--piekenius123--maze-dataset/snapshots/4152acab71d543cc68f0cbc42eb6371ad2fbe975"
    #config.dataset = "/media/raid/workspace/zhaoyanpeng/model/huggingface/hub/datasets--piekenius123--maze-dataset_square/snapshots/bb248c22f7374ec5151192bda09e318729533593"
    config.pretrained.model = "/media/raid/workspace/zhaoyanpeng/model/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/570026eca23479ee7df5a6ce9fb50a835530da30"
        # 你也可以在这里写死你本地 SFT 的路径
    config.pretrained.checkpoint_path ="/mnt/data/zhaoyanpeng/model/maze_fullsft/model.safetensors"
    # config.resume_from = "/media/raid/workspace/zhaoyanpeng/code/flowgrpo/amaze/result/grpo_finetune/bagel_w_1_rewards_7_8/checkpoint-346"
       # config.pretrained.checkpoint_path = None 
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 10
    config.seed = 1
    config.train.timestep_shift = 3.0

    config.resolution = 1024
    # 基础单卡 Batch Size (受显存限制，假设单卡能跑 2)
    base_batch_size = 4
    
    # --- 约束 1: Sampler 必须能整除 ---
    # 逻辑：(GPU数量 * 单卡BS) 必须能被 num_image_per_prompt 整除
    # main.py 第 189 行: assert self.total_samples % self.k == 0
    total_parallel_capacity = gpu_number * base_batch_size

    config.sample.train_batch_size = base_batch_size

    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 2
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 2

    config.train.learning_rate = 5e-6
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1 #config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt // 2 if (config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt) > 1 else 1
    config.train.num_inner_epochs = 4
    config.train.beta = 0
    config.activation_checkpointing = True
    # config.train.beta = 0
    config.train.use_8bit_adam = False
    config.train.clip_range_lt = 0.28
    config.train.clip_range_gt = 0.2
    config.fsdp_optimizer_offload = True
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.3
    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = False
    config.lora_rank=64
    config.lora_alpha=128
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    #config.resume_from = "/media/raid/workspace/zhaoyanpeng/code/zzm/amaze/results/sft_20251120_204756/checkpoints/0004400"
    config.resume_from = None
    config.debug = False
    # samples_per_epoch = (config.sample.train_batch_size * world_size)// num_image_per_prompt * config.sample.num_batches_per_epoch
    config.num_epochs = 50
    config.save_freq = 6  # epoch
    config.eval_freq = 1
    # sample 参数
    config.run_name = "bagel_w_4_rewards_7_8_debug"
    config.sample.guidance_scale = 4.0
    config.sample.eval_guidance_scale = 4.0
    #config.sample.noise_level = 1.0
    config.save_dir = '/mnt/data/zhaoyanpeng/result/grpo_finetune/bagel_w_1_rewards_7_8_debug'
    config.reward_fn = {
        "maze_reward": 1.0,  # Custom reward function for maze
    }
    config.prompt_fn = "maze"
    config.per_prompt_stat_tracking = True
    #config.sample.noise_level=1.3
    return config


def maze_eval():
    config = maze_bagel_acc_w_llm_2gpu()
    # config.pretrained.model = "/media/raid/workspace/zhaoyanpeng/model/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/570026eca23479ee7df5a6ce9fb50a835530da30"
    # 覆盖特定参数
    config.pretrained.checkpoint_path = "/mnt/data/zhaoyanpeng/model/grpo_finetune/bagel_w_1_rewards_7_8_sft_fullsft_0110/checkpoint-400"
    config.sample.test_batch_size = 10  # 减小 batch size
    config.sample.eval_num_steps = 16  # 增加推理步数
    config.sample.num_attempts = 16    # 每个样本推理次数
    config.sample.filter_size_min = 5   # 筛选最小尺寸，如 5 表示 >= 5x5，None 表示不限制下限
    config.sample.filter_size_max = 10   # 筛选最大尺寸，如 10 表示 <= 10x10，None 表示不限制上限
    config.sample.samples_per_size = 50  # 每个尺寸选择的样本数，如 3 表示每个尺寸选前3个，None 表示选择所有
    config.sample.resolution = 1024
    config.logdir = "output_images/grpo"
    config.dataset = "/mnt/data/zhaoyanpeng/model/huggingface/hub/datasets--piekenius123--maze-test_full/snapshots/21272c1a7f4ac46210fc6d2432d5a9cc86c05258"
    config.pretrained.model = "/media/raid/workspace/zhaoyanpeng/model/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/570026eca23479ee7df5a6ce9fb50a835530da30"
    config.dataset_split="test_square_5_10"
    return config

def maze_sft_eval():
    config = maze_bagel_acc_w_llm_2gpu()
    # 覆盖特定参数
    config.pretrained.checkpoint_path = "results/sft_lora_r64_a128/checkpoints/0001000"
    config.sample.test_batch_size = 60  # 减小 batch size
    config.sample.eval_num_steps = 16  # 增加推理步数
    config.sample.num_attempts = 6    # 每个样本推理次数
    config.sample.filter_size_min = 5   # 筛选最小尺寸，如 5 表示 >= 5x5，None 表示不限制下限
    config.sample.filter_size_max = 5   # 筛选最大尺寸，如 10 表示 <= 10x10，None 表示不限制上限
    config.sample.samples_per_size = 50  # 每个尺寸选择的样本数，如 3 表示每个尺寸选前3个，None 表示选择所有
    config.sample.test_batch_size_per_size = 8  # 新参数：每个尺寸内的batch大小（用于避免OOM）
    config.sample.resolution = 1024
    config.use_lora = False
    # config.pretrained.lora_path = "/media/raid/workspace/zhaoyanpeng/code/amaze/results/tmp_20260106_040715"
    config.lora_rank = 64
    config.lora_alpha = 128
    config.logdir = "output_images/fullsft_hexagon_3"
    # config.dataset = "/mnt/data/zhaoyanpeng/model/huggingface/hub/datasets--piekenius123--maze-test_full/snapshots/21272c1a7f4ac46210fc6d2432d5a9cc86c05258"
    config.dataset_split="test"
    # config.sample.filter_shape="hexagon"
    config.dataset = "/home/zhaoyanpeng/Documents/mazes/maze-dataset"
    # config.dataset = "/media/raid/workspace/zhaoyanpeng/model/huggingface/hub/datasets--piekenius123--maze-dataset_square/snapshots/bb248c22f7374ec5151192bda09e318729533593"
    return config

def maze_sft_eval_circle():
    config = maze_bagel_acc_w_llm_2gpu()
    # 覆盖特定参数
    config.pretrained.checkpoint_path = "results/sft_lora_r64_a128/checkpoints/0001000"
    config.sample.test_batch_size = 60  # 减小 batch size
    config.sample.eval_num_steps = 16  # 增加推理步数
    config.sample.num_attempts = 6    # 每个样本推理次数
    config.sample.filter_size_min = 5   # 筛选最小尺寸，如 5 表示 >= 5x5，None 表示不限制下限
    config.sample.filter_size_max = 5   # 筛选最大尺寸，如 10 表示 <= 10x10，None 表示不限制上限
    config.sample.samples_per_size = 700  # 每个尺寸选择的样本数，如 3 表示每个尺寸选前3个，None 表示选择所有
    config.sample.test_batch_size_per_size = 6  # 新参数：每个尺寸内的batch大小（用于避免OOM）
    config.sample.resolution = 1024
    config.use_lora = False
    # config.pretrained.lora_path = "/media/raid/workspace/zhaoyanpeng/code/amaze/results/tmp_20260106_040715"
    config.lora_rank = 64
    config.lora_alpha = 128
    config.logdir = "/media/raid/workspace/zhaoyanpeng/code/flowgrpo/amaze/output_images/circle_6400_500/circle_3_16"
    # config.dataset = "/mnt/data/zhaoyanpeng/model/huggingface/hub/datasets--piekenius123--maze-test_full/snapshots/21272c1a7f4ac46210fc6d2432d5a9cc86c05258"
    config.is_circle=True
    config.dataset_split="test"
    config.reward_fn = {
        "maze_reward_circle": 1.0,  # Custom reward function for maze
    }
    # config.sample.filter_shape="hexagon"
    config.dataset = "/home/zhaoyanpeng/Documents/mazes/maze-dataset"
    # config.dataset = "/media/raid/workspace/zhaoyanpeng/model/huggingface/hub/datasets--piekenius123--maze-dataset_square/snapshots/bb248c22f7374ec5151192bda09e318729533593"
    return config



def get_config(name):
    return globals()[name]() # use command lines instead
    # 从环境变量读取参数（如果存在）
    save_dir = os.environ.get('SAVE_PATH')
    dataset = os.environ.get('DATASET_PATH')
    pretrained_model = os.environ.get('PRETRAINED_MODEL')

    # 如果函数支持参数且环境变量已设置，则传递参数
    func = globals()[name]
    if name == 'maze_qwenimage_edit_2gpu':
        return func(dataset=dataset, pretrained_model=pretrained_model, save_dir=save_dir)
    else:
        return func()

