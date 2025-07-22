class Config:
    def __init__(self):
        # 设备配置
        self.device = 'cuda:3'
        
        # 训练/测试配置
        self.is_training = 'True'
        
        # 数据路径配置
        self.data_train_path = 'tianchi_data/'
        self.data_val_path = 'tianchi_data/'
        self.data_test_path = 'tianchi_data/'
        
        # 序列长度配置
        self.input_length = 10
        self.real_length = 30
        self.total_length = 30
        
        # 图像配置
        self.img_width = 560
        self.img_height = 480
        self.img_channel = 1
        self.patch_size = 1
        
        # 损失函数配置
        self.alpha = 1.0
        self.factor = 70.0
        
        # 模型配置
        self.model_name = 'predrnn'
        self.dataset = 'radar'
        self.num_workers = 0
        self.num_hidden = 64
        self.num_layers = 4
        self.num_heads = 4
        self.filter_size = (5, 5)
        self.stride = 1
        
        # 训练配置
        self.lr = 1e-3
        self.lr_decay = 0.90
        self.delay_interval = 2000.0
        self.batch_size = 2
        self.max_iterations = 80000
        self.max_epoches = 200000
        
        # 显示和保存配置
        self.display_interval = 1
        self.test_interval = 5000
        self.snapshot_interval = 10000
        self.num_save_samples = 50
        self.n_gpu = 1
        
        # 模型路径配置
        self.pretrained_model = ''
        self.perforamnce_dir = 'results/predrnn/'
        self.save_dir = 'checkpoints/predrnn/'
        self.gen_frm_dir = 'results/predrnn/'
        
        # 计划采样配置
        self.scheduled_sampling = True
        self.sampling_stop_iter = 50000
        self.sampling_start_value = 1.0
        self.sampling_changing_rate = 0.00002
        
        # 额外的保留计划采样参数（原代码中使用但未定义）
        self.r_sampling_step_1 = 25000
        self.r_sampling_step_2 = 50000
        self.r_exp_alpha = 5000.0

# 创建默认配置实例
default_config = Config()