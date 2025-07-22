import os
import shutil
import argparse
import numpy as np
import math
from core.data_provider import datasets_factory
from core.models.model_factory import Model
import core.trainer as trainer
from config import default_config  # 导入配置

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# 使用config中的默认值设置参数
# 训练/测试
parser.add_argument('--device', type=str, default=default_config.device)
parser.add_argument('--is_training', type=str, default=default_config.is_training)
parser.add_argument('--data_train_path', type=str, default=default_config.data_train_path)
parser.add_argument('--data_val_path', type=str, default=default_config.data_val_path)
parser.add_argument('--data_test_path', type=str, default=default_config.data_test_path)

# 序列配置
parser.add_argument('--input_length', type=int, default=default_config.input_length)
parser.add_argument('--real_length', type=int, default=default_config.real_length)
parser.add_argument('--total_length', type=int, default=default_config.total_length)

# 图像配置
parser.add_argument('--img_width', type=int, default=default_config.img_width)
parser.add_argument('--img_height', type=int, default=default_config.img_height)
parser.add_argument('--img_channel', type=int, default=default_config.img_channel)
parser.add_argument('--patch_size', type=int, default=default_config.patch_size)

# 损失函数配置
parser.add_argument('--alpha', type=float, default=default_config.alpha)
parser.add_argument('--factor', type=float, default=default_config.factor)

# 模型配置
parser.add_argument('--model_name', type=str, default=default_config.model_name)
parser.add_argument('--dataset', type=str, default=default_config.dataset)
parser.add_argument('--num_workers', type=int, default=default_config.num_workers)
parser.add_argument('--num_hidden', type=int, default=default_config.num_hidden)
parser.add_argument('--num_layers', type=int, default=default_config.num_layers)
parser.add_argument('--num_heads', type=int, default=default_config.num_heads)
parser.add_argument('--filter_size', type=int, default=default_config.filter_size)
parser.add_argument('--stride', type=int, default=default_config.stride)

# 训练配置
parser.add_argument('--lr', type=float, default=default_config.lr)
parser.add_argument('--lr_decay', type=float, default=default_config.lr_decay)
parser.add_argument('--delay_interval', type=float, default=default_config.delay_interval)
parser.add_argument('--batch_size', type=int, default=default_config.batch_size)
parser.add_argument('--max_iterations', type=int, default=default_config.max_iterations)
parser.add_argument('--max_epoches', type=int, default=default_config.max_epoches)

# 显示和保存配置
parser.add_argument('--display_interval', type=int, default=default_config.display_interval)
parser.add_argument('--test_interval', type=int, default=default_config.test_interval)
parser.add_argument('--snapshot_interval', type=int, default=default_config.snapshot_interval)
parser.add_argument('--num_save_samples', type=int, default=default_config.num_save_samples)
parser.add_argument('--n_gpu', type=int, default=default_config.n_gpu)

# 模型路径配置
parser.add_argument('--pretrained_model', type=str, default=default_config.pretrained_model)
parser.add_argument('--perforamnce_dir', type=str, default=default_config.perforamnce_dir)
parser.add_argument('--save_dir', type=str, default=default_config.save_dir)
parser.add_argument('--gen_frm_dir', type=str, default=default_config.gen_frm_dir)

# 计划采样配置
parser.add_argument('--scheduled_sampling', type=bool, default=default_config.scheduled_sampling)
parser.add_argument('--sampling_stop_iter', type=int, default=default_config.sampling_stop_iter)
parser.add_argument('--sampling_start_value', type=float, default=default_config.sampling_start_value)
parser.add_argument('--sampling_changing_rate', type=float, default=default_config.sampling_changing_rate)

# 添加缺失的保留计划采样参数
parser.add_argument('--r_sampling_step_1', type=int, default=default_config.r_sampling_step_1)
parser.add_argument('--r_sampling_step_2', type=int, default=default_config.r_sampling_step_2)
parser.add_argument('--r_exp_alpha', type=float, default=default_config.r_exp_alpha)

args = parser.parse_args()
print(args)

def reserve_schedule_sampling_exp(itr):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0
    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0
    r_random_flip = np.random.random_sample(
        (args.batch_size, args.input_length - 1))
    r_true_token = (r_random_flip < r_eta)
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - 2,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return real_input_flag

def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,                          
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros
    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate  
    else:
        eta = 0.0
    random_flip = np.random.random_sample((args.batch_size,    
                                           args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)  
    ones = np.ones((args.img_width // args.patch_size,  
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,  
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:  # bool值
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)                            
    real_input_flag = np.reshape(real_input_flag,                         
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag

def train_wrapper(model):
    begin = 0
    if args.pretrained_model:  
        model.load(args.pretrained_model)
        begin = int(args.pretrained_model.split('-')[-1])
    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        data_train_path=args.data_train_path,
                                                        dataset=args.dataset,
                                                        data_test_path=args.data_val_path,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True)
    val_input_handle = datasets_factory.data_provider(configs=args,
                                                      data_train_path=args.data_train_path,
                                                      dataset=args.dataset,
                                                      data_test_path=args.data_val_path,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False)
    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itr = begin
    # real_input_flag = {}
    for epoch in range(0, args.max_epoches):
        if itr > args.max_iterations:
            break
        for ims in train_input_handle:  # (4,20,1,64,64)
            if itr > args.max_iterations:
                break
            eta, real_input_flag = schedule_sampling(eta, itr)  # (4,9,64,64,1)
            if itr % args.test_interval == 0:
                print('Validate:')
                trainer.test(model, val_input_handle, args, itr)
            trainer.train(model, ims, real_input_flag, args, itr)
            if itr % args.snapshot_interval == 0 and itr > begin:
                model.save(itr)
            itr += 1

def test_wrapper(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       data_train_path=args.data_train_path,
                                                       dataset=args.dataset,
                                                       data_test_path=args.data_test_path,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False)
    itr = 1
    for i in range(itr):
        trainer.test(model, test_input_handle, args, itr)

if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)
if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)
print('Initializing models')
model = Model(args)
if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)