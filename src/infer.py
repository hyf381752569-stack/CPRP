#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
深度学习模型推理脚本
用于运行模型推理并测量性能指标
"""

import os, time

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy

import torch

from datasets import get_centerdir_dataset
from models import get_model, get_center_model
from utils.utils import variable_len_collate

class Inferencce:
    """
    推理类，负责模型的加载、初始化和推理过程
    """
    def __init__(self, args):
        """
        初始化推理类
        
        Args:
            args: 配置参数字典，包含模型、数据集等配置信息
        """
        # 设置matplotlib显示模式
        # if args['display'] and not args.get('display_to_file_only'):
        if True:
            # 使用交互式后端显示图形
            # plt.switch_backend('TkAgg')
            plt.ion()  # 开启交互模式
        else:
            # 非交互模式，用于后台运行
            plt.ioff()  # 关闭交互模式
            plt.switch_backend("agg")  # 使用非GUI后端

        # 启用cudnn基准模式以优化卷积性能（如果指定）
        if args.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        self.args = args

        # 设置计算设备（GPU或CPU）
        self.device = torch.device("cuda:0" if args['cuda'] else "cpu")

    def initialize(self):
        """
        初始化数据集和模型
        """
        args = self.args

        ###################################################################################################
        # 构建数据集和模型
        self.dataset_it, self.model, self.center_model = self._construct_dataset_and_processing(args, self.device)

    def _construct_dataset_and_processing(self, args, device):
        """
        构建数据集和处理模型
        
        Args:
            args: 配置参数
            device: 计算设备
            
        Returns:
            dataset_it: 数据加载器
            model: 主模型
            center_model: 中心点检测模型
        """

        ###################################################################################################
        # 数据加载器配置
        # 获取数据集工作进程数，默认为0（单进程）
        dataset_workers = args['dataset']['workers'] if 'workers' in args['dataset'] else 0
        # 获取批处理大小，默认为1
        dataset_batch = args['dataset']['batch_size'] if 'batch_size' in args['dataset'] else 1

        # 导入自定义数据变换模块
        from utils import transforms as my_transforms
        # 注释掉的代码：数据预处理变换配置
        # args['dataset']['kwargs']['transform'] = my_transforms.get_transform([
        #     { 'name': 'Padding', 'opts': { 'keys': ('image',), 'pad_to_size_factor': 32 } },  # 填充到32的倍数
        #     { 'name': 'ToTensor', 'opts': { 'keys': ('image',), 'type': (torch.FloatTensor) } },  # 转换为张量
        #     { 'name': 'ToTensor', 'opts': { 'keys': ('image',), 'type': (torch.FloatTensor) } }   # 重复的转换
        # ])

        # 获取数据集（无真实标签，仅用于推理）
        dataset, _ = get_centerdir_dataset(args['dataset']['name'], args['dataset']['kwargs'], no_groundtruth=True)

        # 创建数据加载器
        dataset_it = torch.utils.data.DataLoader(
            dataset, 
            batch_size=dataset_batch,          # 批处理大小
            shuffle=False,                     # 不打乱数据顺序
            drop_last=False,                   # 不丢弃最后一个不完整的批次
            num_workers=dataset_workers,       # 工作进程数
            pin_memory=True if args['cuda'] else False,  # 如果使用GPU则启用内存锁定
            collate_fn=variable_len_collate    # 处理变长数据的整理函数
        )

        ###################################################################################################
        # 加载主模型
        # 根据配置创建主模型
        model = get_model(args['model']['name'], args['model']['kwargs'])
        # 初始化模型输出层（根据向量场数量）
        model.init_output(args['num_vector_fields'])
        # 使用数据并行并移动到指定设备
        model = torch.nn.DataParallel(model).to(device)

        # 准备中心点检测模型
        # 获取中心点检测模型的检查点配置
        center_checkpoint_name = args.get('center_checkpoint_name') if 'center_checkpoint_name' in args else ''
        center_checkpoint_path = args.get('center_checkpoint_path')

        # 创建中心点检测模型
        center_model = get_center_model(
            args['center_model']['name'], 
            args['center_model']['kwargs'],
            is_learnable=args['center_model'].get('use_learnable_center_estimation'),  # 是否使用可学习的中心估计
            use_fast_estimator=True  # 使用快速估计器
        )

        # 初始化中心模型输出
        center_model.init_output(args['num_vector_fields'])
        # 使用数据并行并移动到指定设备
        center_model = torch.nn.DataParallel(center_model).to(device)

        ###################################################################################################
        # 加载模型检查点
        if os.path.exists(args['checkpoint_path']):
            print('从 "%s" 加载主模型检查点' % args['checkpoint_path'])
            state = torch.load(args['checkpoint_path'])
            # 加载主模型权重
            if 'model_state_dict' in state: 
                model.load_state_dict(state['model_state_dict'], strict=True)
            # 如果没有单独的中心模型检查点，且使用可学习的中心估计，则从主检查点加载中心模型权重
            if not args.get('center_checkpoint_path') and 'center_model_state_dict' in state and args['center_model'].get('use_learnable_center_estimation'):
                center_model.load_state_dict(state['center_model_state_dict'], strict=False)
        else:
            raise Exception('检查点路径 {} 不存在!'.format(args['checkpoint_path']))

        # 加载单独的中心模型检查点（如果指定）
        if args['center_model'].get('use_learnable_center_estimation') and len(center_checkpoint_name) > 0:
            if os.path.exists(center_checkpoint_path):
                print('从 "%s" 加载中心模型检查点' % center_checkpoint_path)
                state = torch.load(center_checkpoint_path)
                if 'center_model_state_dict' in state:
                    # 处理输入通道数不匹配的情况
                    if 'module.instance_center_estimator.conv_start.0.weight' in state['center_model_state_dict']:
                        # 获取检查点中的输入权重
                        checkpoint_input_weights = state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight']
                        # 获取当前模型的输入权重
                        center_input_weights = center_model.module.instance_center_estimator.conv_start[0].weight
                        # 如果形状不匹配，只加载前两个通道的权重
                        if checkpoint_input_weights.shape != center_input_weights.shape:
                            state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'] = checkpoint_input_weights[:,:2,:,:]

                            print('警告: #####################################################################################################')
                            print('警告: 中心模型输入形状不匹配 - 将只加载前两个通道的权重，这样做正确吗？！！！')
                            print('警告: #####################################################################################################')

                    # 加载中心模型权重
                    center_model.load_state_dict(state['center_model_state_dict'], strict=False)
            else:
                raise Exception('中心模型检查点路径 {} 不存在!'.format(center_checkpoint_path))

        return dataset_it, model, center_model

    #########################################################################################################
    ## 主要运行函数
    def run(self):
        """
        执行推理过程并测量性能
        """
        args = self.args

        # 用于记录各阶段耗时的字典
        time_array = dict(model=[], center=[], post=[], total=[])
        
        # 禁用梯度计算以节省内存和加速推理
        with torch.no_grad():
            model = self.model            
            center_model = self.center_model
            dataset_it = self.dataset_it

            # 确保批处理大小为1
            assert dataset_it.batch_size == 1

            # 设置模型为评估模式
            model.eval()
            center_model.eval()

            im_image = 0  # 图像计数器
            
            # 处理最多1000张图像
            while im_image < 1000:

                # 遍历数据集
                for sample in self.dataset_it:
                    im_image += 1

                    # 同步CUDA以准确测量时间
                    torch.cuda.synchronize()
                    start_model = time.time()
                    
                    # 运行主模型推理
                    output_batch_ = model(sample['image'])

                    # 同步CUDA并记录中心检测开始时间
                    torch.cuda.synchronize()
                    start_center = time.time()

                    # 运行中心点检测模型
                    center_pred, times = center_model(output_batch_)

                    # 获取预测结果（注释掉）
                    #predictions = center_pred[0]

                    # 确保数据传输完成
                    torch.cuda.synchronize()
                    end = time.time()

                    # 计算各阶段耗时
                    time_model = start_center - start_model  # 主模型耗时
                    time_center_total = end - start_center   # 中心检测总耗时
                    time_center_preprocess = times[0]        # 中心检测预处理耗时
                    time_center_only = times[1]              # 中心检测核心计算耗时
                    time_center_postprocess = times[2]       # 中心检测后处理耗时
                    time_total = end - start_model           # 总耗时

                    # 记录耗时数据
                    time_array['model'].append(time_model)
                    time_array['center'].append(time_center_only + time_center_preprocess)
                    time_array['post'].append(time_center_postprocess)
                    time_array['total'].append(time_total)

                    # 打印当前图像的详细耗时信息
                    print('总耗时: %.1f ms，其中主模型=%.1f ms，中心检测=%.1f ms (预处理=%.1f ms, 核心=%.1f ms, 后处理=%.1f ms)' %
                          (time_total*1000, time_model*1000, time_center_total*1000,
                           time_center_preprocess*1000, time_center_only*1000, time_center_postprocess*1000,))

                    # 如果处理超过1000张图像则跳出循环
                    if im_image > 1000:
                        break

        # 计算性能统计（每100张图像取样一次）
        times_model = np.array(time_array['model'])[100::100]
        times_center = np.array(time_array['center'])[100::100]
        times_post = np.array(time_array['post'])[100::100]
        times_total = np.array(time_array['total'])[100::100]
        
        # 打印最终性能统计
        print('-------------------------------------------------------------')
        print('性能统计:')
        print('主模型: 平均 %.1f ms, 标准差 %.1f ms' % (times_model.mean()*1000, times_model.std()*1000))
        print('中心检测: 平均 %.1f ms, 标准差 %.1f ms' % (times_center.mean() * 1000, times_center.std() * 1000))
        print('后处理: 平均 %.1f ms, 标准差 %.1f ms' % (times_post.mean() * 1000, times_post.std() * 1000))
        print('总计: 平均 %.1f ms, 标准差 %.1f ms' % (times_total.mean() * 1000, times_total.std() * 1000))

def main():
    """
    主函数：加载配置并运行推理
    """
    from config import get_config_args

    # 从环境变量和配置文件获取参数
    args = get_config_args(dataset=os.environ.get('DATASET'), type='test')

    # 创建推理实例并运行
    infer = Inferencce(args)
    infer.initialize()
    infer.run()

if __name__ == "__main__":
    main()