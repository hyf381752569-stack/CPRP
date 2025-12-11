#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
深度学习模型评估脚本
主要功能：评估中心点检测和方向预测模型的性能
包含可视化、多种评估指标和结果保存功能
"""

import os
import copy
from matplotlib import pyplot as plt

import numpy as np
import scipy
import json
import torch

# 导入项目相关模块
from datasets import get_centerdir_dataset  # 获取中心点方向数据集
from models import get_model, get_center_model  # 获取主模型和中心点模型
from utils.utils import tensor_mask_to_ids, variable_len_collate  # 工具函数
from utils.visualize import get_visualizer  # 可视化工具
from utils.evaluation.center_global_min import CenterGlobalMinimizationEval  # 全局最小化评估
from utils.evaluation.center import CenterEvaluation  # 中心点评估
from utils.evaluation.orientation import OrientationEval  # 方向评估
from utils.evaluation.multivariate import MultivariateEval  # 多变量评估

from inference.processing import CenterDirProcesser  # 推理处理器

class Evaluator:
    """
    模型评估器主类
    负责加载模型、处理数据、执行评估和保存结果
    """
    
    def __init__(self, args):
        """
        初始化评估器
        
        Args:
            args (dict): 配置参数字典，包含模型、数据集、评估等配置
        """
        # 启用matplotlib交互模式，用于实时显示图像
        plt.ion()

        # 如果启用cudnn基准测试，可以加速卷积操作
        if args.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        self.args = args

        # 设置计算设备（GPU或CPU）
        self.device = torch.device("cuda:0" if args['cuda'] else "cpu")

    def initialize(self):
        """
        初始化评估器的各个组件
        包括可视化器、数据集和模型加载
        """
        args = self.args

        ###################################################################################################
        # 可视化器设置

        # 默认可视化类（向后兼容）
        viz_name = 'CentersVisualizeTest'
        viz_opts = dict(
            tensorboard_dir=os.path.join(args['save_dir'], 'tensorboard'),  # tensorboard日志目录
            to_file_only=True,  # 是否仅保存到文件
            autoadjust_figure_size=bool(args.get('autoadjust_figure_size'))  # 是否自动调整图像大小
        )

        # 新版本可视化器配置
        if args.get('visualizer'):
            viz_name = args['visualizer']['name']
            viz_opts.update(args['visualizer']['opts'] if 'opts' in args['visualizer'] else dict())

        self.visualizer = get_visualizer(viz_name, viz_opts)

        ###################################################################################################
        # 构建数据集和处理流水线

        self.dataset_it, self.centerdir_groundtruth_op, self.processed_image_iter = self._construct_dataset_and_processing(args, self.device)

    def _construct_dataset_and_processing(self, args, device):
        """
        构建数据集和处理流水线
        
        Args:
            args (dict): 配置参数
            device (torch.device): 计算设备
            
        Returns:
            tuple: (数据加载器, 真值处理操作, 图像处理迭代器)
        """

        ###################################################################################################
        # 数据加载器配置
        dataset_workers = args['dataset']['workers'] if 'workers' in args['dataset'] else 0  # 数据加载线程数
        dataset_batch = args['dataset']['batch_size'] if 'batch_size' in args['dataset'] else 1  # 批次大小

        # 真值加载配置
        groundtruth_loading = args.get('groundtruth_loading')
        if groundtruth_loading is not None and groundtruth_loading != 'minimal':
            groundtruth_loading = True

        # 获取数据集和真值处理操作
        dataset, centerdir_groundtruth_op = get_centerdir_dataset(
            args['dataset']['name'], 
            args['dataset']['kwargs'],
            args['dataset'].get('centerdir_gt_opts') if groundtruth_loading else None
        )

        # 将真值处理操作转移到GPU（如果有的话）
        if centerdir_groundtruth_op is not None:
            centerdir_groundtruth_op = torch.nn.DataParallel(centerdir_groundtruth_op).to(device)

        # 创建数据加载器
        dataset_it = torch.utils.data.DataLoader(
            dataset, 
            batch_size=dataset_batch, 
            shuffle=False,  # 评估时不打乱数据
            drop_last=False,  # 不丢弃最后不完整的批次
            num_workers=dataset_workers, 
            pin_memory=True if args['cuda'] else False,  # GPU时使用固定内存
            collate_fn=variable_len_collate  # 处理变长数据的整理函数
        )

        ###################################################################################################
        # 加载主模型
        model = get_model(args['model']['name'], args['model']['kwargs'])
        model.init_output(args['num_vector_fields'])  # 初始化输出层
        model = torch.nn.DataParallel(model).to(device)  # 多GPU并行并转移到设备

        # 中心点模型列表
        center_model_list = []

        def get_center_fn():
            """获取中心点模型的工厂函数"""
            return get_center_model(
                args['center_model']['name'], 
                args['center_model']['kwargs'],
                is_learnable=args['center_model'].get('use_learnable_center_estimation')
            )

        # 根据中心点检查点路径的数量准备中心点模型
        if args.get('center_checkpoint_path') and isinstance(args['center_checkpoint_path'], list):
            # 多个中心点模型的情况
            assert 'center_checkpoint_name_list' in args and isinstance(args['center_checkpoint_name_list'], list)
            assert len(args['center_checkpoint_name_list']) == len(args['center_checkpoint_path'])

            for center_checkpoint_name, center_checkpoint_path in zip(args['center_checkpoint_name_list'], args['center_checkpoint_path']):
                center_model = get_center_fn()
                center_model.init_output(args['num_vector_fields'])
                center_model_list.append(dict(
                    name=center_checkpoint_name,
                    checkpoint=center_checkpoint_path,
                    model=center_model
                ))
        else:
            # 单个中心点模型的情况
            center_checkpoint_name = args.get('center_checkpoint_name') if 'center_checkpoint_name' in args else ''
            center_checkpoint_path = args.get('center_checkpoint_path')

            center_model = get_center_fn()
            center_model.init_output(args['num_vector_fields'])
            center_model_list.append(dict(
                name=center_checkpoint_name,
                checkpoint=center_checkpoint_path,
                model=center_model
            ))

        # 将所有中心点模型转移到设备
        for center_model_desc in center_model_list:
            center_model_desc['model'] = torch.nn.DataParallel(center_model_desc['model']).to(device)

        ###################################################################################################
        # 加载模型权重快照
        if os.path.exists(args['checkpoint_path']):
            print('从 "%s" 加载模型权重' % args['checkpoint_path'])
            state = torch.load(args['checkpoint_path'])
            
            # 加载主模型权重
            if 'model_state_dict' in state: 
                model.load_state_dict(state['model_state_dict'], strict=True)
            
            # 如果没有单独的中心点检查点，从主检查点加载中心点模型权重
            if not args.get('center_checkpoint_path') and 'center_model_state_dict' in state and args['center_model'].get('use_learnable_center_estimation'):
                for center_model_desc in center_model_list:
                    center_model_desc['model'].load_state_dict(state['center_model_state_dict'], strict=False)
        else:
            raise Exception('检查点路径 {} 不存在!'.format(args['checkpoint_path']))

        # 加载可学习的中心点估计模型权重
        if args['center_model'].get('use_learnable_center_estimation'):
            for center_model_desc in center_model_list:
                if center_model_desc['checkpoint'] is None:
                    continue
                    
                if os.path.exists(center_model_desc['checkpoint']):
                    print('从 "%s" 加载中心点模型权重' % center_model_desc['checkpoint'])
                    state = torch.load(center_model_desc['checkpoint'])
                    
                    if 'center_model_state_dict' in state:
                        # 处理输入权重形状不匹配的问题
                        if 'module.instance_center_estimator.conv_start.0.weight' in state['center_model_state_dict']:
                            checkpoint_input_weights = state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight']
                            center_input_weights = center_model_desc['model'].module.instance_center_estimator.conv_start[0].weight
                            
                            if checkpoint_input_weights.shape != center_input_weights.shape:
                                # 只加载前两个通道的权重
                                state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'] = checkpoint_input_weights[:,:2,:,:]

                                print('警告: #####################################################################################################')
                                print('警告: 中心点输入形状不匹配 - 将只加载前两个通道的权重，这是正确的吗？!!!')
                                print('警告: #####################################################################################################')

                        center_model_desc['model'].load_state_dict(state['center_model_state_dict'], strict=False)
                else:
                    raise Exception('检查点路径 {} 不存在!'.format(center_model_desc['checkpoint']))

        ###################################################################################################
        # 主要处理流水线：创建图像处理器
        processed_image_iter = CenterDirProcesser(model, center_model_list, device)

        return dataset_it, centerdir_groundtruth_op, processed_image_iter

    def compile_evaluation_list(self):
        """
        编译评估列表
        根据配置参数生成所有需要执行的评估组合
        
        Returns:
            dict: 每个中心点模型对应的评估列表字典
        """
        args = self.args

        # 获取中心点检查点名称列表
        center_checkpoint_name = args.get('center_checkpoint_name_list')
        if center_checkpoint_name is None:
            center_checkpoint_name = [args.get('center_checkpoint_name')]
        if center_checkpoint_name[0] is None:
            center_checkpoint_name = ['']

        evaluation_lists_per_center_model = {}

        for center_model_name in center_checkpoint_name:

            #########################################################################################################
            ## 准备评估配置/参数

            if type(args.get('eval')) == dict:
                args_eval = args['eval']
                import itertools, functools

                ##########################################################################################
                # 基于提供的评分列表创建评分函数

                # 索引应该匹配预测中返回的分数（在x,y位置之后）
                scoring_index = {'mask': 0, 'center': 1, 'orientation_confidence': 2}

                # 创建新字典的函数，如果值是列表则组合每个值
                def combitorial_args_fn(X):
                    """组合参数函数：生成参数的所有可能组合"""
                    # 转换为值列表
                    X_vals = [[val] if type(val) not in [list, tuple] else val for val in X.values()]
                    # 对值列表应用笛卡尔积
                    for vals in itertools.product(*X_vals):
                        yield dict(zip(X.keys(), vals))

                # 通过乘以特定分数创建最终分数的函数（即基于score_types名称乘以列）
                def _scoring_fn(scores, score_types):
                    """评分函数：根据指定的评分类型计算最终分数"""
                    selected_scores = [scores[:, scoring_index[t]] for t in score_types]
                    return np.multiply.reduce(selected_scores) if len(selected_scores) > 1 else selected_scores[0]

                # 基于选定列和提供阈值对预测进行阈值处理的函数
                def _scoring_thrs_fn(scores, score_thr_dict):
                    """评分阈值函数：根据阈值字典过滤预测"""
                    selected_scores = [scores[:, scoring_index[t]] > thr for t, thr in score_thr_dict.items() if
                                       thr is not None]
                    return np.multiply.reduce(selected_scores) if len(selected_scores) > 1 else selected_scores[0]

                ##########################################################################################
                # 基于阈值和评估参数的组合创建所有评估类
                evaluation_lists = []

                score_combination_and_thr_list = args_eval.get('score_combination_and_thr')

                if type(score_combination_and_thr_list) not in [list, tuple]:
                    score_combination_and_thr_list = [score_combination_and_thr_list]

                # 在最终分数中忽略的评分类型
                ignore_in_final_score = args_eval.get('ignore_in_final_score')
                if not ignore_in_final_score:
                    ignore_in_final_score = []

                # 遍历不同的评分类型
                for score_combination_and_thr in score_combination_and_thr_list:
                    # 基于请求的分数创建评分函数
                    scoring_fn = functools.partial(_scoring_fn, score_types=set(score_combination_and_thr.keys())-set(ignore_in_final_score))
                    
                    # 遍历所使用分数的不同阈值组合
                    for scoring_thrs in combitorial_args_fn(score_combination_and_thr):
                        # 基于每个分数的请求阈值创建阈值函数
                        scoring_thrs_fn = functools.partial(_scoring_thrs_fn, score_thr_dict=scoring_thrs)
                        
                        # 遍历所有请求的final_score_thr
                        for final_score_thr in args_eval.get('score_thr_final', -np.inf):
                            # 生成实验名称和属性
                            scoring_str = "+".join(scoring_thrs.keys())
                            scoring_thr = ["%s=%.2f" % (t, thr) for t, thr in scoring_thrs.items() if thr is not None]
                            
                            if args_eval.get('top_k_predictions'):
                                exp_name = "%s-final_score_thr=%.2f-%s-top_k_predictions=%d" % (scoring_str,final_score_thr,"-".join(scoring_thr),int(args_eval['top_k_predictions']))
                            else:
                                exp_name = "%s-final_score_thr=%.2f-%s" % (scoring_str,final_score_thr,"-".join(scoring_thr))
                            
                            exp_attributes = dict(scoring=scoring_str,scoring_thr=scoring_thr, final_score_thr=final_score_thr)

                            # 中心点评估器列表
                            center_eval = []
                            if args_eval.get('centers_global_minimization') is not None:
                                ##########################################################################################
                                ## 基于最佳拟合检测最小化的计数指标评估类
                                center_eval += [CenterGlobalMinimizationEval(exp_name=exp_name, exp_attributes=exp_attributes, **args)
                                                    for args in combitorial_args_fn(args_eval['centers_global_minimization'])]

                            if not args_eval.get('skip_center_eval'):
                                ##########################################################################################
                                ## 仅使用AP的中心点评估类
                                center_eval += [CenterEvaluation()]

                            # 方向评估器
                            if args_eval.get('orientation'):
                                ##########################################################################################
                                ## 方向评估类
                                orientation_eval = [OrientationEval(exp_name=exp_name, exp_attributes=exp_attributes, **args)
                                                        for args in combitorial_args_fn(args_eval['orientation'])]
                            else:
                                orientation_eval = []

                            # 添加多变量评估的函数
                            add_multivariate_eval_fn = lambda eval_obj: MultivariateEval(eval_obj, **args['eval']['enable_multivariate_eval']) \
                                                                        if args['eval'].get('enable_multivariate_eval') else eval_obj

                            # 添加到评估列表
                            evaluation_lists.append(dict(
                                scoring_fn=scoring_fn,
                                scoring_thrs_fn=scoring_thrs_fn,
                                final_score_thr=final_score_thr,
                                center_eval=[add_multivariate_eval_fn(c) for c in center_eval],
                                orientation_eval=[add_multivariate_eval_fn(o) for o in orientation_eval],
                                top_k_predictions=args_eval.get('top_k_predictions')
                            ))

            # 检查评估是否已存在，如果需要则返回空列表
            if args.get('skip_if_exists') and args['save_dir'] is not None:
                if self._check_if_results_exist(evaluation_lists, self._get_save_dir(center_model_name)):
                    evaluation_lists = []

            evaluation_lists_per_center_model[center_model_name] = evaluation_lists

        return evaluation_lists_per_center_model

    def _get_checkpoint_timestamp(self):
        """
        获取检查点的时间戳
        
        Returns:
            float: 检查点文件的修改时间
        """
        get_date_fn = lambda p: os.path.getmtime(p) if os.path.exists(p) else 0
        args = self.args

        mod_date = get_date_fn(args['checkpoint_path'])

        return mod_date
    
    def _check_if_results_exist(self, evaluation_lists, save_dir):
        """
        检查结果是否已存在
        
        Args:
            evaluation_lists (list): 评估列表
            save_dir (str): 保存目录
            
        Returns:
            bool: 如果结果已存在且比检查点新则返回True
        """
        # 如果结果比检查点修改时间早，则认为结果无效
        checkpoint_time = self._get_checkpoint_timestamp()

        # 检查点不存在则返回false
        if checkpoint_time == 0:
            return False

        c_eval_exists, o_eval_exists, i_eval_exists = [], [], []
        for eval_args in evaluation_lists:
            # 检查中心点和方向评估结果是否存在且比检查点新
            C = [c_eval.get_results_timestamp(save_dir) >= checkpoint_time for c_eval in eval_args['center_eval']]
            O = [o_eval.get_results_timestamp(save_dir) >= checkpoint_time for o_eval in eval_args['orientation_eval']]

            c_eval_exists.append(all(C))
            o_eval_exists.append(all(O))

        return all(c_eval_exists) and all(o_eval_exists) and all(i_eval_exists)
    
    def _get_save_dir(self, center_model_name):
        """
        获取保存目录路径
        
        Args:
            center_model_name (str): 中心点模型名称
            
        Returns:
            str: 保存目录路径
        """
        MARKER = self.args['center_checkpoint_name'] if 'center_checkpoint_name' in self.args else '##CENTER_MODEL_NAME##'

        if MARKER in self.args['save_dir']:
            return self.args['save_dir'].replace(MARKER, center_model_name)
        else:
            return self.args['save_dir']

    def _visualize_prediction(self, sample, result, predictions, predictions_score, eval_obj, eval_res,
                              difficult, impath, save_root):
        """
        可视化预测结果
        
        Args:
            sample: 输入样本
            result: 模型输出结果
            predictions: 预测结果
            predictions_score: 预测分数
            eval_obj: 评估对象
            eval_res: 评估结果
            difficult: 困难样本标志
            impath: 图像路径
            save_root: 保存根目录
        """
        args = self.args
        
        # 处理多变量评估对象
        if type(eval_obj) == MultivariateEval:
            eval_obj = eval_obj.eval_obj

        if eval_res is None:
            # 如果评估器没有输出则跳过可视化
            return
        elif type(eval_obj) in [CenterEvaluation, CenterGlobalMinimizationEval, OrientationEval]:
            # 解析结果
            gt_missed, pred_missed, pred_gt_match, filename_suffix, pred_gt_match_idx = eval_res

            # 这些评估器没有掩码输出，所以使用空字典
            pred_polygon = {}
        else:
            raise Exception("不支持的评估器类型用于可视化")

        # 根据显示条件决定是否可视化
        if args['display'] is True or \
                type(args['display']) is str and args['display'].lower() == 'all' or \
                type(args['display']) is str and args['display'].lower() == 'error_gt' and gt_missed or \
                type(args['display']) is str and args['display'].lower() == 'error' and (gt_missed or pred_missed):

            # 创建可视化保存文件夹
            visualize_to_folder = os.path.join(save_root, eval_obj.exp_name, eval_obj.save_str())
            os.makedirs(visualize_to_folder, exist_ok=True)

            if len(predictions) > 0:
                # 用于绘图的预测结果 (x,y)
                plot_predictions = np.array(predictions)[:, :2]

                # 匹配的真值信息
                plot_predictions_gt_match = np.concatenate((pred_gt_match[:, :1],
                                                            pred_gt_match_idx[:, :1]), axis=1)
            else:
                plot_predictions = []
                plot_predictions_gt_match = []

            # 获取基础文件名
            base = self.visualizer.impath2name_fn(impath) if impath is not None else None

            # 执行可视化
            self.visualizer(sample, result, plot_predictions, predictions_score, pred_polygon.values(),
                            plot_predictions_gt_match, difficult, filename_suffix+base, visualize_to_folder)

    #########################################################################################################
    ## 主要运行函数
    def run(self, evaluation_lists_per_center_model):
        """
        运行评估流程
        
        Args:
            evaluation_lists_per_center_model (dict): 每个中心点模型的评估列表
        """
        args = self.args

        with torch.no_grad():  # 评估时不需要计算梯度

            #########################################################################################################
            ## 处理每张图像并进行评估
            for im_index, (sample, result) in enumerate(self.processed_image_iter(self.dataset_it, self.centerdir_groundtruth_op)):

                # 从样本中提取信息
                im = sample.get('image')  # 图像数据
                im_shape = sample.get('im_shape')  # 图像形状
                im_name = sample['im_name']  # 图像名称

                instances = sample.get('instance')  # 实例掩码
                instances_ids = sample.get('instance_ids')  # 实例ID
                centerdir_gt = sample.get('centerdir_groundtruth')  # 中心点方向真值
                ignore_flags = sample.get('ignore')  # 忽略标志
                gt_centers_dict = sample.get('center_dict')  # 真值中心点字典

                # 从结果中提取预测信息
                predictions_ = result['predictions']  # 预测结果
                pred_angle_ = result.get('pred_angle')  # 预测角度
                center_model_name = result['center_model_name']  # 中心点模型名称

                # 如果没有图像形状信息，从图像中获取
                if im_shape is None:
                    assert im is not None, '错误: 没有"im"或"im_shape"无法进行评估'
                    im_shape = im.shape[-2:]

                # 基于忽略标志获取困难掩码（值为8表示困难标志）
                difficult = (ignore_flags & 8 > 0).squeeze() if ignore_flags is not None else torch.sparse_coo_tensor(size=im_shape)

                # 如果还没有将掩码转换为实例1D索引，则进行转换以加快重叠计算
                if instances_ids is None:
                    assert instances is not None, '错误: 没有"instances"或"instances_ids"无法进行评估'
                    instances_ids = tensor_mask_to_ids(instances)

                # 提取所有分数（除了前两列的x,y坐标）
                all_scores = predictions_[:, 2:] if len(predictions_) > 0 else []

                assert center_model_name in evaluation_lists_per_center_model

                save_vis_root = self._get_save_dir(center_model_name)
                
                # 对不同的评分、阈值和其他评估参数组合进行评估
                for eval_args in evaluation_lists_per_center_model[center_model_name]:
                    scoring_fn = eval_args['scoring_fn']  # 评分函数
                    scoring_thrs_fn = eval_args['scoring_thrs_fn']  # 评分阈值函数
                    final_score_thr = eval_args['final_score_thr']  # 最终分数阈值
                    center_eval = eval_args['center_eval']  # 中心点评估器列表
                    orientation_eval = eval_args['orientation_eval']  # 方向评估器列表

                    if len(all_scores) > 0:
                        # 1. 应用评分函数
                        predictions_score = scoring_fn(all_scores)

                        # 2. 基于特定评分阈值进行过滤
                        selected_pred_idx = np.where((predictions_score > final_score_thr) *
                                                     scoring_thrs_fn(all_scores) *
                                                     (predictions_.sum(axis=1) != 0))[0]
                        predictions = predictions_[selected_pred_idx,:]
                        predictions_score = predictions_score[selected_pred_idx]

                        # 如果有角度预测，也进行相应过滤
                        if pred_angle_ is not None:
                            pred_angle = pred_angle_[selected_pred_idx,:]

                        # 3. 如果指定了top-k预测，则只保留前k个
                        if eval_args.get('top_k_predictions'):
                            top_k = eval_args['top_k_predictions']
                            k = min(top_k, len(predictions))
                            
                            predictions = predictions[:k]
                            predictions_score = predictions_score[:k]
                            
                            if pred_angle_ is not None:
                                pred_angle = pred_angle[:k]

                    else:
                        # 没有预测结果
                        predictions = []
                        predictions_score = []
                        pred_angle = []

                    # 4. 对center_eval和orientation_eval中的每个评估器进行评估和可视化

                    # 中心点评估的收集指标
                    for c_eval in center_eval:
                        center_eval_res = c_eval.add_image_prediction(
                            im_name, im_index, im_shape,
                            predictions, predictions_score,
                            instances_ids, gt_centers_dict, difficult, centerdir_gt,
                            return_matched_gt_idx=True
                        )

                        # 可视化预测结果
                        self._visualize_prediction(sample, result, predictions, predictions_score,
                                                   c_eval, center_eval_res, difficult, im_name, save_vis_root)

                    # 方向评估的收集指标
                    for o_eval in orientation_eval:
                        orient_eval_res = o_eval.add_image_prediction(
                            im_name, im_index, im_shape,
                            predictions, predictions_score, pred_angle,
                            instances_ids, gt_centers_dict, difficult, centerdir_gt,
                            return_matched_gt_idx=True
                        )

                        # 可视化预测结果
                        self._visualize_prediction(sample, result, predictions, predictions_score,
                                                   o_eval, orient_eval_res, difficult, im_name, save_vis_root)

            ########################################################################################################
            # 最后，将结果输出到显示和文件

            if 'eval' not in args or args['eval']:
                for center_model_name, evaluation_lists in evaluation_lists_per_center_model.items():
                    save_dir = self._get_save_dir(center_model_name)

                    for eval_args in evaluation_lists:
                        center_eval = eval_args['center_eval']
                        orientation_eval = eval_args['orientation_eval']

                        ########################################################################################################
                        ## 基于中心点和方向的评估
                        for c_eval in center_eval + orientation_eval:
                            # 计算并显示最终指标
                            metrics = c_eval.calc_and_display_final_metrics(self.dataset_it, save_dir=save_dir)

                            # 存储参数以供参考
                            if save_dir is not None and args.get('save_eval_args'):
                                self.save_args(os.path.join(save_dir, c_eval.exp_name, c_eval.save_str()), c_eval.get_attributes())

    def save_args(self, save_dir, extra_args=None):
        """
        保存评估参数到JSON文件
        
        Args:
            save_dir (str): 保存目录
            extra_args (dict, optional): 额外的参数
        """
        if extra_args is not None:
            args = copy.deepcopy(self.args)
            args.update(extra_args)
        else:
            args = self.args

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 将参数保存为JSON文件
        with open(os.path.join(save_dir, 'eval_params.json'), 'w') as file:
            file.write(json.dumps(args, indent=4, sort_keys=True, default=lambda o: '<not serializable>'))


def main():
    """
    主函数：执行模型评估流程
    """
    from config import get_config_args

    # 从环境变量或配置文件获取参数
    args = get_config_args(dataset=os.environ.get('DATASET'), type='test')

    # 创建评估器实例
    eval = Evaluator(args)

    # 获取所有需要执行的评估列表（基于不同阈值组合等）
    evaluation_lists = eval.compile_evaluation_list()

    # 继续初始化和运行所有评估，除非evaluation_lists为空
    if any([len(e) > 0 for e in evaluation_lists.values()]):
        # 在检查有效评估列表后进行初始化
        eval.initialize()
        # 最后运行所有评估
        eval.run(evaluation_lists)
    else:
        print('由于输出已存在，跳过评估')


if __name__ == "__main__":
    main()