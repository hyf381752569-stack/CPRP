#!/usr/bin/python
import os
import copy
from matplotlib import pyplot as plt

import numpy as np
import scipy
import json
import torch

from datasets import get_centerdir_dataset
from models import get_model, get_center_model
from utils.utils import tensor_mask_to_ids, variable_len_collate
from utils.visualize import get_visualizer
from utils.evaluation.center_global_min import CenterGlobalMinimizationEval
from utils.evaluation.center import CenterEvaluation
from utils.evaluation.orientation import OrientationEval
from utils.evaluation.multivariate import MultivariateEval

from inference.processing import CenterDirProcesser

class Evaluator:
    def __init__(self, args):
        plt.ion()

        if args.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        self.args = args

        # set device
        self.device = torch.device("cuda:0" if args['cuda'] else "cpu")

    def initialize(self):
        args = self.args

        ###################################################################################################
        # Visualizer

        # default class (legacy support)
        viz_name = 'CentersVisualizeTest'
        viz_opts = dict(tensorboard_dir=os.path.join(args['save_dir'], 'tensorboard'),
                        to_file_only=args.get('display_to_file_only'),
                        autoadjust_figure_size=bool(args.get('autoadjust_figure_size')))

        # newer version
        if args.get('visualizer'):
            viz_name = args['visualizer']['name']
            viz_opts.update(args['visualizer']['opts'] if 'opts' in args['visualizer'] else dict())

        self.visualizer = get_visualizer(viz_name, viz_opts)

        ###################################################################################################
        # set dataset

        self.dataset_it, self.centerdir_groundtruth_op, self.processed_image_iter = self._construct_dataset_and_processing(args, self.device)


    def _construct_dataset_and_processing(self, args, device):

        ###################################################################################################
        # dataloader
        dataset_workers = args['dataset']['workers'] if 'workers' in args['dataset'] else 0
        dataset_batch = args['dataset']['batch_size'] if 'batch_size' in args['dataset'] else 1


        groundtruth_loading = args.get('groundtruth_loading')
        if groundtruth_loading is not None and groundtruth_loading != 'minimal':
            groundtruth_loading = True

        dataset, centerdir_groundtruth_op = get_centerdir_dataset(args['dataset']['name'], args['dataset']['kwargs'],
                                                                  args['dataset'].get('centerdir_gt_opts') if groundtruth_loading else None)

        if centerdir_groundtruth_op is not None:
            centerdir_groundtruth_op = torch.nn.DataParallel(centerdir_groundtruth_op).to(device)

        dataset_it = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch, shuffle=False, drop_last=False,
                                                 num_workers=dataset_workers, pin_memory=True if args['cuda'] else False,
                                                 collate_fn=variable_len_collate)

        ###################################################################################################
        # load model
        model = get_model(args['model']['name'], args['model']['kwargs'])
        model.init_output(args['num_vector_fields'])
        model = torch.nn.DataParallel(model).to(device)

        center_model_list = []

        def get_center_fn():
            return get_center_model(args['center_model']['name'], args['center_model']['kwargs'],
                                    is_learnable=args['center_model'].get('use_learnable_center_estimation'))

        # prepare center_model and center_estimator based on number of center_checkpoint_path that will need to be processed
        if args.get('center_checkpoint_path') and isinstance(args['center_checkpoint_path'],list):
            assert 'center_checkpoint_name_list' in args and isinstance(args['center_checkpoint_name_list'],list)
            assert len(args['center_checkpoint_name_list']) == len(args['center_checkpoint_path'])

            for center_checkpoint_name, center_checkpoint_path in zip(args['center_checkpoint_name_list'], args['center_checkpoint_path']):
                center_model = get_center_fn()

                center_model.init_output(args['num_vector_fields'])
                center_model_list.append(dict(name=center_checkpoint_name,
                                              checkpoint=center_checkpoint_path,
                                              model=center_model))
        else:
            center_checkpoint_name = args.get('center_checkpoint_name') if 'center_checkpoint_name' in args else ''
            center_checkpoint_path = args.get('center_checkpoint_path')

            center_model = get_center_fn()

            center_model.init_output(args['num_vector_fields'])
            center_model_list.append(dict(name=center_checkpoint_name,
                                          checkpoint=center_checkpoint_path,
                                          model=center_model))

        for center_model_desc in center_model_list:
            center_model_desc['model'] = torch.nn.DataParallel(center_model_desc['model']).to(device)

        ###################################################################################################
        # load snapshot
        if os.path.exists(args['checkpoint_path']):
            print('Loading from "%s"' % args['checkpoint_path'])
            state = torch.load(args['checkpoint_path'])
            if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'], strict=True)
            if not args.get('center_checkpoint_path') and 'center_model_state_dict' in state and args['center_model'].get('use_learnable_center_estimation'):
                for center_model_desc in center_model_list:
                    center_model_desc['model'].load_state_dict(state['center_model_state_dict'], strict=False)
        else:
            raise Exception('checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

        if args['center_model'].get('use_learnable_center_estimation'):
            for center_model_desc in center_model_list:
                if center_model_desc['checkpoint'] is None:
                    continue
                if os.path.exists(center_model_desc['checkpoint']):
                    print('Loading center model from "%s"' % center_model_desc['checkpoint'])
                    state = torch.load(center_model_desc['checkpoint'])
                    if 'center_model_state_dict' in state:
                        if 'module.instance_center_estimator.conv_start.0.weight' in state['center_model_state_dict']:
                            checkpoint_input_weights = state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight']
                            center_input_weights = center_model_desc['model'].module.instance_center_estimator.conv_start[0].weight
                            if checkpoint_input_weights.shape != center_input_weights.shape:
                                state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'] = checkpoint_input_weights[:,:2,:,:]

                                print('WARNING: #####################################################################################################')
                                print('WARNING: center input shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                                print('WARNING: #####################################################################################################')

                        center_model_desc['model'].load_state_dict(state['center_model_state_dict'], strict=False)
                else:
                    raise Exception('checkpoint_path {} does not exist!'.format(center_model_desc['checkpoint']))

        ###################################################################################################
        # MAIN PROCESSING PIPELINE:
        processed_image_iter = CenterDirProcesser(model, center_model_list, device)

        return dataset_it, centerdir_groundtruth_op, processed_image_iter

    def compile_evaluation_list(self):
        args = self.args

        # --- FINAL FIX START ---
        # 恢复原始的、更健壮的逻辑来处理该变量，确保它永远不会是None
        center_checkpoint_name = args.get('center_checkpoint_name_list')
        if center_checkpoint_name is None:
            center_checkpoint_name = [args.get('center_checkpoint_name')]
        if center_checkpoint_name[0] is None:
            center_checkpoint_name = ['']
        # --- FINAL FIX END ---

        evaluation_lists_per_center_model = {}

        for center_model_name in center_checkpoint_name:

            if type(args.get('eval')) == dict:
                args_eval = args['eval']
                import itertools, functools

                scoring_index = {'mask': 0, 'center': 1, 'orientation_confidence': 2 }

                def combitorial_args_fn(X):
                    X_vals = [[val] if type(val) not in [list, tuple] else val for val in X.values()]
                    for vals in itertools.product(*X_vals): yield dict(zip(X.keys(), vals))

                def _scoring_fn(scores, score_types):
                    selected_scores = [scores[:, scoring_index[t]] for t in score_types]
                    return np.multiply.reduce(selected_scores) if len(selected_scores) > 1 else (selected_scores[0] if len(selected_scores) > 0 else np.ones(len(scores)))

                def _scoring_thrs_fn(scores, score_thr_dict):
                    selected_scores = [scores[:, scoring_index[t]] > thr for t, thr in score_thr_dict.items() if thr is not None]
                    return np.multiply.reduce(selected_scores) if len(selected_scores) > 1 else (selected_scores[0] if len(selected_scores) > 0 else np.ones(len(scores), dtype=bool))
                
                evaluation_lists = []

                score_combination_and_thr_list = args_eval.get('score_combination_and_thr', [{}])

                if type(score_combination_and_thr_list) not in [list, tuple]:
                    score_combination_and_thr_list = [score_combination_and_thr_list]

                ignore_in_final_score = args_eval.get('ignore_in_final_score', [])

                for score_combination_and_thr in score_combination_and_thr_list:
                    scoring_fn = functools.partial(_scoring_fn, score_types=set(score_combination_and_thr.keys())-set(ignore_in_final_score))
                    for scoring_thrs in combitorial_args_fn(score_combination_and_thr):
                        scoring_thrs_fn = functools.partial(_scoring_thrs_fn, score_thr_dict=scoring_thrs)
                        
                        # --- FINAL FIX START ---
                        # 确保 final_score_thr 也是一个列表以便迭代
                        final_score_thr_list = args_eval.get('score_thr_final', [-np.inf])
                        if not isinstance(final_score_thr_list, list): final_score_thr_list = [final_score_thr_list]
                        for final_score_thr in final_score_thr_list:
                        # --- FINAL FIX END ---
                            
                            # --- UNCERTAINTY FILTERING MODIFICATION START ---
                            # 将不确定性阈值列表加入到循环中
                            uncertainty_thr_list = args_eval.get('uncertainty_thr', [float('inf')])
                            if not isinstance(uncertainty_thr_list, list): uncertainty_thr_list = [uncertainty_thr_list]
                            
                            for uncertainty_thr in uncertainty_thr_list:
                                scoring_str = "+".join(scoring_thrs.keys())
                                scoring_thr_str_list = ["%s=%.2f" % (t, thr) for t, thr in scoring_thrs.items() if thr is not None]
                                scoring_thr_str = "-".join(scoring_thr_str_list)

                                if args_eval.get('top_k_predictions'):
                                    exp_name = "%s-final_score_thr=%.2f-%s-unc_thr=%.2f-top_k_predictions=%d" % (scoring_str,final_score_thr,scoring_thr_str,uncertainty_thr,int(args_eval['top_k_predictions']))
                                else:
                                    exp_name = "%s-final_score_thr=%.2f-%s-unc_thr=%.2f" % (scoring_str,final_score_thr,scoring_thr_str,uncertainty_thr)
                                
                                exp_attributes = dict(scoring=scoring_str,scoring_thr=scoring_thr_str, final_score_thr=final_score_thr, uncertainty_thr=uncertainty_thr)

                                center_eval = []
                                if args_eval.get('centers_global_minimization') is not None:
                                    center_eval += [CenterGlobalMinimizationEval(exp_name=exp_name, exp_attributes=exp_attributes, **args) for args in combitorial_args_fn(args_eval['centers_global_minimization'])]
                                if not args_eval.get('skip_center_eval'):
                                    center_eval += [CenterEvaluation()]
                                
                                orientation_eval = [OrientationEval(exp_name=exp_name, exp_attributes=exp_attributes, **o_args) for o_args in combitorial_args_fn(args_eval['orientation'])] if args_eval.get('orientation') else []

                                add_multivariate_eval_fn = lambda eval_obj: MultivariateEval(eval_obj, **args['eval']['enable_multivariate_eval']) if args['eval'].get('enable_multivariate_eval') else eval_obj

                                evaluation_lists.append(dict(
                                    scoring_fn=scoring_fn,
                                    scoring_thrs_fn=scoring_thrs_fn,
                                    final_score_thr=final_score_thr,
                                    uncertainty_thr=uncertainty_thr,
                                    center_eval=[add_multivariate_eval_fn(c) for c in center_eval],
                                    orientation_eval=[add_multivariate_eval_fn(o) for o in orientation_eval],
                                    top_k_predictions=args_eval.get('top_k_predictions')
                                ))
                            # --- UNCERTAINTY FILTERING MODIFICATION END ---
            
            if args.get('skip_if_exists') and args['save_dir'] is not None:
                if self._check_if_results_exist(evaluation_lists, self._get_save_dir(center_model_name)):
                    evaluation_lists = []
            
            evaluation_lists_per_center_model[center_model_name] = evaluation_lists

        return evaluation_lists_per_center_model

    def _get_checkpoint_timestamp(self):
        get_date_fn = lambda p: os.path.getmtime(p) if os.path.exists(p) else 0
        args = self.args
        mod_date = get_date_fn(args['checkpoint_path'])
        return mod_date
    
    def _check_if_results_exist(self, evaluation_lists, save_dir):
        checkpoint_time = self._get_checkpoint_timestamp()
        if checkpoint_time == 0: return False
        c_eval_exists, o_eval_exists, i_eval_exists = [], [], []
        for eval_args in evaluation_lists:
            C = [c_eval.get_results_timestamp(save_dir) >= checkpoint_time for c_eval in eval_args['center_eval']]
            O = [o_eval.get_results_timestamp(save_dir) >= checkpoint_time for o_eval in eval_args['orientation_eval']]
            c_eval_exists.append(all(C)); o_eval_exists.append(all(O))
        return all(c_eval_exists) and all(o_eval_exists) and all(i_eval_exists)
    
    def _get_save_dir(self, center_model_name):
        MARKER = self.args.get('center_checkpoint_name', '##CENTER_MODEL_NAME##')
        if MARKER in self.args['save_dir']: return self.args['save_dir'].replace(MARKER, center_model_name)
        else: return self.args['save_dir']

    def _visualize_prediction(self, sample, result, predictions, predictions_score, eval_obj, eval_res,
                              difficult, impath, save_root):
        args = self.args
        if type(eval_obj) == MultivariateEval: eval_obj = eval_obj.eval_obj
        if eval_res is None: return
        elif type(eval_obj) in [CenterEvaluation, CenterGlobalMinimizationEval, OrientationEval]:
            gt_missed, pred_missed, pred_gt_match, filename_suffix, pred_gt_match_idx = eval_res
            pred_polygon = {}
        else: raise Exception("Unsupported type of evaluator for visualization")
        if args['display'] is True or (isinstance(args['display'], str) and (args['display'].lower() == 'all' or (args['display'].lower() == 'error_gt' and gt_missed) or (args['display'].lower() == 'error' and (gt_missed or pred_missed)))):
            visualize_to_folder = os.path.join(save_root, eval_obj.exp_name, eval_obj.save_str())
            os.makedirs(visualize_to_folder, exist_ok=True)
            if len(predictions) > 0:
                plot_predictions = np.array(predictions)[:, :2]
                plot_predictions_gt_match = np.concatenate((pred_gt_match[:, :1], pred_gt_match_idx[:, :1]), axis=1)
            else:
                plot_predictions, plot_predictions_gt_match = [], []
            base = self.visualizer.impath2name_fn(impath) if impath is not None else None
            self.visualizer(sample, result, plot_predictions, predictions_score, pred_polygon.values(),
                            plot_predictions_gt_match, difficult, filename_suffix+base, visualize_to_folder)

    def run(self, evaluation_lists_per_center_model):
        args = self.args
        with torch.no_grad():
            for im_index,(sample,result) in enumerate(self.processed_image_iter(self.dataset_it, self.centerdir_groundtruth_op)):
                im_shape = sample.get('im_shape', sample.get('image').shape[-2:])
                im_name = sample['im_name']
                instances_ids = sample.get('instance_ids', tensor_mask_to_ids(sample.get('instance')))
                gt_centers_dict = sample.get('center_dict'); ignore_flags = sample.get('ignore')
                difficult = (ignore_flags & 8 > 0).squeeze() if ignore_flags is not None else torch.sparse_coo_tensor(size=im_shape)
                predictions_ = result['predictions']; pred_angle_ = result.get('pred_angle')
                uncertainty_map = result.get('uncertainty_map')
                center_model_name = result['center_model_name']
                all_scores = predictions_[:, 2:] if len(predictions_) > 0 else []
                assert center_model_name in evaluation_lists_per_center_model
                save_vis_root = self._get_save_dir(center_model_name)
                
                for eval_args in evaluation_lists_per_center_model[center_model_name]:
                    scoring_fn = eval_args['scoring_fn']; scoring_thrs_fn = eval_args['scoring_thrs_fn']
                    final_score_thr = eval_args['final_score_thr']
                    
                    if len(predictions_) > 0:
                        predictions_score = scoring_fn(all_scores)
                        selected_pred_idx_stage1 = np.where((predictions_score > final_score_thr) * scoring_thrs_fn(all_scores))[0]
                        predictions_stage1 = predictions_[selected_pred_idx_stage1,:]
                        
                        # --- UNCERTAINTY FILTERING MODIFICATION START ---
                        uncertainty_thr = eval_args.get('uncertainty_thr', float('inf'))
                        if uncertainty_map is not None and len(predictions_stage1) > 0:
                            x_coords = predictions_stage1[:, 0].astype(int); y_coords = predictions_stage1[:, 1].astype(int)
                            point_uncertainties = uncertainty_map[0, 0, y_coords, x_coords].cpu().numpy()
                            # 使用全局阈值以保持不确定性绝对刻度的一致性
                            uncertainty_mask = point_uncertainties <= uncertainty_thr
                            
                            predictions = predictions_stage1[uncertainty_mask, :]
                            predictions_score = predictions_score[selected_pred_idx_stage1][uncertainty_mask]
                            if pred_angle_ is not None: pred_angle = pred_angle_[selected_pred_idx_stage1,:][uncertainty_mask, :]
                        else:
                            predictions = predictions_stage1
                            predictions_score = predictions_score[selected_pred_idx_stage1]
                            if pred_angle_ is not None: pred_angle = pred_angle_[selected_pred_idx_stage1,:]
                        # --- UNCERTAINTY FILTERING MODIFICATION END ---
                    else:
                        predictions, predictions_score, pred_angle = [], [], []

                    if eval_args.get('top_k_predictions'):
                        top_k = eval_args['top_k_predictions']
                        k = min(top_k, len(predictions))
                        predictions = predictions[:k]; predictions_score = predictions_score[:k]
                        if pred_angle_ is not None: pred_angle = pred_angle[:k]

                    for c_eval in eval_args['center_eval']:
                        center_eval_res = c_eval.add_image_prediction(im_name, im_index, im_shape, predictions, predictions_score, instances_ids, gt_centers_dict, difficult, sample.get('centerdir_groundtruth'), return_matched_gt_idx=True)
                        self._visualize_prediction(sample, result, predictions, predictions_score, c_eval, center_eval_res, difficult, im_name, save_vis_root )

                    for o_eval in eval_args['orientation_eval']:
                        orient_eval_res = o_eval.add_image_prediction(im_name, im_index, im_shape, predictions, predictions_score, pred_angle, instances_ids, gt_centers_dict, difficult, sample.get('centerdir_groundtruth'), return_matched_gt_idx=True)
                        self._visualize_prediction(sample, result, predictions, predictions_score, o_eval, orient_eval_res, difficult, im_name, save_vis_root)

            if args.get('eval', True):
                for center_model_name, evaluation_lists in evaluation_lists_per_center_model.items():
                    save_dir = self._get_save_dir(center_model_name)
                    for eval_args in evaluation_lists:
                        for c_eval in eval_args['center_eval'] + eval_args['orientation_eval']:
                            c_eval.calc_and_display_final_metrics(self.dataset_it, save_dir=save_dir)

    def save_args(self, save_dir, extra_args=None):
        if extra_args is not None:
            args = copy.deepcopy(self.args); args.update(extra_args)
        else:
            args = self.args
        with open(os.path.join(save_dir, 'eval_params.json'), 'w') as file:
            file.write(json.dumps(args, indent=4, sort_keys=True, default=lambda o: '<not serializable>'))

def main():
    from config import get_config_args
    args = get_config_args(dataset=os.environ.get('DATASET'), type='test')
    eval = Evaluator(args)
    evaluation_lists = eval.compile_evaluation_list()
    if any([len(e) > 0 for e in evaluation_lists.values()]):
        eval.initialize()
        eval.run(evaluation_lists)
    else:
        print('Skipping due to already existing output')

if __name__ == "__main__":
    main()
