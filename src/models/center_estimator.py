import torch
import torch.nn as nn
import numpy as np
from models.center_augmentator import CenterAugmentator
from models.localization.centers import Conv1dMultiscaleLocalization, Conv2dDilatedLocalization
import torch.nn.functional as F

class CenterEstimator(nn.Module):
    def __init__(self, args=dict(), is_learnable=True):
        super().__init__()
        
        # --- UNCERTAINTY FILTERING MODIFICATION START ---
        self.probabilistic = args.get('probabilistic', False)
        self.mixture_model = args.get('mixture_model', False)
        self.num_mixtures = args.get('num_mixtures', 1)
        # --- UNCERTAINTY FILTERING MODIFICATION END ---

        self.return_backbone_only = False
        self.use_magnitude_as_mask = args.get('use_magnitude_as_mask')
        instance_center_estimator_op = Conv1dMultiscaleLocalization
        if args.get('use_dilated_nn'):
            from functools import partial
            instance_center_estimator_op = partial(Conv2dDilatedLocalization, **args.get('dilated_nn_args',{}))
        self.instance_center_estimator = instance_center_estimator_op(
            local_max_thr=args.get('local_max_thr', 0.1),
            mask_thr=args.get('mask_thr', 0.01),
            exclude_border_px=args.get('exclude_border_px', 5),
            learnable=is_learnable,
            allow_input_backprop=args.get('allow_input_backprop', True),
            backprop_only_positive=args.get('backprop_only_positive', True),
            apply_input_smoothing_for_local_max=args.get('apply_input_smoothing_for_local_max', 1),
            use_findcontours_for_local_max=args.get('use_findcontours_for_local_max', False),
        )
        if args.get('augmentation'): self.center_augmentator = CenterAugmentator(**args.get('augmentation_kwargs'))
        else: self.center_augmentator = None
        scale_r = 1024 if 'scale_r' not in args else args['scale_r']
        scale_r_gt = 1 if 'scale_r_gt' not in args else args['scale_r_gt']
        self.scale_r_fn = lambda x: x * scale_r
        self.scale_r_gt_fn = lambda x: x * scale_r_gt
        self.inverse_scale_r_fn = lambda x: x / scale_r
        self.use_log_r = args['use_log_r'] if 'use_log_r' in args else True
        use_log_r_base = args['use_log_r_base'] if 'use_log_r_base' in args else 'exp'
        if use_log_r_base.lower() in ['exp', 'e']:
            self.log_r_fn = lambda x: torch.log(x+1); self.inverse_log_r_fn = lambda x: torch.exp(x)-1
        elif use_log_r_base.lower() in ['decimal', '10']:
            self.log_r_fn = lambda x: torch.log10(x+1); self.inverse_log_r_fn = lambda x: torch.pow(10, x)-1
        elif use_log_r_base.lower() in ['pow10']:
            self.log_r_fn = lambda x: torch.log10(x+1); self.inverse_log_r_fn = lambda x: torch.pow(x, 10)-1
        else: raise Exception('Only "exp" and "10" are allowed logarithms for R')
        self.MAX_NUM_CENTERS = 16*128

    def set_return_backbone_only(self, val): self.return_backbone_only = val
    def is_return_backbone_only(self): return self.return_backbone_only
    def init_output(self, num_vector_fields=1):
        self.num_vector_fields = num_vector_fields
        assert self.num_vector_fields >= 3
        self.instance_center_estimator.init_output()
        return None # Original code had a bug here `return input` which is not defined

    def forward(self, input, ignore_gt=False, **gt):
        original_input = input
        uncertainty_map = None
        variance_maps = None
        mixture_weights = None
        mean_maps = None

        # --- UNCERTAINTY FILTERING MODIFICATION START ---
        # 统一解码逻辑，为后续步骤准备好 processed_input 和 uncertainty_map
        if self.probabilistic:
            total_tasks = self.num_vector_fields
            if self.mixture_model and self.num_mixtures > 1:
                K = self.num_mixtures
                expected_channels = total_tasks * K * 3
                assert input.shape[1] == expected_channels, f"GMM mode expected {expected_channels} channels, got {input.shape[1]}"
                B, H, W = input.shape[0], input.shape[2], input.shape[3]
                mu_all = input[:, 0 : total_tasks*K].view(B, total_tasks, K, H, W)
                log_var_all = input[:, total_tasks*K : 2*total_tasks*K].view(B, total_tasks, K, H, W)
                alpha_logits_all = input[:, 2*total_tasks*K :].view(B, total_tasks, K, H, W)

                variance_maps = torch.exp(log_var_all)
                mixture_weights = torch.softmax(alpha_logits_all, dim=2)

                if self.training:
                    best_component_indices = torch.argmax(alpha_logits_all, dim=2, keepdim=True)
                    processed_input = torch.gather(mu_all, 2, best_component_indices).squeeze(2)
                    best_log_var = torch.gather(log_var_all, 2, best_component_indices).squeeze(2)
                    if total_tasks >= 2:
                        uncertainty_map = torch.sqrt(torch.exp(best_log_var[:, 0, ...]) + torch.exp(best_log_var[:, 1, ...])).unsqueeze(1)
                    else:
                        uncertainty_map = torch.sqrt(torch.exp(best_log_var[:, 0, ...])).unsqueeze(1)
                else:
                    alpha_prob = mixture_weights
                    mu_exp = torch.sum(alpha_prob * mu_all, dim=2)
                    processed_input = mu_exp
                    mean_maps = mu_all

                    diff2 = (mu_all - mu_exp.unsqueeze(2)) ** 2
                    var_total = torch.sum(alpha_prob * (variance_maps + diff2), dim=2)

                    if total_tasks >= 2:
                        unc = torch.sqrt(var_total[:, 0, ...] + var_total[:, 1, ...])
                        uncertainty_map = unc.unsqueeze(1)
                    else:
                        uncertainty_map = torch.sqrt(var_total[:, 0, ...]).unsqueeze(1)
            else: # 单高斯
                expected_channels = total_tasks * 2
                assert input.shape[1] == expected_channels, f"Single Gaussian mode expected {expected_channels} channels, got {input.shape[1]}"
                processed_input = input[:, 0:total_tasks]
                log_var_vectors = input[:, total_tasks:]
                variance_maps = torch.exp(log_var_vectors).unsqueeze(2)
                mixture_weights = torch.ones_like(variance_maps)
                mean_maps = processed_input.unsqueeze(2)
                if total_tasks >= 2:
                    uncertainty_map = torch.sqrt(torch.exp(log_var_vectors[:, 0, ...]) + torch.exp(log_var_vectors[:, 1, ...])).unsqueeze(1)
                else:
                    uncertainty_map = torch.sqrt(torch.exp(log_var_vectors[:, 0, ...])).unsqueeze(1)
        else: # 确定性
            processed_input = input
            mean_maps = processed_input.unsqueeze(2)
        # --- UNCERTAINTY FILTERING MODIFICATION END ---

        if self.center_augmentator is not None:
            processed_input = self.center_augmentator(processed_input, **gt)

        ignore = gt.get('ignore')
        assert processed_input.shape[1] >= self.num_vector_fields
        predictions = processed_input[:, 0:self.num_vector_fields]

        S = predictions[:, 0].unsqueeze(1)
        C = predictions[:, 1].unsqueeze(1)
        R = predictions[:, 2].unsqueeze(1)
        R = self.inverse_log_r_fn(self.scale_r_fn(R))
        cls_mask = torch.zeros_like(S, requires_grad=False)
        M = torch.zeros_like(S, requires_grad=False)

        if self.training:
            center_pred, conv_resp = self.instance_center_estimator(C, S, R, M, cls_mask)
        else:
            if self.use_log_r and 'centerdir_groundtruth' in gt and not ignore_gt:
                processed_input[:, 2:3] = R
                if gt.get('centerdir_groundtruth'):
                    gt['centerdir_groundtruth'][0][:, 0] = self.scale_r_gt_fn(gt['centerdir_groundtruth'][0][:, 0])
            
            mask = M if self.use_magnitude_as_mask else (cls_mask * (cls_mask > 0).type(torch.float32))
            res, conv_resp = self.instance_center_estimator(C, S, R, M, mask,
                                                            ignore_region=ignore & 1 if ignore is not None else None)
            conv_resp = torch.relu(conv_resp)

            if res.numel() > 0:
                res = torch.cat((res, torch.ones((len(res),1),device=res.device)),dim=1)
                res = res.cpu().numpy()
                if len(res) > 0:
                    idx = np.lexsort((res[:, 3],res[:, 4]))
                    res = res[idx[::-1], :]
                if res.shape[0] > 2000:
                    res = res[:2000, :]
                if len(res) > 0:
                    selected_centers = np.ones(len(res),dtype=bool)
                    for b in range(len(processed_input)):
                        batch_idx = res[:, 0] == b
                        if not np.any(batch_idx): continue
                        centers_b = res[batch_idx][:, [2, 1, 4]]
                        if ignore is not None and len(res) > 0:
                            ignored_pred = np.array([ignore.clone().cpu().numpy()[b, 0, int(r[0]), int(r[1])] & (255 - 8 - 64 - 128) == 0 for r in centers_b]).astype(bool)
                            if not np.all(ignored_pred):
                                selected_centers[batch_idx] *= ignored_pred
                    center_pred = res[selected_centers, :]
                else:
                    center_pred = res
            else:
                center_pred = res

        center_pred = torch.tensor(center_pred, dtype=torch.float32).to(processed_input.device)
        center_pred = self._pack_center_predictions(center_pred, batch_size=len(processed_input))

        # --- UNCERTAINTY FILTERING MODIFICATION START ---
        return dict(
            output=original_input,
            processed_output=processed_input,
            center_pred=center_pred,
            center_heatmap=conv_resp,
            uncertainty_map=uncertainty_map,
            variance_maps=variance_maps,
            mixture_weights=mixture_weights,
            mean_maps=mean_maps,
        )
        # --- UNCERTAINTY FILTERING MODIFICATION END ---

    def _pack_center_predictions(self, center_pred, batch_size):
        center_pred_all = torch.zeros((batch_size, self.MAX_NUM_CENTERS, center_pred.shape[1] if len(center_pred) > 0 else 5),
                                      dtype=torch.float, device=center_pred.device)
        if len(center_pred) > 0:
            for b in center_pred[:, 0].unique().long():
                valid_centers_idx = torch.nonzero(center_pred[:, 0] == b.float()).squeeze(dim=1).long()
                if len(valid_centers_idx) > self.MAX_NUM_CENTERS:
                    valid_centers_idx = valid_centers_idx[:self.MAX_NUM_CENTERS]
                center_pred_all[b, :len(valid_centers_idx), 0] = 1
                center_pred_all[b, :len(valid_centers_idx), 1:] = center_pred[valid_centers_idx, 1:]
        return center_pred_all
