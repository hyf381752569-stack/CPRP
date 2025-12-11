import torch
import torch.nn as nn
import numpy as np
from models.localization.centers import Conv1dMultiscaleLocalization, Conv2dDilatedLocalization

class CenterEstimatorFast(nn.Module):
    def __init__(self, args=dict(), is_learnable=True):
        super().__init__()
        
        # --- UNCERTAINTY FILTERING MODIFICATION START ---
        self.probabilistic = args.get('probabilistic', False)
        self.mixture_model = args.get('mixture_model', False)
        self.num_mixtures = args.get('num_mixtures', 1)
        # --- UNCERTAINTY FILTERING MODIFICATION END ---
        
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
            apply_input_smoothing_for_local_max=0,
            use_findcontours_for_local_max=args.get('use_findcontours_for_local_max', False),
            local_max_min_dist=1,
            return_time=True
        )

    def set_return_backbone_only(self, val): pass
    def is_return_backbone_only(self): return False

    def init_output(self, num_vector_fields=1):
        self.num_vector_fields = num_vector_fields
        assert self.num_vector_fields >= 3
        self.instance_center_estimator.init_output()
        # Original code had a bug here, `input` is not defined
        return None

    def forward(self, input):
        # --- UNCERTAINTY FILTERING MODIFICATION START ---
        # Fast版本同样需要解码，但不传递uncertainty_map，因为其返回签名不同
        if self.probabilistic:
            total_tasks = self.num_vector_fields
            if self.mixture_model and self.num_mixtures > 1:
                K = self.num_mixtures
                B, H, W = input.shape[0], input.shape[2], input.shape[3]
                mu_all = input[:, 0 : total_tasks*K].view(B, total_tasks, K, H, W)
                alpha_logits_all = input[:, 2*total_tasks*K :].view(B, total_tasks, K, H, W)
                if self.training:
                    best_component_indices = torch.argmax(alpha_logits_all, dim=2, keepdim=True)
                    processed_input = torch.gather(mu_all, 2, best_component_indices).squeeze(2)
                else:
                    alpha_prob = torch.softmax(alpha_logits_all, dim=2)
                    processed_input = torch.sum(alpha_prob * mu_all, dim=2)
            else: # 单高斯
                processed_input = input[:, 0:total_tasks]
        else: # 确定性
            processed_input = input
        
        assert processed_input.shape[1] >= self.num_vector_fields
        predictions = processed_input[:, 0:self.num_vector_fields]
        # --- UNCERTAINTY FILTERING MODIFICATION END ---

        S = predictions[:, 0].unsqueeze(1)
        C = predictions[:, 1].unsqueeze(1)

        res, _, times = self.instance_center_estimator(C, S, None, None, None, ignore_region=None)
        center_pred = res

        return center_pred, times
