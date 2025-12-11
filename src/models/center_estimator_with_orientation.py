import torch
import time
from models.center_estimator import CenterEstimator
from models.center_estimator_fast import CenterEstimatorFast

class CenterOrientationEstimator(CenterEstimator):
    def __init__(self, args=dict(), is_learnable=True):
        super(CenterOrientationEstimator, self).__init__(args, is_learnable=is_learnable)
        self.enable_6dof = args.get('enable_6dof')
        self.use_orientation_confidence_score = args.get('use_orientation_confidence_score')

    def init_output(self, num_vector_fields=1):
        super(CenterOrientationEstimator, self).init_output(num_vector_fields)
        REQUIRED_VECTOR_FIELDS = 5
        if self.enable_6dof: REQUIRED_VECTOR_FIELDS += 4 
        if self.use_orientation_confidence_score: REQUIRED_VECTOR_FIELDS += 1
        assert self.num_vector_fields >= REQUIRED_VECTOR_FIELDS

    def forward(self, input, ignore_gt=False, **gt):
        # 父类forward已包含解码逻辑并返回完整的输出字典
        ret = super(CenterOrientationEstimator, self).forward(input, ignore_gt, **gt)
        
        # --- UNCERTAINTY FILTERING MODIFICATION START ---
        # 使用父类返回的原始输入和解码后的结果
        processed_output = ret.get('processed_output', ret['output'])
        center_pred = ret['center_pred']
        # --- UNCERTAINTY FILTERING MODIFICATION END ---

        batch_size = center_pred.shape[0]
        num_pred = center_pred.shape[1]

        if self.enable_6dof:
            sin_orientation = processed_output[:, 3:6]; cos_orientation = processed_output[:, 6:9]
        else:
            sin_orientation = processed_output[:, 3:4]; cos_orientation = processed_output[:, 4:5]
        prediction_angles = torch.zeros((batch_size,num_pred,sin_orientation.shape[1]))

        if self.use_orientation_confidence_score:
            prediction_confidence_score = processed_output[:, 5:6]
            orientation_confidence_score = torch.zeros((batch_size,num_pred,1)).to(center_pred.device)

        for b in range(batch_size):
            for i, pred in enumerate(center_pred[b]):
                if pred[0] != 0:
                    x,y = pred[1:3]
                    s = sin_orientation[b,:,int(y), int(x)]; c = cos_orientation[b,:, int(y), int(x)]
                    pred_angle = torch.atan2(c, s)
                    pred_angle = torch.rad2deg(pred_angle)
                    pred_angle += 360 * (pred_angle < 0).int()
                    prediction_angles[b,i,:] = pred_angle
                    if self.use_orientation_confidence_score:
                        orientation_confidence_score[b,i] = prediction_confidence_score[b,0,int(y), int(x)]

        ret['pred_angle'] = prediction_angles
        if self.use_orientation_confidence_score:
            ret['center_pred'] = torch.cat((center_pred,orientation_confidence_score),axis=2)
        return ret
    
class CenterOrientationEstimatorFast(CenterEstimatorFast):
    def __init__(self, args=dict(), is_learnable=True):
        super(CenterOrientationEstimatorFast, self).__init__(args, is_learnable=is_learnable)
        self.enable_6dof = args.get('enable_6dof')        
        self.use_orientation_confidence_score = args.get('use_orientation_confidence_score')

    def init_output(self, num_vector_fields=1):
        super(CenterOrientationEstimatorFast, self).init_output(num_vector_fields)
        REQUIRED_VECTOR_FIELDS = 5
        if self.enable_6dof: REQUIRED_VECTOR_FIELDS += 4 
        if self.use_orientation_confidence_score: REQUIRED_VECTOR_FIELDS += 1
        assert self.num_vector_fields >= REQUIRED_VECTOR_FIELDS

    def forward(self, input, ignore_gt=False, **gt):
        # --- UNCERTAINTY FILTERING MODIFICATION START ---
        # Fast版本需要自己解码，因为它父类的返回签名不同
        if self.probabilistic:
            total_tasks = self.num_vector_fields
            if self.mixture_model and self.num_mixtures > 1:
                K = self.num_mixtures
                B, H, W = input.shape[0], input.shape[2], input.shape[3]
                mu_all = input[:, 0 : total_tasks*K].view(B, total_tasks, K, H, W)
                alpha_logits_all = input[:, 2*total_tasks*K :].view(B, total_tasks, K, H, W)
                if self.training:
                    best_component_indices = torch.argmax(alpha_logits_all, dim=2, keepdim=True)
                    processed_input_for_super = torch.gather(mu_all, 2, best_component_indices).squeeze(2)
                else:
                    alpha_prob = torch.softmax(alpha_logits_all, dim=2)
                    processed_input_for_super = torch.sum(alpha_prob * mu_all, dim=2)
            else: # 单高斯
                processed_input_for_super = input[:, 0:total_tasks]
        else: # 确定性
            processed_input_for_super = input
        
        center_pred, times = super(CenterOrientationEstimatorFast, self).forward(processed_input_for_super)
        # --- UNCERTAINTY FILTERING MODIFICATION END ---
        
        start_orient = time.time()
        # 用于定向的图谱与传给父类的图谱是同一个
        predictions = processed_input_for_super[:, 0:self.num_vector_fields]
        num_pred = center_pred.shape[0]

        if self.enable_6dof:
            sin_orientation = predictions[:, 3:6]; cos_orientation = predictions[:, 6:9]
        else:
            sin_orientation = predictions[:, 3:4]; cos_orientation = predictions[:, 4:5]
        if self.use_orientation_confidence_score:
            prediction_confidence_score = predictions[:, 5:6]
            orientation_confidence_score = torch.zeros((num_pred,1)).to(center_pred.device)
        prediction_angles = torch.zeros((num_pred,sin_orientation.shape[1]), device=predictions.device)

        if num_pred > 0:
            batch_indices = center_pred[:, 0].long(); x,y = center_pred[:, 1].long(), center_pred[:, 2].long()
            score = center_pred[:, -1]
            if self.use_orientation_confidence_score:                
                confidence_score = prediction_confidence_score[batch_indices, 0, y, x]
                orientation_confidence_score[:,0] = confidence_score
                score *= confidence_score
            
            valid_indices = torch.ones_like(score, dtype=torch.bool)
            if score.mean() > 0.5 and score.mean() < 2.0: # Heuristic check for score range
                 valid_indices = score <= 0.9

            if torch.any(valid_indices):
                s = sin_orientation[batch_indices[valid_indices], :, y[valid_indices], x[valid_indices]]
                c = cos_orientation[batch_indices[valid_indices], :, y[valid_indices], x[valid_indices]]
                pred_angle = torch.atan2(c, s)
                pred_angle = torch.rad2deg(pred_angle)
                pred_angle += 360 * (pred_angle < 0).int()
                prediction_angles[valid_indices,:] = pred_angle
        
        if self.use_orientation_confidence_score:
            center_pred = torch.cat((center_pred, orientation_confidence_score.to(predictions.device)),axis=1)
        center_pred_with_rot = torch.cat((center_pred, prediction_angles.to(predictions.device)),axis=1)
        end_orient = time.time()
        times = list(times)
        times[-1] = times[-1] + (end_orient-start_orient)
        return center_pred_with_rot, tuple(times)
