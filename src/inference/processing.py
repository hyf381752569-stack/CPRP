import os
from tqdm import tqdm
import numpy as np
import torch
from models.center_groundtruth import CenterDirGroundtruth

class CenterDirProcesser:
    def __init__(self, model, center_model_list, device=None):
        self.model = model
        self.center_model_list = center_model_list
        self.device = device

    def get_center_model_list(self): return self.center_model_list
    def clean_memory(self):
        self.model.cpu(); self.model = None
        if self.center_model_list is not None:
            for center_model_desc in self.get_center_model_list():
                center_model_desc['model'].cpu(); center_model_desc['model'] = None

    def __call__(self, dataset_it, centerdir_groundtruth_op=None, tqdm_kwargs={}):
        assert self.model is not None
        self.model.eval()
        for center_model_desc in self.get_center_model_list():
            assert center_model_desc['model'] is not None
            center_model_desc['model'].eval()

        im_image = 0
        for sample_ in tqdm(dataset_it, **tqdm_kwargs):
            if centerdir_groundtruth_op is not None:
                sample_ = centerdir_groundtruth_op(sample_, torch.arange(0, dataset_it.batch_size).int())
                model = self.model

            im_batch = sample_['image']
            output_batch_ = model(im_batch)

            for center_model_desc in self.get_center_model_list():
                center_model_name = center_model_desc['name']
                center_model = center_model_desc['model']
                center_output = center_model(output_batch_, **sample_)

                # --- 2-STAGE FILTER MODIFICATION START ---
                processed_output_batch = center_output.get('processed_output', center_output['output'])
                center_pred = center_output['center_pred']
                center_heatmap = center_output['center_heatmap']
                raw_output_batch = center_output['output']
                uncertainty_map = center_output.get('uncertainty_map')
                variance_maps = center_output.get('variance_maps')
                mixture_weights = center_output.get('mixture_weights')
                mean_maps = center_output.get('mean_maps')
                # --- 2-STAGE FILTER MODIFICATION END ---
                
                pred_angle = center_output.get('pred_angle')
                gt_centers = None
                if 'centerdir_groundtruth' in sample_:
                    gt_centers = CenterDirGroundtruth.parse_groundtruth_map(sample_['centerdir_groundtruth'],keys=['gt_centers'])
                elif 'center' in sample_:
                    gt_centers = sample_['center'][:,:,[1,0]]

                if gt_centers is not None:
                    instances = sample_['instance'].squeeze(dim=1)
                    center_ignore = sample_['ignore'] == 1 if 'ignore' in sample_ else None
                    gt_centers_dict = CenterDirGroundtruth.convert_gt_centers_to_dictionary(gt_centers, instances=instances, ignore=center_ignore)
                else: gt_centers_dict = []

                sample_keys = sample_.keys()
                for batch_i in range(min(dataset_it.batch_size, len(sample_['im_name']))):
                    im_image += 1
                    output = processed_output_batch[batch_i:batch_i + 1]
                    raw_output = raw_output_batch[batch_i:batch_i + 1] if raw_output_batch is not None else None
                    sample = {k: sample_[k][batch_i:batch_i + 1] for k in sample_keys}
                    im_name = sample['im_name'][0]
                    instance = sample['instance'].squeeze()
                    ignore = sample.get('ignore')
                    if 'centerdir_groundtruth' in sample_:
                        sample['centerdir_groundtruth'] = sample_['centerdir_groundtruth'][0][batch_i]
                    if len(gt_centers_dict) > 0:
                        center_dict = gt_centers_dict[batch_i]
                        if ignore is not None:
                            for id in instance.unique():
                                id = id.item()
                                if id > 0 and id not in center_dict.keys(): instance[instance == id] = 0
                    else: center_dict = None

                    pred_heatmap = torch.relu(center_heatmap[batch_i].unsqueeze(0))
                    predictions = center_pred[batch_i][center_pred[batch_i,:,0] == 1][:,1:].cpu().numpy()
                    idx = np.argsort(predictions[:, -1]); idx = idx[::-1]; predictions = predictions[idx, :]
                    pred_angle_b = pred_angle[batch_i].cpu().numpy()[idx, :] if pred_angle is not None else None
                    if pred_angle_b is not None: assert len(pred_angle_b) == len(predictions)

                    # --- 2-STAGE FILTER MODIFICATION START ---
                    uncertainty_map_b = uncertainty_map[batch_i:batch_i+1] if uncertainty_map is not None else None
                    variance_maps_b = variance_maps[batch_i:batch_i+1] if variance_maps is not None else None
                    mixture_weights_b = mixture_weights[batch_i:batch_i+1] if mixture_weights is not None else None
                    mean_maps_b = mean_maps[batch_i:batch_i+1] if mean_maps is not None else None
                    output_dict = dict(output=output, raw_output=raw_output, predictions=predictions, pred_heatmap=pred_heatmap,
                                       pred_angle=pred_angle_b, center_model_name=center_model_name,
                                       uncertainty_map=uncertainty_map_b,
                                       variance_maps=variance_maps_b,
                                       mixture_weights=mixture_weights_b,
                                       mean_maps=mean_maps_b)
                    # --- 2-STAGE FILTER MODIFICATION END ---
                    
                    sample.update(dict(im_name=im_name, instance=instance, center_dict=center_dict,))
                    yield sample, output_dict
