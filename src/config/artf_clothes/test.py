import copy
import os

import torchvision
if 'InterpolationMode' in dir(torchvision.transforms):
	from torchvision.transforms import InterpolationMode
else:
	from PIL import Image as InterpolationMode

import torch
from utils import transforms as my_transforms

from config.modeling_variants import get_modeling_spec

ARTF_CLOTHES_DIR = os.environ.get('ARTF_CLOTHES_DIR')

if ARTF_CLOTHES_DIR is None:
	raise EnvironmentError("Environment variable ARTF_CLOTHES_DIR must be set to the dataset root.")

OUTPUT_DIR=os.environ.get('OUTPUT_DIR',default='../exp')

NUM_FIELDS = 5
TRAIN_SIZE = int(os.environ.get('TRAIN_SIZE', default=512))
TEST_SIZE = int(os.environ.get('TEST_SIZE', default=512))
USE_DEPTH = os.environ.get('USE_DEPTH', default='False').lower() == 'true'

if USE_DEPTH:
	raise ValueError("aRTFClothes dataset does not provide depth maps. Disable USE_DEPTH.")

IN_CHANNELS = 3

ORIENTATION_ARGS = dict(
	enable=True,
	no_instance_loss=False,
	regression_loss='l1',
	enable_6dof=False,
	symmetries=None,
	regress_confidence_score=False,
)

MODELING_CFG = dict(
	mode=os.environ.get('MODEL_VARIANT', 'gaussian_mixture'),
	num_mixtures=int(os.environ.get('MODEL_NUM_MIXTURES', '2')),
)

MODELING_SPEC = get_modeling_spec(
	mode=MODELING_CFG['mode'],
	orientation_dims=3 if ORIENTATION_ARGS.get('enable_6dof') else 1,
	include_auxiliary_channel=True,
	orientation_confidence=ORIENTATION_ARGS.get('regress_confidence_score', False),
	num_mixtures=MODELING_CFG['num_mixtures'],
)

MODELING_CFG['num_mixtures'] = MODELING_SPEC.num_mixtures
MODELING_CFG['channel_multiplier'] = MODELING_SPEC.channel_multiplier

NUM_VECTOR_FIELDS = MODELING_SPEC.num_vector_fields


def img2tags_fn(x):
	bg = x.split('/')[-4]
	cloth = x.split('/')[-3]
	return [bg, cloth]

model_dir = os.path.join(OUTPUT_DIR, 'artf_clothes', '{args[ablation_str]}',
						  'backbone={args[model][kwargs][backbone]}' + f'_size={TRAIN_SIZE}x{TRAIN_SIZE}',
						  'modeling={args[modeling][mode]}',
						  'mixtures={args[modeling][num_mixtures]}',
						  'num_train_epoch={args[train_settings][n_epochs]}',
						  'depth={args[model][kwargs][use_depth]}',
						  'multitask_weight={args[train_settings][multitask_weighting][name]}')

args = dict(

	cuda=True,
	display=True,
	autoadjust_figure_size=True,
	groundtruth_loading = True,
	save=True,
	save_dir=os.path.join(model_dir,'{args[dataset][kwargs][split]}_results{args[eval_epoch]}',f'test_size={TEST_SIZE}x{TEST_SIZE}',),
	checkpoint_path=os.path.join(model_dir,'checkpoint{args[eval_epoch]}.pth'),

	eval_epoch='',
	ablation_str='',
	modeling=MODELING_CFG,
	num_vector_fields=NUM_VECTOR_FIELDS,

	eval=dict(
		score_combination_and_thr=[
			{
			'center': [0.1,0.01,0.05,0.15,0.2,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.7,0.75,0.8,0.85,0.9,0.94,0.99],
			},
		],
		score_thr_final=[0.01],
		skip_center_eval=True,
		orientation=dict(
			display_best_threshold=False,
			tau_thr=[20], 
		),
		enable_multivariate_eval=dict(
			image2tags_fn=img2tags_fn,
		),
        uncertainty_thr=[float('inf')],
	),
	visualizer=dict(name='OrientationVisualizeTest',
					opts=dict(show_rot_axis=(True,),
							  impath2name_fn=lambda x: ".".join(x.split('/')[-4:]).replace('.jpg','').replace('.png',''))),

	dataset={
		'name': 'artf_clothes',
		'kwargs': {
			'root_dir': os.path.abspath(ARTF_CLOTHES_DIR),
			'split': 'test',
			'fixed_bbox_size': 15,
			'resize_factor': 1,
			'transform': my_transforms.get_transform([
				{'name': 'ToTensor','opts': {'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask'),'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor,torch.ByteTensor)}},
				{'name': 'Resize','opts': {'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask'),'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.NEAREST),'keys_bbox': ('center',),'size': (TEST_SIZE, TEST_SIZE)}},
			]),
			'MAX_NUM_CENTERS':16*128,
		},
		'centerdir_gt_opts': dict(
			ignore_instance_mask_and_use_closest_center=True,
			center_ignore_px=3,
			MAX_NUM_CENTERS=16*128,
		),
		'batch_size': 1,
		'workers': 0,
	},

	model=dict(
		name='fpn',
		kwargs={
			'backbone': 'tu-convnext_base',
			'use_depth': False,
			'num_classes': MODELING_SPEC.num_classes,
			'use_custom_fpn': True,
			'add_output_exp': False,
			'in_channels': IN_CHANNELS,
			'fpn_args': {
				'decoder_segmentation_head_channels': 64,
				'upsampling':4,
				'classes_grouping': MODELING_SPEC.classes_grouping,
				'depth_mean': 0, 'depth_std': 1,
			},
			'init_decoder_gain': 0.1
		},
	),
	center_model=dict(
		name='CenterOrientationEstimator',
		use_learnable_center_estimation=True,
		kwargs=dict(
            probabilistic=MODELING_SPEC.probabilistic,
            mixture_model=MODELING_SPEC.mixture_model,
            num_mixtures=MODELING_SPEC.num_mixtures,
			use_centerdir_radii = False,
			use_magnitude_as_mask=True,
			local_max_thr=0.01, local_max_thr_use_abs=True,
			use_dilated_nn=True,
			dilated_nn_args=dict(
				return_sigmoid=False,
				inner_ch=16,
				inner_kernel=3,
				dilations=[1, 4, 8, 12],
				use_centerdir_radii=False,
				use_centerdir_magnitude=False,
				use_cls_mask=False
				),
			augmentation=False,
			scale_r=1.0,
			scale_r_gt=1,
			use_log_r=False,
			use_log_r_base='10',
			enable_6dof=False,
		),
	),
)


train_settings = dict(
	n_epochs=10,
	multitask_weighting=dict(
		name='off',
		kwargs=dict(
			n_tasks=2
		)
	),
)

args['train_settings'] = train_settings


def get_args():
	return copy.deepcopy(args)
