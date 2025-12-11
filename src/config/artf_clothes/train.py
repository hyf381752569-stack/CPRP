import copy
import os

import torch
from utils import transforms as my_transforms
from torchvision.transforms import InterpolationMode

from config.modeling_variants import get_modeling_spec

ARTF_CLOTHES_DIR = os.environ.get('ARTF_CLOTHES_DIR')

if ARTF_CLOTHES_DIR is None:
	raise EnvironmentError("Environment variable ARTF_CLOTHES_DIR must be set to the dataset root.")

OUTPUT_DIR=os.environ.get('OUTPUT_DIR',default='../exp')

NUM_FIELDS = 5
SIZE = int(os.environ.get('TRAIN_SIZE', default=512))
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

args = dict(

	cuda=True,
	display=False,
	display_it=20,

	tf_logging=['loss'],
	tf_logging_iter=2,

	visualizer=dict(name='OrientationVisualizeTrain'),

	save=True,
	save_interval=2,

	# --------
	n_epochs=10,
	ablation_str="",
	modeling=MODELING_CFG,

	save_dir=os.path.join(OUTPUT_DIR, 'artf_clothes', '{args[ablation_str]}',
						  'backbone={args[model][kwargs][backbone]}' + f'_size={SIZE}x{SIZE}',
						  'modeling={args[modeling][mode]}',
						  'mixtures={args[modeling][num_mixtures]}',
						  'num_train_epoch={args[n_epochs]}',
						  'depth={args[model][kwargs][use_depth]}',
						  'multitask_weight={args[multitask_weighting][name]}',
						  ),


	pretrained_model_path = None,
	resume_path = None,

	pretrained_center_model_path = None,


	train_dataset = {
		'name': 'artf_clothes',
		'kwargs': {
			'root_dir': os.path.abspath(ARTF_CLOTHES_DIR),
			'split': 'train',
			'fixed_bbox_size': 15,
			'resize_factor': 1,
			'transform_per_sample_rng': True,
			'transform': my_transforms.get_transform([
				# for training without augmentation (same as testing)
				{
					'name': 'ToTensor',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask'),
						'type': (
						torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.ByteTensor),
					}
				},
				{
					'name': 'Resize',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask'),
						'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.NEAREST),
						'keys_bbox': ('center',),
						'size': (SIZE, SIZE),
					}
				},
				# for training with random augmentation
				{
				    'name': 'RandomGaussianBlur',
				    'opts': {
				        'keys': ('image',),
				        'rate': 0.5, 'sigma': [0.5, 2]
				    }
				},

				{
					'name': 'ColorJitter',
					'opts': {
						'keys': ('image',), 'p': 0.5,
						'saturation': 0.3, 'hue': 0.3, 'brightness': 0.3, 'contrast':0.3
					}
				}

			]),
			'MAX_NUM_CENTERS':16*128,
		},

		'centerdir_gt_opts': dict(
			ignore_instance_mask_and_use_closest_center=True, # by default
			center_ignore_px=3,

			skip_gt_center_mask_generate=True, # gt_center_mask is not needed since we are not training localization network

			MAX_NUM_CENTERS=16*128,
		),

		'batch_size': 4,

		# hard example disabled
		'hard_samples_size': 0,
		'hard_samples_selected_min_percent':0.1,

		'workers': 4,
		'shuffle': True,
	}, 

	model = dict(
		name='fpn',
		kwargs= {
			'backbone': 'tu-convnext_base',
			'use_depth': False,
			'num_classes': MODELING_SPEC.num_classes,
			'use_custom_fpn':True,
			'add_output_exp': False,
			'in_channels': IN_CHANNELS,
			'fpn_args': {
				'decoder_segmentation_head_channels':64,
				'upsampling':4,
				'classes_grouping': MODELING_SPEC.classes_grouping,
				'depth_mean': 0,
				'depth_std':1,
			},
			'init_decoder_gain': 0.1
		},
		optimizer='Adam',
		lr=1e-4,
		weight_decay=0,

	),
	center_model=dict(
		name='CenterEstimator',
		kwargs=dict(
			probabilistic=MODELING_SPEC.probabilistic,
			mixture_model=MODELING_SPEC.mixture_model,
			num_mixtures=MODELING_SPEC.num_mixtures,
			# use vector magnitude as mask instead of regressed mask
			use_magnitude_as_mask=False,
			# thresholds for conv2d processing
			local_max_thr=0.1, mask_thr=0.01,  exclude_border_px=0,
			use_dilated_nn=True,
			dilated_nn_args=dict(
				# single scale version (nn6)
				inner_ch=16,
				inner_kernel=3,
				dilations=[1, 4, 8, 12],
				freeze_learning=False,
				gradpass_relu=False,
				# version with leaky relu
				leaky_relu=False,
				# input check
				use_centerdir_radii = False,
				use_centerdir_magnitude = False,
				use_cls_mask = False
			),
			allow_input_backprop=False,
			backprop_only_positive=False,
			augmentation=False, 
			scale_r=1.0,
			scale_r_gt=1024,
			use_log_r=True,
			use_log_r_base='10',
			enable_6dof=False,
		),
		optimizer='Adam',
		lr=1e-4,
		weight_decay=0,
	),


	# loss options
	loss_type='OrientationLoss',
	loss_opts={
		'probabilistic': MODELING_SPEC.probabilistic,
		'mixture_model': MODELING_SPEC.mixture_model,
		'num_mixtures': MODELING_SPEC.num_mixtures,
		'num_vector_fields': NUM_VECTOR_FIELDS,
		'foreground_weight': 1,

		'enable_centerdir_loss': True,
		'no_instance_loss': True,
		'centerdir_instance_weighted': True,
		'regression_loss': 'l1',

		'use_log_r': True,
		'use_log_r_base': '10',

	'orientation_args': ORIENTATION_ARGS,
},
	num_vector_fields=NUM_VECTOR_FIELDS,
	multitask_weighting=dict(
		name='off',
		kwargs=dict(
			n_tasks=2
		)
	),
	loss_w={
		'w_r': 1,
		'w_cos': 1,
		'w_sin': 1,
		'w_cent': 0.1,
		'w_orientation': 1,
	},

)

args['lambda_scheduler_fn']=lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9))
#args['lambda_scheduler_fn']=lambda _args: (lambda epoch: 1.0) # disabled

args['model']['lambda_scheduler_fn'] = args['lambda_scheduler_fn']
args['center_model']['lambda_scheduler_fn'] = lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9) if epoch > 1 else 0)


def get_args():
	return copy.deepcopy(args)
