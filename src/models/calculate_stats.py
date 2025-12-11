# calculate_stats.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import numpy as np
from typing import Iterator
import itertools
from thop import profile
import pandas as pd
import warnings
from typing import Optional, Union
import copy # 我们不再需要 deepcopy，但保留 import 以防万一

# 抑制 thop 和其他库可能产生的警告
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 模拟 MultiTaskModel (为了让 FPN.py 的代码能独立运行)
# ==============================================================================
class MultiTaskModel:
    """
    这是一个模拟类，用于满足 FPN.py 中 'from .multitask_model import MultiTaskModel' 的依赖。
    """
    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        # print("Warning: Using mocked MultiTaskModel.shared_parameters")
        return iter([]) # 返回一个空迭代器
    
    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        # print("Warning: Using mocked MultiTaskModel.task_specific_parameters")
        return iter([])

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        # print("Warning: Using mocked MultiTaskModel.last_shared_parameters")
        return iter([])

# ==============================================================================
# 2. 粘贴 FPN.py 的全部内容
# (已应用所有修复)
# ==============================================================================
class FPN(nn.Module, MultiTaskModel):
    def __init__(self, num_classes, backbone='resnet34', use_depth=False, use_custom_fpn=False, add_output_exp=False, init_decoder_gain=0.1, fpn_args={}, pretrained=True, in_channels=None, use_only_depth=True):
        super().__init__()

        # --- 修复 1：创建 fpn_args 的本地副本以防止在循环中破坏它 ---
        local_fpn_args = fpn_args.copy()

        self.use_custom_fpn = use_custom_fpn
        print('Creating FPN segmentation with {} classes'.format(num_classes))
        if 'encoder_depth' not in local_fpn_args:
            local_fpn_args['encoder_depth'] = 4
        if 'upsampling' not in local_fpn_args:
            local_fpn_args['upsampling'] = 2

        self.use_only_depth = use_only_depth

        if use_depth:
            if not use_only_depth:
                # 从原始 fpn_args (未修改的) 中读取
                self.depth_mean = fpn_args['depth_mean'] 
                self.depth_std = fpn_args['depth_std']
        
        # 从本地副本中安全地移除键
        local_fpn_args.pop('depth_mean', None)
        local_fpn_args.pop('depth_std', None)
            # else:

        

        self.use_depth = use_depth
        self.in_channels = in_channels if in_channels is not None else (4 if use_depth else 3)

        if self.use_custom_fpn:
            self.model = Custom_FPN(backbone, classes=sum(num_classes), activation=None, in_channels=self.in_channels, decoder_merge_policy='add', encoder_weights='imagenet' if pretrained else None, **local_fpn_args)
        else:
            # --- 修复 2：将 in_channels=self.in_channels 传递给 smp.FPN ---
            self.model = smp.FPN(backbone, classes=sum(num_classes), activation=None,
                                 in_channels=self.in_channels, # <--- 修复在此
                                 decoder_merge_policy='add', encoder_weights='imagenet' if pretrained else None, **local_fpn_args)

        self.preprocess_params = smp.encoders.get_preprocessing_params(backbone, pretrained="imagenet")
        if use_depth:
            if not use_only_depth:
                # 检查 self.depth_mean 是否已在 'if not use_only_depth' 块中被定义
                if hasattr(self, 'depth_mean') and (type(self.depth_mean)==int or type(self.depth_mean)==float): # if one number, add to mean and std vectors
                    self.preprocess_params['mean'].append(self.depth_mean)
                    self.preprocess_params['std'].append(self.depth_std)
                elif hasattr(self, 'depth_mean'): # if list, extend the vectors
                    self.preprocess_params['mean'].extend(self.depth_mean)
                    self.preprocess_params['std'].extend(self.depth_std)
            else:
                self.preprocess_params['mean'] = [0]
                self.preprocess_params['std'] = [1]


        self.add_output_exp = add_output_exp
        self.init_decoder_gain = init_decoder_gain

        if self.add_output_exp:
            output_exp_scale_var = torch.ones(size=(sum(num_classes),), dtype=torch.float32, requires_grad=True)
            self.output_exp_scale = torch.nn.Parameter(output_exp_scale_var)
            self.register_parameter('output_exp_scale',self.output_exp_scale)

    def init_output(self, num_vector_fields=1):
        with torch.no_grad():
            decoder = self.model.decoder
            decoder_convs = []
            if self.use_custom_fpn:
                for head in self.model.segmentation_head_list if len(self.model.segmentation_head_list) > 0 else [self.model.segmentation_head]:
                    output_conv = head[0]
                    decoder_convs += [output_conv.block[0], head[2]]
                decoder_convs += [decoder.p2.skip_conv, decoder.p3.skip_conv, decoder.p4.skip_conv, decoder.p5]
            else:
                output_conv = self.model.segmentation_head[0]
                decoder_convs += [output_conv]
                decoder_convs += [decoder.p2.skip_conv, decoder.p3.skip_conv, decoder.p4.skip_conv, decoder.p5]

            for seg_block in decoder.seg_blocks:
                for conv_block in seg_block.block:
                    decoder_convs.append(conv_block.block[0])

            for c in decoder_convs:
                if type(c) == torch.nn.modules.conv.Conv2d:
                    print('initialize decoder layer with size: ', c.weight.size())
                    torch.nn.init.xavier_normal_(c.weight,gain=self.init_decoder_gain)
                    if c.bias is not None:
                        torch.nn.init.zeros_(c.bias)


    def forward(self, input, only_encode=False):
        input = preprocess_input(input,**self.preprocess_params)
        output = self.model.forward(input)
        if self.add_output_exp:
            output = torch.stack((output[:,0],output[:,1],
                                  torch.exp(torch.exp(self.output_exp_scale[2])*output[:,2])-1,
                                  output[:,3]),dim=1)

        return output

    # --- 修复 3：正确实现 MultiTaskModel 的方法 ---
    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters shared by all tasks."""
        if isinstance(self.model, MultiTaskModel):
            return self.model.shared_parameters()
        else:
            # 假定 smp.FPN 的 encoder 和 decoder 是共享的
            return itertools.chain(self.model.encoder.parameters(), self.model.decoder.parameters())

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters specific to each task."""
        if isinstance(self.model, MultiTaskModel):
            return self.model.task_specific_parameters()
        else:
            # 假定 smp.FPN 的 segmentation_head 是特定于任务的
            return self.model.segmentation_head.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer."""
        if isinstance(self.model, MultiTaskModel):
            return self.model.last_shared_parameters()
        else:
            # 假定 smp.FPN 的 decoder 是最后一个共享层
            return self.model.decoder.parameters()


def _find_last_conv_index(sequence):
    # find last convolution layer in segmentation head
    return max([i for i, layer in enumerate(sequence)
                               if type(layer) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv1d,
                                                  nn.ConvTranspose1d, nn.Conv3d, nn.ConvTranspose3d]])


try:
    # older version <=v0.2.1
    from segmentation_models_pytorch.fpn.decoder import FPNBlock, MergeBlock, FPNDecoder
except:
    # newer version >=v0.3.0
    from segmentation_models_pytorch.decoders.fpn.decoder import FPNBlock, MergeBlock, Conv3x3GNReLU, FPNDecoder
from segmentation_models_pytorch.base import SegmentationModel, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.base.initialization import initialize_head

class Conv3x3GNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, upsample=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ]
        if upsample:
            layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        super().__init__(*layers)


class Custom_FPN(SegmentationModel,MultiTaskModel):
    """FPN_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_depth: number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
        decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
        decoder_merge_policy: determines how to merge outputs inside FPN.
            One of [``add``, ``cat``]
        decoder_dropout: spatial dropout rate in range (0, 1).
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation (str, callable): activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax2d``, callable, None]
        upsampling: optional, final upsampling factor
            (default is 4 to preserve input -> output spatial shape identity)
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_segmentation_head_channels: int = 64,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 2,
        aux_params: Optional[dict] = None,
        checkpoint_encoder_features: bool = False,
        classes_grouping: Optional[list] = None,

    ):
        super().__init__()
        
        # (移除了调试打印)
        print("in_channels FPN", in_channels)

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head_list = nn.ModuleList()
        self.classes_grouping = classes_grouping
        if not self.classes_grouping:
            self.segmentation_head = SegmentationHead(
                in_channels=self.decoder.out_channels,
                segmentation_channels=decoder_segmentation_head_channels,
                out_channels=classes,
                activation=activation,
                kernel_size=3,
                upsampling=upsampling,
            )
        else:
            self.segmentation_head = nn.Sequential(nn.Identity())
            for group_members in self.classes_grouping:
                num_group_members = len(group_members) if type(group_members) in [list,tuple] else group_members
                self.segmentation_head_list.append(SegmentationHead(
                    in_channels=self.decoder.out_channels,
                    segmentation_channels=decoder_segmentation_head_channels,
                    out_channels=num_group_members,
                    activation=activation,
                    kernel_size=3,
                    upsampling=upsampling,
                ))

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        
        self.checkpoint_encoder_features = checkpoint_encoder_features
        self.name = "fpn-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        super().initialize()

        for seg_head in self.segmentation_head_list:
            initialize_head(seg_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)

        if self.checkpoint_encoder_features and any([f.requires_grad for f in features]):
            decoder_output = torch.utils.checkpoint.checkpoint(self.decoder, *features)
        else:
            decoder_output = self.decoder(*features)
        print(type(decoder_output))
        if isinstance(decoder_output, (list, tuple)):
            print(f"[DEBUG] decoder_output is a {type(decoder_output)}, length = {len(decoder_output)}")

        if self.classes_grouping is not None:
            assert len(self.segmentation_head_list) == len(self.classes_grouping)
            # concatenate all segmentation heads and permute them back to original order based on self.classes_grouping
            masks = torch.cat([head(decoder_output) for head in self.segmentation_head_list], dim=1)
            new_idx = list(np.concatenate(self.classes_grouping))
            old_idx = [new_idx.index(i) for i in range(len(new_idx))]
            masks = masks[:,old_idx]
        else:
            masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters shared by all tasks.
        Returns
        -------
        """
        # encode and decode parameters are shared
        encoder_decoder_params = [self.encoder.parameters(), self.decoder.parameters()]

        if self.classes_grouping is None:
            # all parameters except for the last convolution layer are shared
            last_conv_layer_idx = _find_last_conv_index(self.segmentation_head)

            # add all parameters except for the last convolution layer to shared parameters
            encoder_decoder_params.append(self.segmentation_head[:last_conv_layer_idx].parameters())

        return itertools.chain(*encoder_decoder_params)

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters specific to each task.
        Returns
        -------
        """
        if self.classes_grouping is None:
            # only the last convolution/linear layer in segmentation head is task specific
            # all parameters except for the last convolution layer are shared
            last_conv_layer_idx = _find_last_conv_index(self.segmentation_head)

            return itertools.chain(*[h.parameters() for h in self.segmentation_head[last_conv_layer_idx:]])
        else:
            # each segmentation head is task specific
            return itertools.chain(*[h.parameters() for h in self.segmentation_head_list])



    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        """Parameters of the last shared layer.
        Returns
        -------
        """
        if self.classes_grouping is None:
            # all parameters except for the last convolution layer are shared
            last_conv_layer_idx = _find_last_conv_index(self.segmentation_head)

            # last shared parameters ar all in the segmentation head before last convolution layer
            return itertools.chain(*[h.parameters() for h in self.segmentation_head[:last_conv_layer_idx]])
        else:
            # all decoder layers are last shared parameters
            return self.decoder.parameters()

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels,  segmentation_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d_1 = Conv3x3GNReLU(in_channels, segmentation_channels, upsample=False)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        conv2d_2 = nn.Conv2d(segmentation_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = Activation(activation)

        super().__init__(conv2d_1, upsampling, conv2d_2, activation)


def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = torch.tensor(mean)
        mean = mean.reshape([1, -1, 1, 1])
        x = x - mean.to(x.device)

    if std is not None:
        std = torch.tensor(std)
        std = std.reshape([1, -1, 1, 1])
        x = x / std.to(x.device)

    return x


# ==============================================================================
# 3. 计算 FLOPs 和 Params 的脚本
# (已修改为测试 4 通道输入)
# ==============================================================================

def get_model_stats(model_name: str, num_classes_list: list, use_custom_fpn: bool, backbone: str, input_size: tuple, fpn_init_args: dict):
    """
    实例化一个 FPN 模型并计算其 FLOPs 和 Params。
    
    Args:
        model_name (str): 用于在表格中显示的配置名称。
        num_classes_list (list): 传递给 FPN 的 num_classes 参数。
        use_custom_fpn (bool): 传递给 FPN 的 use_custom_fpn 参数。
        backbone (str): 要使用的骨干网络名称。
        input_size (tuple): (C, H, W) 格式的输入尺寸。
        fpn_init_args (dict): 传递给 FPN 构造函数的额外参数。
    
    Returns:
        dict: 包含统计信息的结果字典。
    """
    print(f"\n--- Testing: {model_name} (Custom FPN: {use_custom_fpn}) ---")
    
    # 1. 实例化模型
    #    我们设置 pretrained=False 以避免下载权重
    model = FPN(
        num_classes=num_classes_list,
        backbone=backbone,
        use_custom_fpn=use_custom_fpn,
        pretrained=False,
        **fpn_init_args # 传递 'use_depth', 'use_only_depth', 'fpn_args'
    )
    model.eval() # 设置为评估模式

    # 2. 创建一个虚拟输入
    #    输入尺寸 (Batch=1, C, H, W)
    dummy_input = torch.randn(1, *input_size)

    # 3. 使用 thop.profile 计算 FLOPs 和 Params
    flops, params = profile(
        model,
        inputs=(dummy_input, ),
        custom_ops={
            nn.UpsamplingBilinear2d: lambda m, i, o: (0, 0),
            F.interpolate: lambda *args, **kwargs: (0, 0),
        },
        verbose=False
    )


    return {
        "Config": model_name,
        "Custom FPN": use_custom_fpn,
        "FLOPs (G)": flops / 1e9,
        "Params (M)": params / 1e6
    }

def main():
    # --- 在这里定义您的配置 ---
    # `convnext_b` 是 smp/timm 中的正确名称
    BACKBONE = 'tu-convnext_base' 
    
    # --- 修改：测试 4 通道输入 ---
    INPUT_SIZE = (4, 768, 768) 
    
    # --- 修改：为 4 通道输入配置 FPN 构造函数 ---
    # 这将使 FPN 的 in_channels=4，并正确设置4通道的均值/标准差
    FPN_INIT_ARGS = {
        'use_depth': True,
        'use_only_depth': False, # 这将触发 .append 逻辑
        'fpn_args': {
            'depth_mean': 0.0, # 第4通道（Depth）的均值
            'depth_std': 1.0   # 第4通道（Depth）的标准差
        }
    }

    # 定义我们要对比的 "k" 值和它们对应的输出通道数
    configurations = {
        "deterministic": [2],
        "CalmPredictor(k=1)": [3],
        "CalmPredictor(k=2)": [4 * 2],
        "CalmPredictor(k=3)": [4 * 3],
        "CalmPredictor(k=4)": [4 * 4],
    }
    
    results = []
    
    # --- 循环测试两种 FPN (默认 smp.FPN 和 Custom_FPN) ---
    for use_custom in [False, True]:
        for name, num_classes in configurations.items():
            try:
                stats = get_model_stats(
                    model_name=name,
                    num_classes_list=num_classes,
                    use_custom_fpn=use_custom,
                    backbone=BACKBONE,
                    input_size=INPUT_SIZE,
                    # 传递 FPN_INIT_ARGS (不再需要 deepcopy，因为 FPN.__init__ 已修复)
                    fpn_init_args=FPN_INIT_ARGS 
                )
                results.append(stats)
            except Exception as e:
                print(f"Error processing {name} (Custom FPN: {use_custom}): {e}")
                # 打印更详细的堆栈跟踪以进行调试
                import traceback
                traceback.print_exc()

    # --- 打印结果 ---
    if not results:
        print("\nNo results to display. Did the models fail to build?")
        print("Please ensure 'convnext_b' is a valid backbone for segmentation-models-pytorch.")
        return

    df = pd.DataFrame(results)
    
    # 按 Custom FPN 分组打印
    for custom_fpn_status, group in df.groupby('Custom FPN'):
        print("\n\n" + "="*50)
        if custom_fpn_status:
            print(f"    Results for {BACKBONE} (use_custom_fpn = True)")
        else:
            print(f"    Results for {BACKBONE} (use_custom_fpn = False)")
        print("="*50)
        print(f"Input Size: {INPUT_SIZE}")
        print(group[['Config', 'FLOPs (G)', 'Params (M)']].to_string(index=False, float_format="%.2f"))

if __name__ == "__main__":
    main()