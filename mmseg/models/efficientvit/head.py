# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch
import torch.nn as nn

from .nn import (
    ConvLayer,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpSampleLayer,
)

from .utils import list_sum

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@MODELS.register_module()
class SegHead(BaseDecodeHead):
    def __init__(
        self,
        fid_list: list[str],
        in_channels: list[int],
        stride_list: list[int],
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand,
        num_classes: int,
        dropout=0,
        norm="bn2d",
        act_func="hswish",
        merge="add",
        **kwargs
    ):
        super().__init__(
            input_transform=None,
            num_classes=num_classes,
            in_channels=in_channels[0],
            channels=head_width * (final_expand or 1),
        )

        inputs = {}
        for fid, in_channel, stride in zip(fid_list, in_channels, stride_list):
            factor = stride // head_stride
            if factor == 1:
                inputs[fid] = ConvLayer(
                    in_channel, head_width, 1, norm=norm, act_func=None
                )
            else:
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor),
                    ]
                )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "segout": OpSequential(
                [
                    (
                        None
                        if final_expand is None
                        else ConvLayer(
                            head_width,
                            head_width * final_expand,
                            1,
                            norm=norm,
                            act_func=act_func,
                        )
                    ),
                ]
            )
        }

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = None

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [
            op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)
        ]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)

        out = feature_dict["segout"]

        return self.cls_seg(out)
