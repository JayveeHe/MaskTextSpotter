# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.quantized import QFunctional

class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, in_channels_list, out_channels, top_blocks=None):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        self.inner_blocks_inst = []
        self.layer_blocks_inst = []

        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            for module in [inner_block_module, layer_block_module]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks_inst.append(self._modules[inner_block])
            self.layer_blocks_inst.append(self._modules[layer_block])
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.qfunc = QFunctional()

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        x = [self.quant(a) for a in x]
        # last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        last_inner_blocks_name = self.inner_blocks[-1]
        last_inner = getattr(self, last_inner_blocks_name)(x[-1])
        # last_inner = self.inner_blocks_inst[-1](x[-1])
        results = []
        tmp_last_inner_res = getattr(self, self.layer_blocks[-1])(last_inner)
        # tmp_last_inner_res = self.layer_blocks_inst[-1](last_inner)
        results.append(self.dequant(tmp_last_inner_res))
        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
        # for feature, inner_block, layer_block in zip(
        #         x[:-1][::-1], self.inner_blocks_inst[:-1][::-1], self.layer_blocks_inst[:-1][::-1]
        # ):
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            # feature = self.quant(feature)
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            # Jayvee: before quantized
            # last_inner = inner_lateral + inner_top_down
            # results.insert(0, getattr(self, layer_block)(last_inner))
            # inner_top_down = self.quant(inner_top_down)
            if last_inner.is_quantized:
                last_inner = self.qfunc.add(inner_lateral, inner_top_down)
            else:
                last_inner = inner_lateral + inner_top_down
            tmp_last_inner_res = getattr(self, layer_block)(last_inner)
            results.insert(0, self.dequant(tmp_last_inner_res))

        if self.top_blocks is not None:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def __init__(self):
        super(LastLevelMaxPool, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]
