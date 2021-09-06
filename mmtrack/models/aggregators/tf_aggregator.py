import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule

from ..builder import AGGREGATORS


@AGGREGATORS.register_module()
class TFBlenderAggregator(nn.Module):
    def __init__(self,
                 num_convs=1,
                 channels=256,
                 kernel_size=3,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(TFBlenderAggregator, self).__init__()
        assert num_convs > 0, 'The number of convs must be bigger than 1.'
        self.embed_convs = nn.ModuleList()
        for i in range(num_convs):
            if i == num_convs - 1:
                new_norm_cfg = None
                new_act_cfg = None
            else:
                new_norm_cfg = norm_cfg
                new_act_cfg = act_cfg
            self.embed_convs.append(
                ConvModule(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    norm_cfg=new_norm_cfg,
                    act_cfg=new_act_cfg))

        self.tf_blenders = nn.ModuleList()

        new_norm_cfg = norm_cfg
        new_act_cfg = act_cfg
        self.tf_blenders.append(
             ConvModule(
                 in_channels=channels * 8,
                 out_channels=channels * 4,
                 kernel_size=1,
                 padding=0,
                 norm_cfg=new_norm_cfg,
                 act_cfg=new_act_cfg))
        self.tf_blenders.append(
             ConvModule(
                 in_channels=channels * 4,
                 out_channels=channels * 2,
                 kernel_size=3,
                 padding=1,
                 norm_cfg=new_norm_cfg,
                 act_cfg=new_act_cfg))
        self.tf_blenders.append(
             ConvModule(
                 in_channels=channels * 2,
                 out_channels=channels,
                 kernel_size=1,
                 padding=0,
                 norm_cfg=None,
                 act_cfg=None))

    def forward(self, x, ref_x):
        """Aggregate reference feature maps `ref_x`.

        The aggregation mainly contains two steps:
        1. Computing the cos similarity between `x` and `ref_x`.
        2. Use the normlized (i.e. softmax) cos similarity to weightedly sum
        `ref_x`.

        Args:
            x (Tensor): of shape [1, C, H, W]
            ref_x (Tensor): of shape [N, C, H, W]. N is the number of reference
                feature maps.

        Returns:
            Tensor: The aggregated feature map with shape [1, C, H, W].
        """
        assert len(x.shape) == 4 and len(x) == 1, \
            "Only support 'batch_size == 1' for x"
        x_embed = x
        for embed_conv in self.embed_convs:
            x_embed = embed_conv(x_embed)
        x_embed = x_embed / x_embed.norm(p=2, dim=1, keepdim=True)

        ref_x_embed = ref_x
        for embed_conv in self.embed_convs:
            ref_x_embed = embed_conv(ref_x_embed)
        ref_x_embed = ref_x_embed / ref_x_embed.norm(p=2, dim=1, keepdim=True)

        tf_weight = torch.cat((x_embed.repeat(ref_x_embed.shape[0],1,1,1), \
                               ref_x_embed, \
                               x_embed.repeat(ref_x_embed.shape[0],1,1,1) - ref_x_embed, \
                               x.repeat(ref_x_embed.shape[0],1,1,1), \
                               ref_x, \
                               x.repeat(ref_x_embed.shape[0],1,1,1) - ref_x, \
                               - x_embed.repeat(ref_x_embed.shape[0],1,1,1) + ref_x_embed,  \
                               - x.repeat(ref_x_embed.shape[0],1,1,1) + ref_x \
                              ), dim=1)

        for tf_blend in self.tf_blenders:
            tf_weight = tf_blend(tf_weight)

        ada_weights = tf_weight

        ada_weights = ada_weights.softmax(dim=0)
        agg_x = torch.sum(ref_x * ada_weights, dim=0, keepdim=True)
        return agg_x