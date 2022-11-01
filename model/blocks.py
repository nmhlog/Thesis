from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from spconv.pytorch import functional as Fsp
from spconv.pytorch import ops 

from spconv.pytorch.modules import SparseModule
from torch import nn


class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


# current 1x1 conv in spconv2x has a bug. It will be removed after the bug is fixed
class Custom1x1Subm3d(spconv.SparseConv3d):

    def forward(self, input):
        features = torch.mm(input.features, self.weight.view(self.out_channels, self.in_channels).T)
        if self.bias is not None:
            features += self.bias
        out_tensor = spconv.SparseConvTensor(features, input.indices, input.spatial_shape,
                                             input.batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


class ResidualBlock(SparseModule):

    def __init__(self, in_channels, out_channels, norm_fn,padding = 1,dilation=1, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, bias=False))

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
                indice_key=indice_key), norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                bias=False,
                indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape,
                                           input.batch_size)
        output = self.conv_branch(input)
        out_feats = output.features + self.i_branch(identity).features
        output = output.replace_feature(out_feats)

        return output


class UNET(nn.Module):

    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {
            'block{}'.format(i):
            block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            self.u = UNET(
                nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key='spconv{}'.format(indice_key_id)))

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):

        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape,
                                           output.batch_size)
        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            out_feats = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(out_feats)
            output = self.blocks_tail(output)
        return output

   
class ASPP(SparseModule):
    def __init__(self,in_channels, out_channels, norm_fn, block_reps=2, indice_key=6, rate = [6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block6 = ResidualBlock( in_channels, out_channels, norm_fn, padding=rate[0], dilation=rate[0], indice_key='bb_subm{}'.format(indice_key))
        self.aspp_block12 = ResidualBlock( in_channels, out_channels, norm_fn, padding=rate[1], dilation=rate[1], indice_key='bb_subm{}'.format(indice_key)) 
        self.aspp_block18 = ResidualBlock( in_channels, out_channels, norm_fn, padding=rate[2], dilation=rate[2], indice_key='bb_subm{}'.format(indice_key))
        self.conv_1x1 = Custom1x1Subm3d(len(rate) * out_channels, out_channels, 1, indice_key='bb_subm{}'.format(indice_key))


    def forward(self, input):
        x1 = self.aspp_block6(input)
        x2 = self.aspp_block12(input)
        x3 = self.aspp_block18(input)
        
        out_feats = torch.cat((x1.features,x2.features,x3.features), dim=1)
        x3 = x3.replace_feature(out_feats)
        out = self.conv_1x1(x3)
        return out
    
from spconv.core import AlgoHint, ConvAlgo
from spconv.pytorch import functional as Fsp
from spconv.pytorch import ops 
class SparseGlobalMaxPool(SparseModule):
    def forward(self, input):
        assert isinstance(input, spconv.SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        ndim = len(spatial_shape)
        ksize = spatial_shape

        out_spatial_shape = ops.get_conv_output_size(
            spatial_shape, ksize, [1] * ndim, [0] * ndim,[1] * ndim)
        """
                     indices: torch.Tensor,
                     batch_size: int,
                     spatial_shape: List[int],
                     algo: ConvAlgo,
                     ksize: List[int],
                     stride: List[int],
                     padding: List[int],
                     dilation: List[int],
                     out_padding: List[int],
                     subm: bool = False,
                     transpose: bool = False,
                     num_out_act_bound: int = -1):
        """
        out_buff = ops.get_indice_pairs_implicit_gemm(indices, batch_size, \
                                                      spatial_shape,ConvAlgo.MaskImplicitGemm, \
                                                      ksize, [1] * ndim ,[0] * ndim, [1]* ndim, [0]* ndim, False)
        out_inds, indice_num_per_loc, pair_fwd, pair_bwd,\
        pair_mask_fwd_splits, pair_mask_bwd_splits,\
        mask_argsort_fwd_splits, mask_argsort_bwd_splits, masks = out_buff

        out_features = Fsp.indice_avgpool_implicit_gemm (features, pair_fwd.to(device),\
                                                         pair_bwd.to(device),\
                                                         out_inds.shape[0],True)
        out_tensor = spconv.SparseConvTensor(out_features, out_inds,
                                             out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor
    
class AttentionBlock(SparseModule):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients, norm_fn,indice_key):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = spconv.SparseSequential(
            spconv.SubMConv3d(F_g, n_coefficients, kernel_size=1, indice_key='attn_subm{}'.format(indice_key)),
            norm_fn(n_coefficients)
            # nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = spconv.SparseSequential(
            spconv.SubMConv3d(F_l, n_coefficients, kernel_size=1, indice_key='attn_subm{}'.format(indice_key)),
            norm_fn(n_coefficients)
            # nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(n_coefficients)
        )

        self.psi = spconv.SparseSequential(
            
            spconv.SubMConv3d(n_coefficients, 1, kernel_size=1, indice_key='attn_subm{}'.format(indice_key)),
            norm_fn(1),
            # nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = spconv.SparseSequential(nn.ReLU(inplace=True))
 
    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        x1 = x1.replace_feature(g1.features + x1.features)
        psi = self.relu(x1)
        psi = self.psi(psi)
        out = skip_connection.features * psi.features
        psi = psi.replace_feature(out)
        return psi


class ASPP_V2(SparseModule):
    def __init__(self,in_channels, out_channels, norm_fn,indice_key=6, rate = [1, 6, 12, 18]):
        super(ASPP_V2, self).__init__()

        self.atrous_block1 = Custom1x1Subm3d(in_channels, out_channels, kernel_size=1, indice_key=indice_key,bias=False)
        self.atrous_block6 = ResidualBlock( in_channels, out_channels, norm_fn, padding=rate[1], dilation=rate[1], indice_key='bb_subm{}'.format(indice_key))
        self.atrous_block12 = ResidualBlock( in_channels, out_channels, norm_fn, padding=rate[2], dilation=rate[2], indice_key='bb_subm{}'.format(indice_key))        
        self.atrous_block18 = ResidualBlock( in_channels, out_channels, norm_fn, padding=rate[3], dilation=rate[3], indice_key='bb_subm{}'.format(indice_key))

        self.conv_1x1 = Custom1x1Subm3d(len(rate) * out_channels, out_channels, kernel_size=1, indice_key=indice_key)
 
    def forward(self, x):

        # spconv.SparseInverseConv3d(nPlanes[6], nPlanes[5], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(6))
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        out_feats = torch.cat((atrous_block6.features, atrous_block12.features,atrous_block18.features,atrous_block1.features), dim=1)
        x3 = atrous_block18.replace_feature(out_feats)
        out = self.conv_1x1(x3)
        return out
    
class UNET_ATTN_ASPP(SparseModule):

    def __init__(self, nPlanes, norm_fn, block_reps):
        super().__init__()
        
        self.attn0 = AttentionBlock(nPlanes[0], nPlanes[0], int(nPlanes[0]/2),norm_fn,0)
        self.attn1 = AttentionBlock(nPlanes[1], nPlanes[1], int(nPlanes[1]/2),norm_fn,1)
        self.attn2 = AttentionBlock(nPlanes[2], nPlanes[2], int(nPlanes[2]/2),norm_fn,2)
        self.attn3 = AttentionBlock(nPlanes[3], nPlanes[3], int(nPlanes[3]/2),norm_fn,3)
        self.attn4 = AttentionBlock(nPlanes[4], nPlanes[4], int(nPlanes[4]/2),norm_fn,4)
        self.attn5 = AttentionBlock(nPlanes[5], nPlanes[5], int(nPlanes[5]/2),norm_fn,5)
        
        self.block0 = self._make_layers(nPlanes[0], nPlanes[0], block_reps, norm_fn, indice_key=0)
        self.conv1 = spconv.SparseSequential(
            norm_fn(nPlanes[0]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(1))
        )
        self.block1 = self._make_layers(nPlanes[1], nPlanes[1], block_reps, norm_fn, indice_key=1)

        self.conv2 = spconv.SparseSequential(
            norm_fn(nPlanes[1]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[1], nPlanes[2], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(2))
        )
        self.block2 = self._make_layers(nPlanes[2], nPlanes[2], block_reps, norm_fn, indice_key=2)

        self.conv3 = spconv.SparseSequential(
            norm_fn(nPlanes[2]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[2], nPlanes[3], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(3))
        )
        self.block3 = self._make_layers(nPlanes[3], nPlanes[3], block_reps, norm_fn, indice_key=3)


        self.conv4 = spconv.SparseSequential(
            norm_fn(nPlanes[3]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[3], nPlanes[4], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(4))
        )
        self.block4 = self._make_layers(nPlanes[4], nPlanes[4], block_reps, norm_fn, indice_key=4)

        self.conv5 = spconv.SparseSequential(
            norm_fn(nPlanes[4]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[4], nPlanes[5], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(5))
        )
        self.block5 = self._make_layers(nPlanes[5], nPlanes[5], block_reps, norm_fn, indice_key=5)

        self.conv6 = spconv.SparseSequential(
            norm_fn(nPlanes[5]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[5], nPlanes[6], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(6))
        )
        self.aspp = ASPP_V2(nPlanes[6], nPlanes[6],  norm_fn, indice_key=6)
        # self.block6 = self._make_layers(nPlanes[6], nPlanes[6], block_reps, norm_fn, indice_key=6)

        self.deconv6 = spconv.SparseSequential(
            norm_fn(nPlanes[6]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[6], nPlanes[5], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(6))
        )

        self.deblock5 = self._make_layers(nPlanes[5] * 2, nPlanes[5], block_reps, norm_fn, indice_key=5)
        self.deconv5 = spconv.SparseSequential(
            norm_fn(nPlanes[5]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[5], nPlanes[4], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(5))
        )

        self.deblock4 = self._make_layers(nPlanes[4] * 2, nPlanes[4], block_reps, norm_fn, indice_key=4)
        self.deconv4 = spconv.SparseSequential(
            norm_fn(nPlanes[4]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[4], nPlanes[3], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(4))
        )

        self.deblock3 = self._make_layers(nPlanes[3] * 2, nPlanes[3], block_reps, norm_fn, indice_key=3)
        self.deconv3 = spconv.SparseSequential(
            norm_fn(nPlanes[3]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[3], nPlanes[2], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(3))
        )

        self.deblock2 = self._make_layers(nPlanes[2] * 2, nPlanes[2], block_reps, norm_fn, indice_key=2)
        self.deconv2 = spconv.SparseSequential(
            norm_fn(nPlanes[2]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[2], nPlanes[1], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(2))
        )

        self.deblock1 = self._make_layers(nPlanes[1] * 2, nPlanes[1], block_reps, norm_fn, indice_key=1)
        self.deconv1 = spconv.SparseSequential(
            norm_fn(nPlanes[1]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(1))
        )

        self.deblock0 = self._make_layers(nPlanes[0] * 2, nPlanes[0], block_reps, norm_fn, indice_key=0)

        
    def _make_layers(self, inplanes, planes, block_reps, norm_fn, indice_key=0):
        blocks = [ResidualBlock(inplanes, planes, norm_fn, indice_key='bb_subm{}'.format(indice_key))]
        for i in range(block_reps - 1):
            blocks.append(ResidualBlock(planes, planes, norm_fn, indice_key='bb_subm{}'.format(indice_key)))
        return spconv.SparseSequential(*blocks)

    def forward(self, x):
        out0 = self.block0(x)
        
        out1 = self.conv1(out0)
        out1 = self.block1(out1)

        out2 = self.conv2(out1)
        out2 = self.block2(out2)

        out3 = self.conv3(out2)
        out3 = self.block3(out3)

        out4 = self.conv4(out3)
        out4 = self.block4(out4)

        out5 = self.conv5(out4)
        out5 = self.block5(out5)

        out6 = self.conv6(out5)
        out6 = self.aspp(out6)
        
        
        d_out5 = self.deconv6(out6)
        attn5 = self.attn5(d_out5,out5)
        out_feats = torch.cat((attn5.features, d_out5.features), dim=1)
        d_out5 = d_out5.replace_feature(out_feats)
        d_out5 = self.deblock5(d_out5)

        d_out4 = self.deconv5(d_out5)
        attn4 = self.attn4(d_out4,out4)
        out_feats = torch.cat((attn4.features, d_out4.features), dim=1)
        d_out4 = d_out4.replace_feature(out_feats)
        d_out4 = self.deblock4(d_out4)

        d_out3 = self.deconv4(d_out4)
        attn3 = self.attn3(d_out3,out3)
        out_feats = torch.cat((attn3.features, d_out3.features), dim=1)
        d_out3 = d_out3.replace_feature(out_feats)
        d_out3 = self.deblock3(d_out3)

        d_out2 = self.deconv3(d_out3)
        attn2 = self.attn2(d_out2,out2)
        out_feats = torch.cat((attn2.features, d_out2.features), dim=1)
        d_out2 = d_out2.replace_feature(out_feats)
        d_out2 = self.deblock2(d_out2)

        d_out1 = self.deconv2(d_out2)
        attn1 = self.attn1(d_out1,out1)
        out_feats = torch.cat((attn1.features, d_out1.features), dim=1)
        d_out1 = d_out1.replace_feature(out_feats)
        d_out1 = self.deblock1(d_out1)

        d_out0 = self.deconv1(d_out1)
        attn0 = self.attn0(d_out0,out0)
        out_feats = torch.cat((attn0.features, d_out0.features), dim=1)
        d_out0 = d_out0.replace_feature(out_feats)
        d_out0 = self.deblock0(d_out0)

        return d_out0
class UNET_ASPPv2(SparseModule):

    def __init__(self, nPlanes, norm_fn, block_reps):
        super().__init__()
        
        self.block0 = self._make_layers(nPlanes[0], nPlanes[0], block_reps, norm_fn, indice_key=0)
        
        self.conv1 = spconv.SparseSequential(
            norm_fn(nPlanes[0]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(1))
        )
        self.block1 = self._make_layers(nPlanes[1], nPlanes[1], block_reps, norm_fn, indice_key=1)

        self.conv2 = spconv.SparseSequential(
            norm_fn(nPlanes[1]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[1], nPlanes[2], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(2))
        )
        self.block2 = self._make_layers(nPlanes[2], nPlanes[2], block_reps, norm_fn, indice_key=2)

        self.conv3 = spconv.SparseSequential(
            norm_fn(nPlanes[2]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[2], nPlanes[3], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(3))
        )
        self.block3 = self._make_layers(nPlanes[3], nPlanes[3], block_reps, norm_fn, indice_key=3)


        self.conv4 = spconv.SparseSequential(
            norm_fn(nPlanes[3]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[3], nPlanes[4], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(4))
        )
        self.block4 = self._make_layers(nPlanes[4], nPlanes[4], block_reps, norm_fn, indice_key=4)

        self.conv5 = spconv.SparseSequential(
            norm_fn(nPlanes[4]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[4], nPlanes[5], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(5))
        )
        self.block5 = self._make_layers(nPlanes[5], nPlanes[5], block_reps, norm_fn, indice_key=5)

        self.conv6 = spconv.SparseSequential(
            norm_fn(nPlanes[5]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[5], nPlanes[6], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(6))
        )
        self.aspp = ASPP_V2(nPlanes[6], nPlanes[6],  norm_fn, indice_key=6)
        # self.block6 = self._make_layers(nPlanes[6], nPlanes[6], block_reps, norm_fn, indice_key=6)

        self.deconv6 = spconv.SparseSequential(
            norm_fn(nPlanes[6]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[6], nPlanes[5], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(6))
        )

        self.deblock5 = self._make_layers(nPlanes[5] * 2, nPlanes[5], block_reps, norm_fn, indice_key=5)
        self.deconv5 = spconv.SparseSequential(
            norm_fn(nPlanes[5]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[5], nPlanes[4], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(5))
        )

        self.deblock4 = self._make_layers(nPlanes[4] * 2, nPlanes[4], block_reps, norm_fn, indice_key=4)
        self.deconv4 = spconv.SparseSequential(
            norm_fn(nPlanes[4]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[4], nPlanes[3], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(4))
        )

        self.deblock3 = self._make_layers(nPlanes[3] * 2, nPlanes[3], block_reps, norm_fn, indice_key=3)
        self.deconv3 = spconv.SparseSequential(
            norm_fn(nPlanes[3]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[3], nPlanes[2], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(3))
        )

        self.deblock2 = self._make_layers(nPlanes[2] * 2, nPlanes[2], block_reps, norm_fn, indice_key=2)
        self.deconv2 = spconv.SparseSequential(
            norm_fn(nPlanes[2]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[2], nPlanes[1], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(2))
        )

        self.deblock1 = self._make_layers(nPlanes[1] * 2, nPlanes[1], block_reps, norm_fn, indice_key=1)
        self.deconv1 = spconv.SparseSequential(
            norm_fn(nPlanes[1]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(1))
        )

        self.deblock0 = self._make_layers(nPlanes[0] * 2, nPlanes[0], block_reps, norm_fn, indice_key=0)

        
    def _make_layers(self, inplanes, planes, block_reps, norm_fn, indice_key=0):
        blocks = [ResidualBlock(inplanes, planes, norm_fn, indice_key='bb_subm{}'.format(indice_key))]
        for i in range(block_reps - 1):
            blocks.append(ResidualBlock(planes, planes, norm_fn, indice_key='bb_subm{}'.format(indice_key)))
        return spconv.SparseSequential(*blocks)

    def forward(self, x):
        out0 = self.block0(x)
        
        out1 = self.conv1(out0)
        out1 = self.block1(out1)

        out2 = self.conv2(out1)
        out2 = self.block2(out2)

        out3 = self.conv3(out2)
        out3 = self.block3(out3)

        out4 = self.conv4(out3)
        out4 = self.block4(out4)

        out5 = self.conv5(out4)
        out5 = self.block5(out5)

        out6 = self.conv6(out5)
        out6 = self.aspp(out6)

        d_out5 = self.deconv6(out6)
        out_feats = torch.cat((d_out5.features, out5.features), dim=1)
        d_out5 = d_out5.replace_feature(out_feats)
        d_out5 = self.deblock5(d_out5)

        d_out4 = self.deconv5(d_out5)
        out_feats = torch.cat((d_out4.features, out4.features), dim=1)
        d_out4 = d_out4.replace_feature(out_feats)
        d_out4 = self.deblock4(d_out4)

        d_out3 = self.deconv4(d_out4)
        out_feats = torch.cat((d_out3.features, out3.features), dim=1)
        d_out3 = d_out3.replace_feature(out_feats)
        d_out3 = self.deblock3(d_out3)

        d_out2 = self.deconv3(d_out3)
        out_feats = torch.cat((d_out2.features, out2.features), dim=1)
        d_out2 = d_out2.replace_feature(out_feats)
        d_out2 = self.deblock2(d_out2)

        d_out1 = self.deconv2(d_out2)
        out_feats = torch.cat((d_out1.features, out1.features), dim=1)
        d_out1 = d_out1.replace_feature(out_feats)
        d_out1 = self.deblock1(d_out1)

        d_out0 = self.deconv1(d_out1)
        out_feats = torch.cat((d_out0.features, out0.features), dim=1)
        d_out0 = d_out0.replace_feature(out_feats)
        d_out0 = self.deblock0(d_out0)

        return d_out0


class UNET_ASPP(SparseModule):

    def __init__(self, nPlanes, norm_fn, block_reps):
        super().__init__()
        
        self.block0 = self._make_layers(nPlanes[0], nPlanes[0], block_reps, norm_fn, indice_key=0)
        
        self.conv1 = spconv.SparseSequential(
            norm_fn(nPlanes[0]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(1))
        )
        self.block1 = self._make_layers(nPlanes[1], nPlanes[1], block_reps, norm_fn, indice_key=1)

        self.conv2 = spconv.SparseSequential(
            norm_fn(nPlanes[1]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[1], nPlanes[2], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(2))
        )
        self.block2 = self._make_layers(nPlanes[2], nPlanes[2], block_reps, norm_fn, indice_key=2)

        self.conv3 = spconv.SparseSequential(
            norm_fn(nPlanes[2]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[2], nPlanes[3], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(3))
        )
        self.block3 = self._make_layers(nPlanes[3], nPlanes[3], block_reps, norm_fn, indice_key=3)


        self.conv4 = spconv.SparseSequential(
            norm_fn(nPlanes[3]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[3], nPlanes[4], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(4))
        )
        self.block4 = self._make_layers(nPlanes[4], nPlanes[4], block_reps, norm_fn, indice_key=4)

        self.conv5 = spconv.SparseSequential(
            norm_fn(nPlanes[4]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[4], nPlanes[5], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(5))
        )
        self.block5 = self._make_layers(nPlanes[5], nPlanes[5], block_reps, norm_fn, indice_key=5)

        self.conv6 = spconv.SparseSequential(
            norm_fn(nPlanes[5]),
            nn.ReLU(),                                   
            spconv.SparseConv3d(nPlanes[5], nPlanes[6], kernel_size=2, stride=2, bias=False, indice_key='bb_spconv{}'.format(6))
        )
        self.aspp = ASPP(nPlanes[6], nPlanes[6],  norm_fn, indice_key=6)
        # self.block6 = self._make_layers(nPlanes[6], nPlanes[6], block_reps, norm_fn, indice_key=6)

        self.deconv6 = spconv.SparseSequential(
            norm_fn(nPlanes[6]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[6], nPlanes[5], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(6))
        )

        self.deblock5 = self._make_layers(nPlanes[5] * 2, nPlanes[5], block_reps, norm_fn, indice_key=5)
        self.deconv5 = spconv.SparseSequential(
            norm_fn(nPlanes[5]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[5], nPlanes[4], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(5))
        )

        self.deblock4 = self._make_layers(nPlanes[4] * 2, nPlanes[4], block_reps, norm_fn, indice_key=4)
        self.deconv4 = spconv.SparseSequential(
            norm_fn(nPlanes[4]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[4], nPlanes[3], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(4))
        )

        self.deblock3 = self._make_layers(nPlanes[3] * 2, nPlanes[3], block_reps, norm_fn, indice_key=3)
        self.deconv3 = spconv.SparseSequential(
            norm_fn(nPlanes[3]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[3], nPlanes[2], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(3))
        )

        self.deblock2 = self._make_layers(nPlanes[2] * 2, nPlanes[2], block_reps, norm_fn, indice_key=2)
        self.deconv2 = spconv.SparseSequential(
            norm_fn(nPlanes[2]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[2], nPlanes[1], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(2))
        )

        self.deblock1 = self._make_layers(nPlanes[1] * 2, nPlanes[1], block_reps, norm_fn, indice_key=1)
        self.deconv1 = spconv.SparseSequential(
            norm_fn(nPlanes[1]),
            nn.ReLU(),                                             
            spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='bb_spconv{}'.format(1))
        )

        self.deblock0 = self._make_layers(nPlanes[0] * 2, nPlanes[0], block_reps, norm_fn, indice_key=0)

        
    def _make_layers(self, inplanes, planes, block_reps, norm_fn, indice_key=0):
        blocks = [ResidualBlock(inplanes, planes, norm_fn, indice_key='bb_subm{}'.format(indice_key))]
        for i in range(block_reps - 1):
            blocks.append(ResidualBlock(planes, planes, norm_fn, indice_key='bb_subm{}'.format(indice_key)))
        return spconv.SparseSequential(*blocks)

    def forward(self, x):
        out0 = self.block0(x)
        
        out1 = self.conv1(out0)
        out1 = self.block1(out1)

        out2 = self.conv2(out1)
        out2 = self.block2(out2)

        out3 = self.conv3(out2)
        out3 = self.block3(out3)

        out4 = self.conv4(out3)
        out4 = self.block4(out4)

        out5 = self.conv5(out4)
        out5 = self.block5(out5)

        out6 = self.conv6(out5)
        out6 = self.aspp(out6)

        d_out5 = self.deconv6(out6)
        out_feats = torch.cat((d_out5.features, out5.features), dim=1)
        d_out5 = d_out5.replace_feature(out_feats)
        d_out5 = self.deblock5(d_out5)

        d_out4 = self.deconv5(d_out5)
        out_feats = torch.cat((d_out4.features, out4.features), dim=1)
        d_out4 = d_out4.replace_feature(out_feats)
        d_out4 = self.deblock4(d_out4)

        d_out3 = self.deconv4(d_out4)
        out_feats = torch.cat((d_out3.features, out3.features), dim=1)
        d_out3 = d_out3.replace_feature(out_feats)
        d_out3 = self.deblock3(d_out3)

        d_out2 = self.deconv3(d_out3)
        out_feats = torch.cat((d_out2.features, out2.features), dim=1)
        d_out2 = d_out2.replace_feature(out_feats)
        d_out2 = self.deblock2(d_out2)

        d_out1 = self.deconv2(d_out2)
        out_feats = torch.cat((d_out1.features, out1.features), dim=1)
        d_out1 = d_out1.replace_feature(out_feats)
        d_out1 = self.deblock1(d_out1)

        d_out0 = self.deconv1(d_out1)
        out_feats = torch.cat((d_out0.features, out0.features), dim=1)
        d_out0 = d_out0.replace_feature(out_feats)
        d_out0 = self.deblock0(d_out0)

        return d_out0

