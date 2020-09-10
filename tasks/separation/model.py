# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ downstream/model.py ]
#   Synopsis     [ Implementation of downstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformer.model import TransformerConfig, TransformerInitModel, TransformerModel, TransformerSpecPredictionHead
from transformer.mam import fast_position_encoding
import asteroid.filterbanks as fb
from asteroid import torch_utils
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.masknn import TDConvNet


class SeparationConfig(object):
    def __init__(self, config):
        self.n_filters = config['separation']['n_filters']
        self.kernel_size = config['separation']['kernel_size']
        self.stride = config['separation']['stride']
        self.sample_rate = config['separation']['sample_rate']
        self.segment = config['separation']['segment']
        self.n_src = config['separation']['n_src']
        self.fb_name = config['separation']['fb_name']
        self.p_inv = config['separation']['p_inv'] if 'p_inv' in config['separation'] else 'dec'

class ModelConfig(TransformerConfig, SeparationConfig):
    def __init__(self, config):
        TransformerConfig.__init__(self, config['model'])
        SeparationConfig.__init__(self, config['model'])


class ConvTasnet(TransformerInitModel):
    def __init__(self, config, input_dim, output_dim, output_attentions=False, keep_multihead_output=False):
        super().__init__(config, output_attentions)
        self.encoder, self.decoder = fb.make_enc_dec(self.config.fb_name,
                                                     n_filters=self.config.n_filters,
                                                     kernel_size=self.config.kernel_size,
                                                     stride=self.config.stride,
                                                     sample_rate=self.config.sample_rate,
                                                     who_is_pinv=self.config.p_inv)
        self.masker = TDConvNet(in_chan=self.encoder.filterbank.n_feats_out,
                                out_chan=self.encoder.filterbank.n_feats_out,
                                n_src=self.config.n_src)

        self.criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')

    def forward(self, x, head_mask=None, labels=None):
        if len(x.shape) == 2:   # [batch, n_frames]
            x = x.unsqueeze(1)  # [batch, n_channels, n_frames]
        tf_rep = self.encoder(x)    # [batch, n_filters, n_frames]
        est_masks = self.masker(tf_rep)
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        est_sources = torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), x)
        if labels is not None:
            loss = self.criterion(est_sources, labels)
            return loss, est_sources
        return est_sources, est_masks

 
##########################
# Transformer ConvTasnet #
##########################
class TransformerForTasnet(TransformerInitModel):
    def __init__(self, config, input_dim, output_dim, output_attentions=False, keep_multihead_output=False):
        super().__init__(config, output_attentions)
        self.Transformer = TransformerModel(config, input_dim, output_attentions=output_attentions,
                                            keep_multihead_output=keep_multihead_output)
        spec_dim = output_dim if output_dim is not None else input_dim
        self.MaskerHead = TransformerSpecPredictionHead(config, spec_dim * self.config.n_src)
        self.apply(self.init_Transformer_weights)

        self.encoder, self.decoder = fb.make_enc_dec(self.config.fb_name,
                                                     n_filters=self.config.n_filters,
                                                     kernel_size=self.config.kernel_size,
                                                     stride=self.config.stride,
                                                     sample_rate=self.config.sample_rate,
                                                     who_is_pinv=self.config.p_inv)

        self.criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
        if self.config.downsample_type == 'Conv':
            self.down_conv = nn.Conv1d(self.config.n_filters,
                                       self.config.n_filters * self.config.downsample_rate,
                                       kernel_size=129,
                                       padding=64,
                                       groups=16,
                                       stride=self.config.downsample_rate)
            self.up_deconv = nn.ConvTranspose1d(self.config.n_filters * self.config.downsample_rate,
                                                self.config.n_filters,
                                                kernel_size=129,
                                                padding=64,
                                                groups=16,
                                                stride=self.config.downsample_rate)

    def up_sample_frames(self, spec, return_first=False):
        if self.config.downsample_type == 'Conv':
            batch_size, n_src, n_frames, n_filters = spec.shape
            spec = spec.reshape(batch_size*n_src, n_frames, n_filters)
            spec = self.up_deconv(spec.transpose(1, 2)).transpose(1, 2).contiguous()
            return spec.reshape(batch_size, n_src, spec.shape[1], spec.shape[2])

        dr = self.config.downsample_rate
        if len(spec.shape) != 4: 
            spec = spec.unsqueeze(0)
            assert(len(spec.shape) == 4), 'Input should have acoustic feature of shape BxCxTxD'
        # spec shape: [batch_size, n_src, sequence_length // downsample_rate, output_dim * downsample_rate]
        spec_flatten = spec.view(spec.shape[0], spec.shape[1], spec.shape[2]*dr, spec.shape[3]//dr)
        if return_first: return spec_flatten[0]
        return spec_flatten # spec_flatten shape: [batch_size, n_src, sequence_length * downsample_rate, output_dim // downsample_rate]

    def down_sample_frames(self, spec):
        if self.config.downsample_type == 'Conv':
            return self.down_conv(spec.transpose(1, 2)).transpose(1, 2).contiguous()
        dr = self.config.downsample_rate
        left_over = spec.shape[1] % dr
        if left_over != 0: spec = spec[:, :-left_over, :]
        spec_stacked = spec.view(spec.shape[0], spec.shape[1]//dr, spec.shape[2]*dr)
        return spec_stacked

    def process_data(self, spec):
        hidden_size = self.config.hidden_size

        spec_stacked = self.down_sample_frames(spec)

        spec_len = (spec_stacked.sum(dim=-1) != 0).long().sum(dim=-1).tolist()
        batch_size = spec_stacked.shape[0]
        seq_len = spec_stacked.shape[1]
        
        if self.config.pos_enc == 'Conv':
            pos_enc = None
        else:
            pos_enc = fast_position_encoding(seq_len, hidden_size).to(dtype=spec_stacked.dtype, device=spec_stacked.device)
        attn_mask = spec.new_ones((batch_size, seq_len))

        for idx in range(len(spec_stacked)):
            attn_mask[idx, spec_len[idx]:] = 0

        return spec_stacked, pos_enc, attn_mask

    def forward(self, x, head_mask=None, labels=None):
        if len(x.shape) == 2:   # [batch, n_frames]
            x = x.unsqueeze(1)  # [batch, n_channels, n_frames]
        tf_rep = self.encoder(x)    # [batch, n_filters, n_frames]
        tf_rep = tf_rep.transpose(1, 2).contiguous() # [batch, n_frames, n_filters]

        spec, pos_enc, attention_mask = self.process_data(spec=tf_rep)

        transformer_output = self.Transformer(spec, pos_enc, attention_mask,
                                              output_all_encoded_layers=False, head_mask=head_mask)
        if self.output_attentions:
            all_attentions, sequence_output = transformer_output
        else:
            sequence_output = transformer_output
        est_masks, est_state = self.MaskerHead(sequence_output) # [batch, n_frames, n_src * n_filters]
        est_masks = est_masks.reshape(spec.size(0), spec.size(1), self.config.n_src, -1)
        masked_spec = est_masks * spec.unsqueeze(2)

        masked_spec = masked_spec.permute(0, 2, 1, 3).contiguous()   # [batch, n_src, n_frames, n_filters]
        masked_tf_rep = self.up_sample_frames(masked_spec)
        masked_tf_rep = masked_tf_rep.permute(0, 1, 3, 2).contiguous()   # [batch, n_src, n_filters, n_frames]

        est_sources = torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), x)
        if labels is not None:
            loss = self.criterion(est_sources, labels)
            return loss, est_sources
        elif self.output_attentions:
            return all_attentions, est_sources
        return est_sources, est_masks
