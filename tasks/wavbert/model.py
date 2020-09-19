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
from transformer.mam import fast_position_encoding, process_wav_MAM_data
import asteroid.filterbanks as fb
from asteroid import torch_utils
from asteroid.losses import PITLossWrapper, singlesrc_neg_sisdr
from asteroid.masknn import TDConvNet


class WaveBertConfig(object):
    def __init__(self, config):
        self.n_filters = config['separation']['n_filters']
        self.kernel_size = config['separation']['kernel_size']
        self.stride = config['separation']['stride']
        self.sample_rate = config['separation']['sample_rate']
        self.segment = config['separation']['segment']
        self.n_src = config['separation']['n_src']
        self.fb_name = config['separation']['fb_name']
        self.p_inv = config['separation']['p_inv'] if 'p_inv' in config['separation'] else 'dec'

        self.mask_proportion = config['transformer']['mask_proportion']
        self.mask_consecutive_min = config['transformer']['mask_consecutive_min'] * self.sample_rate
        self.mask_consecutive_max = config['transformer']['mask_consecutive_max'] * self.sample_rate
        self.mask_allow_overlap = config['transformer']['mask_allow_overlap']
        self.mask_bucket_ratio = config['transformer']['mask_bucket_ratio']
        self.mask_frequency = config['transformer']['mask_frequency']
        self.noise_proportion = config['transformer']['noise_proportion']

class ModelConfig(TransformerConfig, WaveBertConfig):
    def __init__(self, config):
        TransformerConfig.__init__(self, config['model'])
        WaveBertConfig.__init__(self, config['model'])


class ConvTasnetForWaveBert(TransformerInitModel):
    def __init__(self, config, input_dim, output_dim, output_attentions=False, keep_multihead_output=False):
        super().__init__(config, output_attentions)
        self.encoder, self.decoder = fb.make_enc_dec(self.config.fb_name,
                                                     n_filters=self.config.n_filters,
                                                     kernel_size=self.config.kernel_size,
                                                     stride=self.config.stride,
                                                     sample_rate=self.config.sample_rate,
                                                     who_is_pinv=self.config.p_inv)
        # setting n_src=1 in pretraining would lead to size mismatch for
        # self.masker.mask_net.1.weight and self.masker.mask_net.1.bias
        self.masker = TDConvNet(in_chan=self.encoder.filterbank.n_feats_out,
                                out_chan=self.encoder.filterbank.n_feats_out,
                                n_src=1)
        # To load pretrained checkpoint, first change the name of those
        # parameter keys in the state_dict, then load_state_dict(...,
        # strict=False)

        # self.criterion = PITLossWrapper(singlesrc_neg_sisdr, pit_from='pw_pt')
        self.criterion = nn.L1Loss()

    # def forward(self, x, head_mask=None, labels=None):
    def forward(self, x, head_mask=None, wav_label=None, mask_label=None):
        if len(x.shape) == 2:   # [batch, n_frames]
            x = x.unsqueeze(1)  # [batch, n_channels, n_frames]
        tf_rep = self.encoder(x)    # [batch, n_filters, n_frames]
        est_masks = self.masker(tf_rep) # [batch, 1, n_filters, n_frames]
        # masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        masked_tf_rep = est_masks
        est_sources = torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), x).squeeze()
        if wav_label is not None and mask_label is not None:
            assert mask_label.sum() > 0, 'Without any masking, loss might go NaN. Modify your data preprocessing (utility/mam.py)'
            loss = self.criterion(est_sources.masked_select(mask_label), wav_label.masked_select(mask_label))
            return loss, est_sources
        return est_sources, est_masks

 
##########################
# Transformer ConvTasnet #
##########################
class TransformerForWaveBert(TransformerInitModel):
    def __init__(self, config, input_dim, output_dim, output_attentions=False, keep_multihead_output=False):
        super().__init__(config, output_attentions)
        self.Transformer = TransformerModel(config, input_dim, output_attentions=output_attentions,
                                            keep_multihead_output=keep_multihead_output)
        spec_dim = output_dim if output_dim is not None else input_dim
        self.SpecHead = TransformerSpecPredictionHead(config, spec_dim)
        self.apply(self.init_Transformer_weights)

        self.encoder, self.decoder = fb.make_enc_dec(self.config.fb_name,
                                                     n_filters=self.config.n_filters,
                                                     kernel_size=self.config.kernel_size,
                                                     stride=self.config.stride,
                                                     sample_rate=self.config.sample_rate,
                                                     who_is_pinv=self.config.p_inv)

        self.criterion = nn.L1Loss()
        # self.criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
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
            return self.up_deconv(spec.transpose(1, 2)).transpose(1, 2).contiguous()

        dr = self.config.downsample_rate
        if len(spec.shape) != 3: 
            spec = spec.unsqueeze(0)
            assert(len(spec.shape) == 3), 'Input should have acoustic feature of shape BxTxD'
        # spec shape: [batch_size, sequence_length // downsample_rate, output_dim * downsample_rate]
        spec_flatten = spec.view(spec.shape[0], spec.shape[1]*dr, spec.shape[2]//dr)
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

    def forward(self, x, head_mask=None, wav_label=None, mask_label=None):
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
            sequence_output = transformer_output # [batch, n_frames, n_filters]
        pred_spec, pred_state = self.SpecHead(sequence_output) # [batch, n_frames, n_filters]
        pred_tf_rep = self.up_sample_frames(pred_spec)  # [batch, n_frames, n_filters]

        pred_tf_rep = pred_tf_rep.transpose(1,2).contiguous()   # [batch, n_filters, n_frames]
        pred_wav = torch_utils.pad_x_to_y(self.decoder(pred_tf_rep), x).squeeze()
        if wav_label is not None and mask_label is not None:
            assert mask_label.sum() > 0, 'Without any masking, loss might go NaN. Modify your data preprocessing (utility/mam.py)'
            loss = self.criterion(pred_wav.masked_select(mask_label), wav_label.masked_select(mask_label))
            return loss, pred_wav
        elif self.output_attentions:
            return all_attentions, pred_wav
        return pred_wav, pred_state