# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ transformer/runner.py ]
#   Synopsis     [ runner for pre-training the transformer models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import torch
import torchvision
import random
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from tasks.wav2vec_v2.model import ModelConfig, ConvTasnetForWav2vecV2
from transformer.optimization import BertAdam, WarmupLinearSchedule
from transformer.mam import fast_position_encoding, process_wav_MAM_data
from utility.audio import plot_spectrogram_to_numpy, plot_waveform_to_numpy, plot_attention_to_numpy


##########
# RUNNER #
##########
class Runner():
    ''' Handler for complete pre-training progress of upstream models '''
    def __init__(self, args, config, dataloader, ckpdir):
        
        self.device = torch.device('cuda') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): print('[Runner] - CUDA is available!')
        self.model_kept = []
        self.global_step = 1
        self.log = SummaryWriter(ckpdir)

        self.args = args
        self.config = config
        self.dataloader = dataloader
        self.ckpdir = ckpdir

        # optimizer
        self.learning_rate = float(config['optimizer']['learning_rate'])
        self.warmup_proportion = config['optimizer']['warmup_proportion']
        self.gradient_accumulation_steps = config['optimizer']['gradient_accumulation_steps']
        self.gradient_clipping = config['optimizer']['gradient_clipping']

        # Training details
        self.apex = config['runner']['apex']
        self.total_steps = config['runner']['total_steps']
        self.log_step = config['runner']['log_step']
        self.save_step = config['runner']['save_step']
        self.max_keep = config['runner']['max_keep']

        # model
        self.transformer_config = config['model']['transformer']
        self.input_dim = self.transformer_config['input_dim']
        self.output_dim = None # output dim is the same as input dim if not using duo features


    def set_model(self):
        print('[Runner] - Initializing Transformer model...')
        
        # build the Transformer model with speech prediction head
        model_config = ModelConfig(self.config)
        self.dr = model_config.downsample_rate
        
        self.model = ConvTasnetForWav2vecV2(model_config, self.input_dim, self.output_dim).to(self.device)
        # if self.config['model']['tasnet'] == 'ConvTasnet':
            # self.model = ConvTasnetForWaveBert(model_config, self.input_dim, self.output_dim).to(self.device)
        # elif self.config['model']['tasnet'] == 'Transformer':
            # self.model = TransformerForWaveBert(model_config, self.input_dim, self.output_dim).to(self.device)
        self.model.train()

        if self.args.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[Runner] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[Runner] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # Setup optimizer
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        if self.apex:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                    lr=self.learning_rate,
                                    bias_correction=False,
                                    max_grad_norm=1.0)
            if self.config['optimizer']['loss_scale'] == 0:
                self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.config['optimizer']['loss_scale'])
            self.warmup_linear = WarmupLinearSchedule(warmup=self.warmup_proportion,
                                                      t_total=self.total_steps)
        else:
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=self.learning_rate,
                                      warmup=self.warmup_proportion,
                                      t_total=self.total_steps)


    def save_model(self, name='states', to_path=None):
        all_states = {
            'model': self.model.state_dict() if not self.args.multi_gpu else self.model.module.state_dict(),
            # 'encoder': self.model.encoder.state_dict() if not self.args.multi_gpu else self.model.module.encoder.state_dict(),
            # 'decoder': self.model.decoder.state_dict() if not self.args.multi_gpu else self.model.module.decoder.state_dict(),
            # 'SpecHead': self.model.SpecHead.state_dict() if not self.args.multi_gpu else self.model.module.SpecHead.state_dict(),
            # 'Transformer': self.model.Transformer.state_dict() if not self.args.multi_gpu else self.model.module.Transformer.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Global_step': self.global_step,
            'Settings': {
                'Config': self.config,
                'Paras': self.args,
            },
        }

        if to_path is None:
            new_model_path = '{}/{}-{}.ckpt'.format(self.ckpdir, name, self.global_step)
        else:
            new_model_path = to_path

        torch.save(all_states, new_model_path)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)


    def process_data(self, batch):
        """Process training data for the masked acoustic model"""
        with torch.no_grad():
            
            assert(len(batch) == 2), 'dataloader should return (mixture, sources)'
            # Unpack and Hack bucket: Bucketing should cause acoustic feature to have shape 1xBxTxD'
            mixture, sources = batch
            assert self.config['dataloader']['task'] == 'enh_single', 'dataloader task should be "enh_single"'
            noisy_wav = mixture
            clean_wav = sources[:, 0]
            if random.random() < 0.8:
                wav_stacked = clean_wav.to(device=self.device)
            else:
                wav_stacked = noisy_wav.to(device=self.device)
            # noise_wav = noisy_wav - clean_wav
            # batch_is_valid, wav_masked, mask_label, wav_stacked = process_wav_MAM_data(clean_wav=clean_wav.clone(),
                                                                                       # noisy_wav=noisy_wav.clone(),
                                                                                       # noise_wav=noise_wav.clone(),
                                                                                       # config=self.model.config)
            # wav_masked = wav_masked.to(device=self.device)
            # mask_label = mask_label.to(device=self.device)
            # wav_stacked = wav_stacked.to(device=self.device)

        return wav_stacked # (x, pos_enc, mask_label, attention_mask. y)


    def train(self):
        ''' Self-Supervised Pre-Training of Transformer Model'''

        pbar = tqdm(total=self.total_steps)
        while self.global_step <= self.total_steps:

            kappa = self.model.kappa if not self.args.multi_gpu else self.model.module.kappa
            progress = tqdm(self.dataloader, desc="Iteration", postfix={'kappa': kappa})

            step = 0
            loss_val = 0
            for batch in progress:
                try:
                    wav_stacked = self.process_data(batch)
                    if self.global_step > self.total_steps: break
                    # if not batch_is_valid: continue
                    step += 1
                    
                    loss, cos_sim = self.model(wav_stacked)
                    
                    # Accumulate Loss
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                    if self.apex and self.args.multi_gpu:
                        raise NotImplementedError
                    elif self.apex:
                        self.optimizer.backward(loss)
                    elif self.args.multi_gpu:
                        loss = loss.sum()
                        loss.backward()
                        # log_loss = torch.log10(loss)
                        # log_loss.backward()
                    else:
                        loss.backward()
                        # log_loss = torch.log10(loss)
                        # log_loss.backward()
                    loss_val += loss.item()

                    # Update
                    if (step+1) % self.gradient_accumulation_steps == 0:
                        if self.apex:
                            # modify learning rate with special warm up BERT uses
                            # if conifg.apex is False, BertAdam is used and handles this automatically
                            lr_this_step = self.learning_rate * self.warmup_linear.get_lr(self.global_step, self.warmup_proportion)
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr_this_step

                        # Step
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                        if math.isnan(grad_norm):
                            print('[Runner] - Error : grad norm is NaN @ step ' + str(self.global_step))
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                        if self.global_step % self.log_step == 0:
                            # Log
                            self.log.add_scalar('lr', self.optimizer.get_lr()[0], self.global_step)
                            self.log.add_scalar('loss', (loss_val), self.global_step)
                            self.log.add_scalar('gradient norm', grad_norm, self.global_step)
                            progress.set_description("Loss %.4f" % (loss_val))

                        if self.global_step % self.save_step == 0:
                            self.save_model('states')
                            cos_sim_0 = torch.tensor(plot_attention_to_numpy(cos_sim[0].data.cpu().numpy()))
                            cos_sim_1 = torch.tensor(plot_attention_to_numpy(cos_sim[1].data.cpu().numpy()))
                            cos_sim_2 = torch.tensor(plot_attention_to_numpy(cos_sim[2].data.cpu().numpy()))
                            self.log.add_image('cos_sim_img/0', cos_sim_0, self.global_step)
                            self.log.add_image('cos_sim_img/1', cos_sim_1, self.global_step)
                            self.log.add_image('cos_sim_img/2', cos_sim_2, self.global_step)

                        loss_val = 0
                        pbar.update(1)
                        self.global_step += 1
                        
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(e)
                        print('CUDA out of memory at step: ', self.global_step)
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                    else:
                        raise
            if self.args.multi_gpu: self.model.module.kappa *= 0.98
            else: self.model.kappa *= 0.98
                
        pbar.close()
        self.log.close()
