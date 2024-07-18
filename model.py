## File adapated from https://github.com/mechanistic-interpretability-grokking/progress-measures-paper

import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import time
import math
import logging
import torch.nn.functional as F
import einops
import random
from dataclasses import dataclass, field
import os
import wandb
import dataclasses
from collections import defaultdict
from typing import Optional, List

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from torch.utils.data import DataLoader
import argparse

import plot
import helpers 

@dataclass(frozen = True)
class Config():
    load_model_path: str = "saved_runs/layers_2_heads_1_4_1M/900.pth" #@param
    data_path: str = "data/cnf_tokens_1M.npy" #@param
    data_path_interpretation: str = "data/cnf_tokens_100K.npy" #@param
    varlen_data_path: str = "data/cnf_tokens_7_8_9_varlen.npy" #@param
    satunsat_data_path: str = "data/cnf_tokens_sat_unsat.npy" #@param
    base_cache_path: str = "cache" #@param
    lr: float = 1e-3 #@param
    weight_decay: float = 1.0 #@param
    nvars: int = 5 #@param
    nclauses: int = 10 #@param
    d_model: int = 128 #@param
    frac_train: float = 0.6 #@param
    num_epochs: int = 1000 #@param
    save_models: bool = True #@param
    save_every: int = 100 #@param
    log_every: int = 10 #@param

    # Stop training when test loss is <stopping_thresh
    stopping_thresh: int = -1 #@param
    seed: int = 0 #@param

    num_layers: int = 2
    batch_style: str = 'full'
    batch_size: int = 10000
    d_vocab: int = 16 # nvars * 2 + 6
    n_ctx: int = 41 # nclauses * 4 + 1; length of each formula
    d_mlp: int = 4 * d_model
    num_heads: list = field(default_factory=lambda: [4,4])
    act_type: str = 'ReLU' #@param ['ReLU', 'GeLU']
    use_ln: bool = False
    verbose: bool = False

    take_metrics_every_n_epochs: int = 100 #@param
    device: t.device = t.device("cuda")
    device_ids: Optional[List[int | t.device]] = None
    data_parallel: bool = False

    # Analysis settings
    differential: bool = False #@param
    just_eval: bool = True #@param
    activation_threshold: float = 0.5 # Used by alpha to translate activations to booleans
    high_activation: float = 2 # Used by gamma to translate booleans to activations

    @property
    def cache_path(self):
        cache_file_name = "-".join(self.load_model_path.split("/")[-2:] + ["data", self.data_path_interpretation.split("/")[-1]])
        return os.path.join(self.base_cache_path, cache_file_name)

    def d_head(self, block_id):
        return self.d_model // self.num_heads[block_id]

    @property
    def random_answers(self):
        return np.random.randint(low=0, high=self.p, size=(self.p, self.p))

    def is_it_time_to_save(self, epoch):
        return (epoch % self.save_every == 0)

    def is_it_time_to_take_metrics(self, epoch):
        return epoch % self.take_metrics_every_n_epochs == 0

class HookPoint(nn.Module):
    '''
    A helper class to get access to intermediate activations (inspired by Garcon).
    By wrapping intermediate activations, gives a convenient way to add PyTorch hooks.
    '''
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    
    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output, 
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        return x


class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_model))
    
    def forward(self, x):
        return t.einsum('dbp -> bpd', self.W_E[:, x])

#| export
class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_vocab))
    
    def forward(self, x):
        return (x @ self.W_U)

#| export
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model)/np.sqrt(d_model))
    
    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

#| export
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon
    
    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

#| export
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(t.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(t.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(t.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked/np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(t.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = t.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

#| export
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.act_type = act_type
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']
        
    def forward(self, x):
        x = self.hook_pre(t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

# export
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    
    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x

class TransformerBlockNoMLP(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    
    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
        return x

#| export
class Transformer(nn.Module):
    def __init__(self, config: Config, use_cache=False, use_ln=True):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache
        self.embed = Embed(d_vocab = config.d_vocab, d_model = config.d_model)
        self.pos_embed = PosEmbed(max_ctx = config.n_ctx, d_model = config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model = config.d_model,
            d_mlp = config.d_mlp,
            d_head = config.d_head(i),
            num_heads = config.num_heads[i],
            n_ctx = config.n_ctx,
            act_type = config.act_type,
            model=[self]) for i in range(config.num_layers)])

        self.unembed = Unembed(d_vocab = config.d_vocab, d_model = config.d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
    
    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')

def full_loss(config : Config, model: Transformer, data):
    '''Takes the cross entropy loss of the model on the data'''
    # Take the final position only
    logits = model(data[:,:-1])[:, -1]
    labels = t.tensor(data[:,-1]).to(config.device)
    return helpers.cross_entropy_high_precision(logits, labels), helpers.get_acc(logits, labels), helpers.is_acc(logits, labels)

def gen_fixedlen_train_test(config : Config, interpretation=False):
    '''Generate train and test split'''
    all_data = np.load(config.data_path_interpretation if interpretation else config.data_path)
    random.seed(config.seed)
    np.random.seed(config.seed)
    sat_token = 2*config.nvars + 3
    unsat_token = 2*config.nvars + 4

    num_sat_instances = np.count_nonzero(all_data[:,-1] == sat_token)
    num_unsat_instances = np.count_nonzero(all_data[:,-1] == unsat_token)
    print("SAT instances in fixed len data: ", num_sat_instances)
    print("UNSAT instances in fixed len data: ", num_unsat_instances)
    div = int(config.frac_train*len(all_data))

    ''' Following snippet for train and test datasets with balanced SAT and UNSAT instances'''
    sorted_data = all_data[all_data[:, -1].argsort(kind="stable")]
    train = np.concatenate((sorted_data[:div//2,:], sorted_data[num_sat_instances:num_sat_instances+div//2,:]))
    test = np.concatenate((sorted_data[div//2:num_sat_instances,:], sorted_data[num_sat_instances+div//2:,:]))
    np.random.shuffle(train)
    np.random.shuffle(test)

    print("SAT instances in train: ", np.count_nonzero(train[:,-1] == sat_token))
    print("UNSAT instances in train: ", np.count_nonzero(train[:,-1] == unsat_token))
    print("SAT instances in test: ", np.count_nonzero(test[:,-1] == sat_token))
    print("UNSAT instances in test: ", np.count_nonzero(test[:,-1] == unsat_token))
    return train, test

def gen_varlen_test(config : Config):
    '''Load variable length test data'''
    all_data = np.load(config.varlen_data_path)
    sat_token = 2*config.nvars + 3
    unsat_token = 2*config.nvars + 4
    print("SAT instances in varlen data: ", np.count_nonzero(all_data[:,-1] == sat_token))
    print("UNSAT instances in varlen data: ", np.count_nonzero(all_data[:,-1] == unsat_token))
    return all_data

def gen_satunsat_test(config : Config):
    '''Load test data with pairs of SAT/UNSAT instances
    We assume that the dataset is ordered such that every even index is a SAT formula and the next odd index
    is the corresponding UNSAT formula'''
    all_data = np.load(config.satunsat_data_path)
    sat_token = 2*config.nvars + 3
    unsat_token = 2*config.nvars + 4
    print("SAT instances in varlen data: ", np.count_nonzero(all_data[:,-1] == sat_token))
    print("UNSAT instances in varlen data: ", np.count_nonzero(all_data[:,-1] == unsat_token))
    return all_data

class Trainer:
    def __init__(self, config : Config, run_name = None, model = None) -> None:
        wandb.init(project = "2sat", config = dataclasses.asdict(config), mode="disabled")
        self.model = model if model is not None else Transformer(config, use_cache=False)

        if config.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=config.device_ids)

        print("number of parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.model.to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr = config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step/10, 1))
        self.run_name = f"2sat_{int(time.time())}" if run_name is None else run_name
        self.train, self.test = gen_fixedlen_train_test(config = config)
        self.metrics_dictionary = defaultdict(dict) # so we can safely call 'update' on keys
        print('training length = ', len(self.train))
        print('testing length = ', len(self.test))
        self.train_losses = []
        self.test_losses = []
        self.config = config
        
        if not os.path.isdir(helpers.root/self.run_name):
                os.makedirs(helpers.root/self.run_name, exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(helpers.root/self.run_name/"log.txt"),
                logging.StreamHandler(),
            ])


    def save_epoch(self, epoch, save_to_wandb = True):
        ''' precondition! train loss and test losses have been appended to '''
        model = self.model.module if self.config.data_parallel else self.model
        save_dict = {
            'model': model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'epoch': epoch,
        }
        if save_to_wandb:
            wandb.log(save_dict)
            self.logger.info("Saved epoch to wandb")
        if self.config.save_models: 
            t.save(save_dict, helpers.root/self.run_name/f"{epoch}.pth")
            self.logger.info(f"Saved model to {helpers.root/self.run_name/f'{epoch}.pth'}")
        self.metrics_dictionary[epoch].update(save_dict)

    def do_a_training_step(self, epoch: int):
        '''returns train_loss, test_loss'''
        num_batches_train = math.ceil(np.shape(self.train)[0]/float(self.config.batch_size))
        # print("num batches train: ", num_batches_train)
        train_loss_epoch = 0
        train_acc_epoch = 0
        for batch_num in range(num_batches_train):
            start_id = batch_num * self.config.batch_size
            train_data = t.LongTensor(self.train[start_id : start_id + self.config.batch_size, ...]).to(self.config.device)
            train_loss_b, train_acc_b, _ = full_loss(config = self.config, model = self.model, data = train_data)
            train_loss_epoch += train_loss_b.item()
            train_acc_epoch += train_acc_b.item()
            train_loss_b.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()
        train_loss = train_loss_epoch / num_batches_train
        train_acc = train_acc_epoch / num_batches_train

        num_batches_test = math.ceil(np.shape(self.test)[0]/float(self.config.batch_size))
        test_loss_epoch = 0
        test_acc_epoch = 0
        for batch_num in range(num_batches_test):
            start_id = batch_num * self.config.batch_size
            test_data = t.LongTensor(self.test[start_id : start_id + self.config.batch_size, ...]).to(self.config.device)
            test_loss_b, test_acc_b, _ = full_loss(config = self.config, model = self.model, data = test_data)
            test_loss_epoch += test_loss_b.item()
            test_acc_epoch += test_acc_b.item()
        test_loss = test_loss_epoch / num_batches_test
        test_acc = test_acc_epoch / num_batches_test
        
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        if epoch % self.config.log_every == 0:
            self.logger.info(f'Epoch {epoch}, train loss {train_loss:.4f}, train acc {train_acc:.4f},\
                   test loss {test_loss:.4f}, test acc {test_acc:.4f}')
        
        return train_loss, test_loss

    def initial_save_if_appropriate(self):
        if self.config.save_models:
            if not os.path.isdir(helpers.root/self.run_name):
                os.makedirs(helpers.root/self.run_name, exist_ok=True)
            model = self.model.module if self.config.data_parallel else self.model
            save_dict = {
                'model': model.state_dict(),
                'train_data' : self.train,
                'test_data' : self.test}
            t.save(save_dict, helpers.root/self.run_name/'init.pth')


    def post_training_save(self, save_optimizer_and_scheduler = True, log_to_wandb = True):
        if not self.config.save_models:
            os.makedirs(helpers.root/self.run_name, exist_ok=True)

        model = self.model.module if self.config.data_parallel else self.model
        save_dict = {
            'model': model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'epoch': self.config.num_epochs,
        }
        if save_optimizer_and_scheduler:
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
        if log_to_wandb:
            wandb.log(save_dict)
        t.save(save_dict, helpers.root/self.run_name/f"final.pth")
        self.logger.info(f"Saved model to {helpers.root/self.run_name/f'final.pth'}")
        self.metrics_dictionary[save_dict['epoch']].update(save_dict)


    def take_metrics(self, train, epoch):
        with t.inference_mode():
            def sum_sq_weights():
                # TODO refactor- taken from app code
                row = []
                for name, param in self.model.named_parameters():
                    row.append(param.pow(2).sum().item())
                return row

            self.logger.info('taking metrics')

            metrics = {
                'epoch': epoch, 
                'trig_loss': -1,
                'sum_of_squared_weights': sum_sq_weights(),
                'excluded_loss': -1,
                'coefficients': -1,
            }
            wandb.log(metrics)
            self.logger.info("Logged metrics to wandb")
            self.metrics_dictionary[epoch].update(metrics)

def train_model(config: Config, model=None, run_name: str = None):
    print(config)
    world = Trainer(config = config, run_name=run_name, model=model)
    print(f'Run name {world.run_name}')
    world.initial_save_if_appropriate()

    for epoch in range(config.num_epochs):
        train_loss, test_loss = world.do_a_training_step(epoch)
        if test_loss < config.stopping_thresh:
            break
        if config.is_it_time_to_save(epoch = epoch):
            world.save_epoch(epoch = epoch)
        if config.is_it_time_to_take_metrics(epoch = epoch):
            world.take_metrics(epoch = epoch, train = world.train)

    world.post_training_save(save_optimizer_and_scheduler=True)
    plot.lines([world.train_losses, world.test_losses], labels=['train', 'test'], log_y=True)
    return world # to export the dictionary with the training metrics



def test_model(config: Config):
    model_dict = t.load(config.load_model_path)
    print(model_dict.keys())
    print("Train loss: ", model_dict['train_loss'])
    print("Test loss: ", model_dict['test_loss'])
    print("Epoch: ", model_dict['epoch'])
    model = Transformer(config, use_cache=False)
    model.to('cuda')
    model.load_state_dict(model_dict['model'])

    if config.data_path != None:
        train, test = gen_fixedlen_train_test(config = config)
        
        num_samples = np.shape(test)[0]
        num_batches_test = math.ceil(num_samples/float(config.batch_size))
        test_loss_epoch = 0
        test_acc_epoch = 0

        if config.just_eval:
            for batch_num in range(num_batches_test):
                cache = {}
                model.remove_all_hooks()
                model.cache_all(cache)
                start_id = batch_num * config.batch_size
                test_data = test[start_id : start_id + config.batch_size, ...]
                test_loss_b, test_acc_b, is_acc = full_loss(config = config, model = model, data = test_data)
                test_loss_epoch += test_loss_b.item()
                test_acc_epoch += test_acc_b.item()

            test_loss = test_loss_epoch / num_batches_test
            test_acc = test_acc_epoch / num_batches_test    
            print(f'test loss {test_loss:.4f}, test acc {test_acc:.4f}')
        
        else:
            graph_stats_acc_map = []

            for sample_num in range(num_samples):
                cache = {}
                model.remove_all_hooks()
                model.cache_all(cache)
                test_data = test[sample_num: sample_num + 1, ...]
                test_loss_b, test_acc_b, is_acc = full_loss(config = config, model = model, data = test_data)
                test_loss_epoch += test_loss_b.item()
                test_acc_epoch += test_acc_b.item()

                # NOTE: The following graph stats code does not work with batches 
                graph_stats = helpers.get_graph_stats(helpers.implication_graph(test_data[0],config)[1])
                graph_stats_acc_map.append(np.array([*graph_stats,test_acc_b.item()]))

                if sample_num % config.batch_size == 0:
                    helpers.generate_attention_map(config, cache, sample_num)
                    print(f'Formula {sample_num}: {helpers.decode_to_CNFs(test_data[0], config)}')
                    print(f'Model correct for formula {sample_num}: {is_acc[0]}')

                    if config.differential:
                        if test_data[0,-1] == 2*config.nvars + 4: # if formula is UNSAT
                            print("Differential Analysis: Converting UNSAT to SAT...")
                            pair_of_formulas = helpers.convert_UNSAT_to_SAT(test_data[0], config)
                            if pair_of_formulas is None:
                                print("Could not find a SAT version of the formula")
                            else:
                                (sat_formula, unsat_formula) = pair_of_formulas
                                sat_formula = np.expand_dims(sat_formula, axis=0)
                                cache = {}
                                model.remove_all_hooks()
                                model.cache_all(cache)
                                test_loss_b, test_acc_b, is_acc = full_loss(config = config, model = model, data = sat_formula)
                                helpers.generate_attention_map(config, cache, f'{sample_num}_SAT')
                                print(f'Formula {sample_num}: {helpers.decode_to_CNFs(sat_formula[0], config)}')
                                print(f'Model correct for SAT version of formula {sample_num}: {is_acc[0]}')

                                unsat_formula = np.expand_dims(unsat_formula, axis=0)
                                cache = {}
                                model.remove_all_hooks()
                                model.cache_all(cache)
                                test_loss_b, test_acc_b, is_acc = full_loss(config = config, model = model, data = unsat_formula)
                                helpers.generate_attention_map(config, cache, f'{sample_num}_UNSAT')
                                print(f'Formula {sample_num}: {helpers.decode_to_CNFs(unsat_formula[0], config)}')
                                print(f'Model correct for UNSAT version of formula {sample_num}: {is_acc[0]}')

            test_loss = test_loss_epoch / num_samples
            test_acc = test_acc_epoch / num_samples    
            print(f'test loss {test_loss:.4f}, test acc {test_acc:.4f}')

            graph_stats_acc_map = np.stack(graph_stats_acc_map, axis=0)
            helpers.print_graph_stats(graph_stats_acc_map)

    if config.varlen_data_path != None:
        varlen_test = gen_varlen_test(config = config)
        num_batches_varlen_test = math.ceil(np.shape(varlen_test)[0]/float(config.batch_size))
        varlen_test_loss_epoch = 0
        varlen_test_acc_epoch = 0
        for batch_num in range(num_batches_varlen_test):
            start_id = batch_num * config.batch_size
            test_data = varlen_test[start_id : start_id + config.batch_size, ...]
            test_loss_b, test_acc_b, is_acc = full_loss(config = config, model = model, data = test_data)
            varlen_test_loss_epoch += test_loss_b.item()
            varlen_test_acc_epoch += test_acc_b.item()

        varlen_test_loss = varlen_test_loss_epoch / num_batches_varlen_test
        varlen_test_acc = varlen_test_acc_epoch / num_batches_varlen_test    
        print(f'variable length test loss {varlen_test_loss:.4f}, test acc {varlen_test_acc:.4f}')
    
    if config.satunsat_data_path != None:
        test = gen_satunsat_test(config = config)
        
        num_samples = np.shape(test)[0]
        test_loss_epoch = 0
        test_acc_epoch = 0

        for idx in range(num_samples // 2):
            cache = {}
            model.remove_all_hooks()
            model.cache_all(cache)
            sat_formula = test[2*idx: 2*idx + 1, ...]
            unsat_formula = test[2*idx+1: 2*idx + 2, ...]
            test_loss_b, test_acc_b, is_acc = full_loss(config = config, model = model, data = sat_formula)
            test_loss_epoch += test_loss_b.item()
            test_acc_epoch += test_acc_b.item()
            helpers.generate_attention_map(config, cache, 2*idx)
            print(f'SAT Formula {2*idx}: {helpers.decode_to_CNFs(sat_formula[0], config)}')
            print(f'Model correct for formula {2*idx}: {is_acc[0]}')

            cache = {}
            model.remove_all_hooks()
            model.cache_all(cache)
            test_loss_b, test_acc_b, is_acc = full_loss(config = config, model = model, data = unsat_formula)
            test_loss_epoch += test_loss_b.item()
            test_acc_epoch += test_acc_b.item()
            helpers.generate_attention_map(config, cache, 2*idx+1)
            print(f'UNSAT Formula {2*idx + 1}: {helpers.decode_to_CNFs(unsat_formula[0], config)}')
            print(f'Model correct for formula {2*idx + 1}: {is_acc[0]}')


        test_loss = test_loss_epoch / num_samples
        test_acc = test_acc_epoch / num_samples    
        print(f'test loss {test_loss:.4f}, test acc {test_acc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2-SAT Transformer Mechanistic Interpretability')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of transformer blocks')
    parser.add_argument('--num_heads', type=int, nargs='+', default=[4,4],
                        help='list of number of heads for each block')
    parser.add_argument('--num_vars', type=int, default=5,
                        help='number of variables in the formulas')
    parser.add_argument('--num_clauses', type=int, default=10,
                        help='number of clauses in the formulas')
    parser.add_argument('--run_name', type=str, default=None,
                        help='name of saved models folder')
    parser.add_argument('--train', action='store_true',
                        help='train or test mode')
    parser.add_argument('--load_model_path', type=str, default=None,
                        help='path to saved model')
    parser.add_argument('--data_path', type=str, default=None,
                        help='path to instance data')
    parser.add_argument('--varlen_data_path', type=str, default=None,
                        help='path to variable length instance data')
    parser.add_argument('--satunsat_data_path', type=str, default=None,
                        help='path to pairs of SAT/UNSAT instance data')
    parser.add_argument('--differential', action='store_true',
                        help='perform differential analysis')
    parser.add_argument('--just_eval', action='store_true',
                        help='just evaluate the model')
    cmd_args = parser.parse_args()
    print(cmd_args)
    if cmd_args.train:
        config=Config(num_layers=cmd_args.num_layers, num_heads=cmd_args.num_heads, nvars=cmd_args.num_vars, nclauses=cmd_args.num_clauses, 
                      d_vocab=cmd_args.num_vars * 2 + 6, n_ctx=cmd_args.num_clauses * 4 + 1, data_path=cmd_args.data_path)
        for i in range(len(config.num_heads)):
            assert config.d_model % config.num_heads[i] == 0
        train_model(config=config, run_name=cmd_args.run_name)
    else:
        config=Config(num_layers=cmd_args.num_layers,num_heads=cmd_args.num_heads, nvars=cmd_args.num_vars, nclauses=cmd_args.num_clauses,
                      d_vocab=cmd_args.num_vars * 2 + 6, n_ctx=cmd_args.num_clauses * 4 + 1, load_model_path=cmd_args.load_model_path,
                      data_path=cmd_args.data_path, varlen_data_path=cmd_args.varlen_data_path, satunsat_data_path=cmd_args.satunsat_data_path, 
                      differential=cmd_args.differential, just_eval=cmd_args.just_eval)
        for i in range(len(config.num_heads)):
            assert config.d_model % config.num_heads[i] == 0
        test_model(config=config)
