__all__ = ['PrunerScheduler']

from cProfile import label
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer
#import deepspeed

from .scheduler import get_constant_linear_schedule_with_warmup
from .bbs import Prune as PruneBBS
from .misc import generate_prune_dict

from .logger import Logger


class PrunerScheduler:
    TYPE = 'bbs'
    def __init__(self,
            model,
            steps_per_epoch: int,
            num_steps: int,
            sparsities: list,
            prune_freq: int,
            log_path: str, 
            rank: int,
            optimizer: Optimizer,
            seq_len: int,
            prune_dict: dict=None, 
            ckpt_sparsities: list=None,
            pruner_resume_dict: dict=None,
            bank_size: int=64,
            finetune: bool=False,
            log_freq: int=100,
            **kwargs):  
        """
        Args:
            model (unwarped_model) : model reference 
            steps_per_epoch (int): number of steps per epoch
            num_steps (int) : total steps of the training phase
            bank_size (int) : number of weight used for pruning per group
            prune_freq (int) : pruning frequency
            sparsities (list) : list of sparsities used during pruning
            log_path (str) : the saving path for log file
            rank (int) : gpu rank number
            optimizer (torch.nn.optim.Optimizer) : 
            prune_dict (dict) : prune dict, the key is the layer name, the value is the specific sparisity
            ckpt_sparsities (list) : each sparsities is the target sparisity 
                for model which need to be saved during pruning. When this parameter is setting, the sparisities 
                parameter need to only contain single sparsity value.
            pruner_resume_dict: key in dict [path,load_model_states,load_optimizer_states] 
            finetune (bool) : only finetune the model
            log_freq (int) : logging frequency,
            is_zero3 (bool) : use deepspeed ZeRO 3, default: false
            seq_len (int): input sequence length used for automatic cgb generation,default: None
        """
        self.model = model
        self.steps_per_epoch = steps_per_epoch
        self.num_steps = num_steps
        self.bank_size = bank_size
        self.sparsities = sparsities
        self.prune_freq = prune_freq
        self.log_path = os.path.join(log_path,'prune')
        self.rank = rank
        self.optimizer = optimizer
        self.pruners = [] 
        self.optim_schedulers = []
        self.stage_wise_steps = []
        self.prune_dicts = []
        self.prune_steps = []
        self.step = 0
        self.best_metric = 0
        self.index = -1
        self.kwargs = kwargs
        self.resume_tag = False
        self.finetune = finetune
        self.log_freq = log_freq
        # decrease the sparsity with some buffer
        self.ckpt_sparsities = []
        if ckpt_sparsities:
            self.ckpt_sparsities = [x-0.015 for x in ckpt_sparsities]


        os.makedirs(self.log_path,exist_ok=True)
        #split steps into multi stage 
        self.stage_wise_steps = list(get_stage_steps(sparsities=sparsities,
                                                    num_steps=num_steps,
                                                    save_path=os.path.join(self.log_path,'sparsity_step.jpg'),
                                                    type='exp'))

        # build logger
        logger_args = {
            'log_dir':self.log_path,
            'rank': self.rank,
            'name': 'pruner'
        }  
        self.logger = Logger(**logger_args)
        self.logger.info(f"Model Sparsity: {sparsities}")

        for i,sparsity in enumerate(sparsities):
            temp_prune_dict,self.set_up_infos = generate_prune_dict(model, sparsity,seq_len)
            prune_step = int(0.8*(self.stage_wise_steps[i][1]-self.stage_wise_steps[i][0]))
            if self.finetune:
                prune_step = 0
            self.stage_wise_steps[i].insert(1,self.stage_wise_steps[i][0]+prune_step) 
            
            for k, v in prune_dict.items():
                if k in temp_prune_dict.keys():
                    if v==0:
                        temp_prune_dict.pop(k)
                    else:
                        temp_prune_dict[k] = v
            
            self.prune_steps.append(prune_step)
            self.prune_dicts.append(temp_prune_dict)
            
            self.logger.info(f"stage wise steps: {self.stage_wise_steps[i]}")
            self.logger.info(f"Prune dict: {self.prune_dicts[i]}")
            self.logger.info(f"Set up infos: {self.set_up_infos}")

            scheduler = get_constant_linear_schedule_with_warmup(optimizer,
                                    prune_step,
                                    self.stage_wise_steps[i][2]-self.stage_wise_steps[i][0])
            #if i==(len(sparsities)-1):
            scheduler.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
            self.optim_schedulers.append(scheduler)

        # resume from ckpt
        if pruner_resume_dict:
            state_dict = torch.load(pruner_resume_dict['path'])
            self.step = state_dict['step']
            self.index = state_dict['index']
            self.optim_schedulers[self.index].load_state_dict(state_dict['lr_scheduler'])
            if pruner_resume_dict['load_model_states']:
                self.model.load_state_dict(state_dict['model'])
            if pruner_resume_dict['load_optimizer_states']:
                self.optimizer.load_state_dict(state_dict['optimizer'])

            self.resume_tag = True
            self.init_pruner()
            self.logger.info(f"resume from {pruner_resume_dict['path']}")
            self.logger.info(f"initial step {self.step}")
            self.logger.info(f"initial model sparsity {self.init_sparsity(self.prune_dicts[self.index])[1]}")
        else:
            self.init_pruner()


    def init_pruner(self):
       if (self.index+1 < len(self.stage_wise_steps) and self.step == self.stage_wise_steps[self.index+1][0]) or self.resume_tag:
            self.best_metric = 0.
            if not self.resume_tag:
                self.index += 1
            # check which phase is the resume ckpt in,prune phase or finetune phase 
            if self.resume_tag:
                resume_pruner_step = self.prune_steps[self.index] if self.step>=self.stage_wise_steps[self.index][1] \
                        else self.step-self.stage_wise_steps[self.index][0]
                self.logger.info(f"index: {self.index} | prune_steps:{self.prune_steps[self.index]} | step {self.step} | stage_wise_steps 1: \
                        {self.stage_wise_steps[self.index][1]} | stage_wise_steps 0: {self.stage_wise_steps[self.index][0]}")
                self.logger.info(f"resume_pruner_step: {resume_pruner_step} | sparse step: {self.prune_steps[self.index]}")

            self.pruner = PruneBBS(
                model=self.model,
                pretrain_step=0,
                sparse_step=0 if self.finetune else self.prune_steps[self.index],
                restore_sparsity=True,
                current_step=resume_pruner_step if self.resume_tag else 0,
                frequency=self.prune_freq,
                prune_dict=self.prune_dicts[self.index],
                deploy_device='asic',
                group_size=self.bank_size,
                set_up_infos=self.set_up_infos,
            )
            self.logger.info(f"Done of initial pruner. Model sparsity is {self.pruner.sparsity()[1]}")
            

            if self.resume_tag:
                self.resume_tag=False

        
    def prune(self):
        self.optim_schedulers[self.index].step()
        self.pruner.prune()

        if self.step == self.stage_wise_steps[self.index][2] - 1:
            self.logger.info(f"Finished pruning model with sparsity {self.sparsities[self.index]}")

        #if self.rank in [-1,0]:
        if self.step%self.log_freq==0: 
            self.logger.info(f"model sparsity{self.sparsity()[1]}")
        if self.step>0 and (self.step+1)%self.prune_freq==0:
            curr_sparsity = self.pruner.sparsity()[1]
            
            if self.ckpt_sparsities and curr_sparsity>=self.ckpt_sparsities[0]:
                ckpt_path = os.path.join(
                    self.log_path,
                    'sparsity{:.4f}.pth'.format(curr_sparsity)
                )
                torch.save(self.model.state_dict(),ckpt_path)
                self.ckpt_sparsities.pop(0)

                self.logger.info(f"Get model with sparsity{curr_sparsity}")

        self.step += 1
        self.init_pruner() 


    def sparsity(self):
        total_param = 0
        total_zero = 0
        layer_sparse_rate = {}
        total_sparse_rate = -1
        with torch.no_grad():
            for name,parameter in self.model.named_parameters():
                if self.prune_dicts[self.index].get(name): 
                    num_param = torch.numel(parameter) 
                    zero_param = num_param-torch.nonzero(parameter).shape[0]
                    layer_sparse_rate[name] = zero_param/num_param
                    total_param += num_param
                    total_zero += zero_param
            total_sparse_rate = total_zero / total_param
        return  layer_sparse_rate,total_sparse_rate

    
    def init_sparsity(self,prune_dict):
        total_param = 0
        total_nonezero = 0
        layer_sparse_rate = {}
        for name, parameter in self.model.named_parameters():
            if any(name == one for one in prune_dict):
                temp = parameter.data.cpu().numpy()
                total_param = total_param + temp.size
                total_nonezero = total_nonezero + np.flatnonzero(temp).size
                layer_sparse_rate[name] = 1 - np.flatnonzero(temp).size / temp.size
        total_sparse_rate = 1 - total_nonezero / total_param
        return layer_sparse_rate, total_sparse_rate


    def update_metrics(self, metric):
        if self.rank in [-1,0]:
            # temporary change , need to delete after MYD project
            if metric > self.best_metric and self.step >= self.stage_wise_steps[self.index][1]:
                self.best_metric = metric
                ckpt_path = os.path.join(
                    self.log_path,
                    f'sparsity{str(self.sparsities[self.index])}_metric{metric}.pth'
                )
                torch.save(self.model.state_dict(), ckpt_path)
                self.logger.info(f"Found the best model! Metric is {metric}, checkpoint saved..")


    def save(self,**kwargs):
        if self.rank in [-1,0] and self.step%self.steps_per_epoch==0:
            # save latest ckpt
            state_dict = {
                "step": self.step,
                "index": self.index,
                "lr_scheduler": self.optim_schedulers[self.index].state_dict(),
                
            }
            if not self.is_zero3:
                state_dict.update({
                    "model":self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                })
            # WARNING !!!: the key in the kwargs should not contain the key of the state dict
            state_dict.update(kwargs)

            ps_ckpt_path = os.path.join(self.log_path,'pruner_scheduler.pth')
            torch.save(state_dict,ps_ckpt_path)
            self.logger.info(f"Saved pruner ckpt with step={self.step},index={self.index}")



def even_split(a, n):
    k, m = divmod(a, n)
    return ([i*k+min(i, m),(i+1)*k+min(i+1, m)] for i in range(n))


def get_stage_steps(sparsities: list,num_steps: int,save_path:str,type: str='linear'):
    reverted_sparsities = [1/(1-x) for x in sparsities]
    if type=='linear':
        ratio = [x/sum(reverted_sparsities) for x in reverted_sparsities]
    elif type=='exp':
        scale = 0.3
        bias = 0.5
        ratio  = [2**(scale*x+bias) for x in reverted_sparsities]
        ratio = [x/sum(ratio) for x in ratio]
    else:
        raise NotImplementedError("only support linear and exp stategy")

    #save sparsity-num_step plot 
    step_per_sparsity = [int(x*num_steps+0.5) for x in ratio]
    plt.plot(reverted_sparsities,step_per_sparsity,label=type)
    plt.legend(loc='upper left')
    plt.savefig(save_path)

    stage_wise_step = []
    start_step,end_step = 0,0
    for i,r in enumerate(ratio):
       end_step = start_step+int(ratio[i]*num_steps+0.5)
       stage_wise_step.append([start_step,min(end_step,num_steps)]) 
       start_step = end_step
    return stage_wise_step


