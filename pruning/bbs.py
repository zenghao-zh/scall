__all__ = ["Prune"]

import os
import torch
import numpy
import numpy as np

from .compiler_frontend_prune_unit import (
    update_mask,
    prune_dim_2,
    prune_dim_3,
    prune_dim_4
)


class Prune:
    def __init__(
        self,
        model,
        pretrain_step: int = 0,
        sparse_step: int = 0,
        current_step: int=0,
        frequency: int = 100,
        prune_dict: dict = {},
        restore_sparsity: bool = False,
        fix_sparsity: bool = False,
        prune_device: str = "default",
        deploy_device: str = "none",
        group_size: int = 64,
        set_up_infos: dict = None,
    ):
        self._model = model
        self.set_up_infos = set_up_infos
        self._t = current_step
        self._initial_sparsity = {}
        self._pretrain_step = pretrain_step
        self._sparse_step = sparse_step
        self._frequency = frequency
        self._prune_dict = prune_dict
        self._restore_sparsity = restore_sparsity
        self._fix_sparsity = fix_sparsity
        self._prune_device = prune_device
        self._deploy_device = deploy_device
        self._fpga_input_group = 4
        # self._asic_input_gloup = 8
        self._group_size = group_size
        self._asic_input_gloup = 512 // group_size
        self._mask = {}
        self._check_parameter()
        self._prepare()


    def _check_parameter(self):
        assert isinstance(self._pretrain_step, int)
        assert isinstance(self._sparse_step, int)
        assert isinstance(self._frequency, int)
        assert isinstance(self._prune_dict, dict)
        assert isinstance(self._restore_sparsity, bool)
        assert isinstance(self._fix_sparsity, bool)
        assert self._prune_device in ["default", "cpu"]
        assert self._deploy_device in ["none", "fpga", "asic"]
        

    def _prepare(self):
        with torch.no_grad():
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    if (
                        (self._deploy_device == "fpga")
                        and (len(parameter.shape) == 4)
                        and (parameter.shape[1] < self._fpga_input_group)
                    ):
                        self._prune_dict.pop(name)
                        print(
                            "For %s, the parameter %s cannot be balanced pruned and will be deleted from the prune_dict."
                            % (self._deploy_device, name)
                        )
                        continue
                    elif (
                        (self._deploy_device == "asic")
                        and (len(parameter.shape) == 4)
                        and (parameter.shape[1] < self._asic_input_gloup)
                        and ([parameter.shape[2], parameter.shape[3]] == [1, 1])
                    ):
                        self._prune_dict.pop(name)
                        print(
                            "For %s, the parameter %s cannot be balanced pruned and will be deleted from the prune_dict."
                            % (self._deploy_device, name)
                        )
                        continue
                    weight = self._get_weight(parameter)
                    if self._restore_sparsity == True:
                        mask = torch.where(
                            weight == 0,
                            torch.zeros_like(weight),
                            torch.ones_like(weight),
                        )
                        self._initial_sparsity[name] = (
                            1
                            - mask.cpu().float().numpy().sum()
                            / weight.cpu().float().numpy().size
                        )
                        self._mask[name] = mask
                    else:
                        self._initial_sparsity[name] = 0
                        self._mask[name] = torch.ones_like(weight)


    def _update_mask(self, weight, keep_k):
        if keep_k >= 1:
            reshape_weight = weight.reshape(-1)
            index = torch.topk(reshape_weight.abs(), keep_k)[1].cpu().numpy().tolist()
            mask = numpy.zeros(reshape_weight.shape)
            mask[index] = 1
            mask = mask.reshape(weight.shape)
            mask = torch.as_tensor(mask, dtype=weight.dtype, device=weight.device)
        else:
            mask = torch.zeros_like(weight)
        return mask
        

    def _update_mask_conditions(self):
        condition1 = self._fix_sparsity == False
        condition2 = (
            self._pretrain_step < self._t <= self._pretrain_step + self._sparse_step
        )
        condition3 = (self._t - self._pretrain_step) % self._frequency == 0
        return condition1 and condition2 and condition3


    def _get_weight(self, parameter):
        if self._prune_device == "default":
            weight = parameter.data
        elif self._prune_device == "cpu":
            weight = parameter.data.to(device=torch.device("cpu"))
        return weight
        

    def prune(self):
        with torch.no_grad():
            self._t = self._t + 1
            for name, parameter in self._model.named_parameters():
                if any(name == one for one in self._prune_dict):
                    weight = self._get_weight(parameter)
                    if self._update_mask_conditions():
                        weight = weight * self._mask[name]
                        target_sparsity = self._prune_dict[name]
                        current_sparse_step = (
                            self._t - self._pretrain_step
                        ) // self._frequency
                        total_srarse_step = self._sparse_step // self._frequency
                        current_sparsity = (
                            target_sparsity
                            + (self._initial_sparsity[name] - target_sparsity)
                            * (1.0 - current_sparse_step / total_srarse_step) ** 3
                        )
                        ####################################
                        keep_k = int(
                            weight.numel() * (1.0 - current_sparsity)
                        )
                        # print('current_sparsity, name, weight.shape: ', current_sparsity, name, weight.cpu().numpy().shape)
                        ##########################################
                        if self._deploy_device == "none":
                            mask = self._update_mask(weight, keep_k)
                            self._mask[name] = mask
                        elif self._deploy_device == "asic":
                            cgb = self.set_up_infos['cgb'][name]
                            dtype_ = self.set_up_infos['dtype'][name]
                            # if len(weight.shape) == 4:
                            #     weight_ = weight.permute([2, 3, 1, 0])
                            #     weight_ = weight_.cpu().numpy()
                            #     mask = prune_dim_4(weight=weight_, keep_k=keep_k, dtype=dtype_, cgb=cgb, group_size_value=self._group_size)
                            #     mask = np.transpose(mask, [3, 2, 0, 1])
                            # elif len(weight.shape) == 3:
                            #     weight_ = weight.permute([1,2,0])
                            #     weight_ = weight_.cpu().numpy()
                            #     mask = prune_dim_3(weight=weight_, keep_k_N=keep_k, dtype=dtype_, cgb=cgb, group_size_value=self._group_size)
                            #     mask = np.transpose(mask, [2,0,1])
                            # elif len(weight.shape) == 2:
                            #     tmp_weight = weight.cpu().numpy()
                            #     mask = prune_dim_2(weight=tmp_weight, keep_k=keep_k, dtype_=dtype_, cgb=cgb, group_size_value=self._group_size)
                            # else:
                            #     mask = update_mask(weight, keep_k)
                            
                            if len(weight.shape) == 4:
                                mask = prune_dim_4(weight=weight_, keep_k=keep_k, cgb=cgb, dtype=dtype_, group_size_value=self._group_size)
                            elif len(weight.shape) == 3:
                                weight_ = weight.permute([2,1,0])  # ker, in, out
                                weight_ = weight_.cpu().float().numpy()
                                mask = prune_dim_3(weight=weight_, keep_k_N=keep_k, cgb=cgb, dtype=dtype_, group_size_value=self._group_size)
                            elif len(weight.shape) == 2:
                                weight_ = weight.cpu().float().numpy()
                                mask = prune_dim_2(weight=weight_, keep_k=keep_k, dtype_=dtype_, cgb=cgb, group_size_value=self._group_size)
                            else:
                                mask = update_mask(weight=weight, keep_k=keep_k)
                            mask = torch.as_tensor(data=mask, dtype=weight.dtype, device=weight.device)
                            self._mask[name] = mask
                    parameter.mul_(self._mask[name])


    def sparsity(self):
        total_param = 0
        total_nonezero = 0
        layer_sparse_rate = {}
        for name, parameter in self._model.named_parameters():
            if any(name == one for one in self._prune_dict):
                temp = parameter.data.cpu().float().numpy()
                total_param = total_param + temp.size
                total_nonezero = total_nonezero + numpy.flatnonzero(temp).size
                layer_sparse_rate[name] = 1 - numpy.flatnonzero(temp).size / temp.size
        total_sparse_rate = 1 - total_nonezero / total_param
        return layer_sparse_rate, total_sparse_rate


def quick_debug_cls():
    import torchvision.models as models
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = models.resnet50(pretrained=True)

    step = 10
    prune_dict = {}
    set_up_infos = {"cgb":{}, "dtype":{}}
    for k, v in model.named_parameters():
        if len(v.shape) < 2:
            continue
        if len(v.shape) == 2:
            set_up_infos["cgb"][k] = 512
        else:
            set_up_infos["cgb"][k] = 64
        print(k, v.shape)
        prune_dict[k] = 1. - (1 / 16)
        set_up_infos["dtype"][k] = 'int8'

    prune = Prune(model, step * 0, step * 8, 10, prune_dict, deploy_device="asic", set_up_infos=set_up_infos)
    prune.prune()

    output_dir="/hdd1/tao/eval_software/debug_case/prune_opti_debug_resnet50/"
    tmp_data = torch.randn(1, 3, 224, 224)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,  'resnet50.onnx')
    torch.onnx.export(model, tmp_data, output_path, training=torch.onnx.TrainingMode.TRAINING, do_constant_folding=False)


def quick_debug_nlp():
    bert_path = "/hdd1/tao/prune_params_927/test_framework/frontendcompilerv3/model_and_infos/total/models/nlp/textattack_bert_base_uncased_MRPC#N1#128/traced.pt"
    model = torch.jit.load(bert_path)

    prune_dict = {}
    set_up_infos = {"cgb":{}, "dtype":{}}
    for k, v in model.named_parameters():
        if len(v.shape) < 2:
            continue
        if k == 'bert.pooler.dense.weight' or k == "classifier.weight":
            set_up_infos["cgb"][k] = 256
        else:
            set_up_infos["cgb"][k] = 64
        print(k, v.shape)
        prune_dict[k] = 1. - (1 / 16)
        set_up_infos["dtype"][k] = 'int8'

    step = 10
    prune = Prune(model, step * 0, step * 8, 10, prune_dict, deploy_device="asic", set_up_infos=set_up_infos)
    prune.prune()
    model_save_dir = "/hdd1/tao/eval_software/debug_case/prune_opti_debug_bert/"
    os.makedirs(model_save_dir, exist_ok=True)
    torch.jit.save(model, model_save_dir+"traced.pt")


class yeild_func:
    def __init__(self, params):
        super(yeild_func, self).__init__()
        self.params = params

    def named_parameters(self):
        for k, v in self.params.items():
            if "weight" in k:
                v = torch.as_tensor(data=v.asnumpy(), dtype=torch.float32) if not torch.is_tensor(v) else v
                self.params[k] = v
                yield (k, self.params[k])

    def to_mx_ndarray(self):
        import mxnet as mx
        for k in self.params.keys():
            if "weight" in k:
                self.params[k] = mx.nd.array(self.params[k])


def quick_debug_yolo():
    import mxnet as mx
    model_path = "/hdd1/tao/prune_params_927/test_framework/frontendcompilerv3/model_and_infos/total/models/detection/yolo3_darknet53_coco#N1#256/yolo3_darknet53_coco"
    mx_sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch=0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = yeild_func(arg_params)

    prune_dict = {}
    set_up_infos = {"cgb":{}, "dtype":{}}
    for k, v in model.named_parameters():
        if len(v.shape) < 2:
            continue
        if len(v.shape) == 2:
            set_up_infos["cgb"][k] = 512
        else:
            set_up_infos["cgb"][k] = 64
        print(k, v.shape)
        prune_dict[k] = 1. - (1 / 16)
        set_up_infos["dtype"][k] = 'int8'
    prune = Prune(model, 0, 1, 10, prune_dict, deploy_device="asic", set_up_infos=set_up_infos)
    prune.prune()

    model.to_mx_ndarray()
    output_dir="/hdd1/tao/eval_software/debug_case/prune_opti_debug_yolo_params_json/"
    os.makedirs(output_dir)
    mx.model.save_checkpoint(os.path.join(output_dir, "yolo3_darknet53_coco"), 0, mx_sym, arg_params, aux_params)
