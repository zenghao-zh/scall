# Pruner 实验流程

### 1. 确定模型的prune\_dict

该步骤在misc.py 中 generate\_prune\_dict() func内设置。

prune dict的主要意义是指定模型中需要被稀疏的那些权重层。在进行设置的时候，遍历模型中的所有层，并通过键值对的形式对符合要求的层设置其稀疏率。prune\_dict\[name] = sparsity。

并非模型的所有层都会被稀疏化，一般size为(N, N)级别的权重层（主要为卷积层和linear层）会被加入prune\_dict中，逐步被算法稀疏为指定倍率的稀疏层。通常来说，模型的第一层与最后一层由于最接近输入输出，对模型性能有较大影响，会不稀疏或者少稀疏一定倍率。

`prune_dict, setup_info = generate_prune_dict(model,sparsity)`


### 2. 初始化stage wise pruner
stage wise pruner是一个高度解耦的稀疏工具。只需要在train pipe中 正确的导入一个scheduler: from sparseoptimizer.pruning.scheduler import PrunerScheduler 然后正确配置该 scheduler即可。

通常来说，只需要传入unwarpped模型和optimizer，model\_name(用以调用上一步骤中设置prune\_dict的策略)，每个epoch的steps和训练的总steps。它就可以在此模型的训练过程中逐步将模型稀疏化至指定倍率。

stage wise pruner可以通过prune freq和log freq设置模型变稀疏的频率和展示的频率。并通过一个list，例如\[0.5, 0.75, 0.875]，去分stage将模型稀疏至指定稀疏度并finetune，然后存下相应稀疏率的最佳模型。其参数含义和一个简单设置示例如下例所示：

{% code lineNumbers="true" %}
```python
例如可以创建一个类或者专门的config文件如yaml统一管理这些参数。
from stage_wise_pruner import PrunerScheduler
class CFG:
	model = unwrapped model # 传入model
	optimizer = optimizer # 你的优化器 例如torch.optim.Adam
    prune_dict = prune_dict # generate_prune_dict产生的prune_dict
	step_per_epoch = len(trainset) / batch_size # 每个epoch的step数
	num_steps = steps_per_epoch * epochs #训练的step总数
        sparsities = [0.5, 0.75] # 一个存放各阶段稀疏率的list
        log_path = './prune_log' # 你的prune_log存储路径
	prune_freq = 500 # 修改模型稀疏率的频率 (多少step一次)
        log_freq = 500 # loginfo的频率 (多少step一次)
        local_rank = 0 # 多卡训练的情况下，local_rank为当前gpu的index数值，如果是单卡训练，只需要赋值为-1或0
        
pruner = PrunerScheduler(
        model=CFG.model
        optimizer=CFG.optimizer,
        steps_per_epoch=CFG.steps_per_epoch,
        num_steps=CFG.num_steps,
        prune_freq=CFG.prune_freq,
        log_freq=CFG.log_freq,
        sparsities=CFG.sparsities,
        log_path=CFG.log_path,
        rank=CFG.local_rank,
        bank_size=64,
)
```
{% endcode %}

#### 3. mask和更新模型权重
在每次模型权重更新后，都需要调用 `pruner.prune()` 对模型权重进行稀疏和mask

### 4. 更新最佳metric
在每次对模型进行evaluation后，调用`pruner.update_metrics(acc)`，使得模型可以根据metric保存最佳的模型权重
注意：默认传入的metric数值越高越好，如果是数值越低越好，需要在前面加个负号
