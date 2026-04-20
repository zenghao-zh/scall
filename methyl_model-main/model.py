import torch.nn as nn
import torch
import torch.autograd as autograd
# from .modeling import DNN, AveragePooler, BertSinusoidEmbedding, BertRelativeEmbedding, ConvFeatureExtractionModel
from transformers import (
    BertPreTrainedModel
)

from transformers.models.bert.modeling_bert import (
    SequenceClassifierOutput,
    BertEncoder
)


class AveragePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        """
        :param hidden_states: [bsz x len x dim]
        :param attention_mask: [bsz x len]
        """
        attention_mask = attention_mask.unsqueeze(-1)  # [bsz x len x 1]
        # 求和
        sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)  # [bsz x dim]
        # 求每条数据的真实长度
        input_len = torch.sum(attention_mask, 1)  # [bsz x 1]
        # 防止长度为0
        input_len = torch.clamp(input_len, min=1e-9, max=512)
        # 求平均
        avg_embeddings = sum_embeddings / input_len
        # 全连接
        pooled_output = self.dense(avg_embeddings)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class DssmBertModel(BertPreTrainedModel):
    """
    自主模型, 采用双塔的Bert分别对sequence以及signal进行表征
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        dssm_config = config
        dssm_config.num_hidden_layers = config.num_dssm_layers

        self.embed = nn.Embedding(config.vocab_size, config.embed_size)
        self.seq_fc = nn.Linear(config.embed_size+config.base_feat_nums, config.hidden_size)
        self.seq_bert = BertEncoder(dssm_config)

        self.sig_fc = nn.Linear(config.signal_size, config.hidden_size)
        self.sig_bert = BertEncoder(dssm_config)

        concat_config = config
        concat_config.num_hidden_layers = config.num_classify_layers

        self.concat_fc = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.concat_bert = BertEncoder(concat_config)
        self.concat_pooler = AveragePooler(dssm_config)

        self.concat_dropout = nn.Dropout(config.hidden_dropout_prob)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, input_base_feat, input_signals, attention_mask=None, token_type_ids=None) -> SequenceClassifierOutput:
        """
        Args:
            input_base_ids(2d array like [batch_size, seq_len]):
                input base sequence. the left side of dssm
            input_base_feat(3d array like [batch_size, seq_len, feat_num]):
                input base addition feature, the feat_num == self.config.base_feat_nums
            input_signals:((3d array like [batch_size, seq_len*channels]):
                input signals, the right side of dssm
            attention_mask(2d array like [batch_size, seq_len]):
                input base mask sequence(only use in bert model)
            token_type_ids (2d array like [batch_size, seq_len]):
                input base token sequence (only use in bert model)
        """
        batch_size = input_ids.shape[0]
        seq_embed = self.embed(input_ids)
        seq_input = torch.cat([seq_embed, input_base_feat.float()], axis=-1)

        #sequence feature
        seq_bert_out = self.seq_bert(self.seq_fc(seq_input))[0]


        # signal feature
        sig_bert_out = self.sig_bert(self.sig_fc(input_signals.float()))[0]

        #combine feature to softmax
        combine_hidden_input = torch.cat([seq_bert_out, sig_bert_out], axis=-1)
        combine_bert_output = self.concat_bert(self.concat_fc(combine_hidden_input))[0]

        combine_pool_output = self.concat_pooler(combine_bert_output, torch.ones_like(input_ids))

        combine_pool_output = self.concat_dropout(combine_pool_output)

        logits = self.classifier(self.classifier_dropout(combine_pool_output))

        # print ("logits", logits.shape, logits)
        return SequenceClassifierOutput(
            logits=logits
        )
