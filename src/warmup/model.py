import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import BertConfig, BertModel, AutoConfig, AutoModel


class SentencePairModel(nn.Module):
    def __init__(self, pretrained_model=None, args=None):
        super().__init__()
        if pretrained_model is None:
            self.config = AutoConfig.from_pretrained(args.pretrained_path)
            self.ptm = AutoModel.from_config(self.config)
        else:
            self.ptm = pretrained_model

        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(self.ptm.config.hidden_size, 1)

    def train_step(self, target=None, inputs_embeds=None, input_ids=None, attention_mask=None,
                   token_type_ids=None, length=None):

        input_ids = input_ids.view(-1, input_ids.shape[-1]).squeeze()
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]).squeeze()
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]).squeeze()

        each_length = length.shape[-1]
        length = length.view(-1, 1).squeeze()
        target = target.view(-1, 1).squeeze()

        outputs = self.ptm(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           inputs_embeds=inputs_embeds)

        cls_embedding = outputs[0][:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        # calculate loss based on length
        loss_fct = CrossEntropyLoss()
        ptr = 0
        loss = 0

        loss_cnt = 0

        for i, set_len in enumerate(length):
            if set_len == 0:
                continue
            elif i + 1 < len(length) and length[i + 1] == 0 and i != 0:
                # cause padding will add more zero
                ptr += set_len
                assert target[ptr: ptr + set_len].sum().item() == 0, "sum: " + str(
                    target[ptr: ptr + set_len].sum().item())
                continue
            elif (i + 1) % each_length == 0 and target[ptr: ptr + set_len].sum().item() <= 0:
                ptr += set_len
                continue

            assert target[ptr: ptr + set_len].sum().item() == 1, f"sum: {target[ptr: ptr + set_len].sum().item(), i}"

            target_idx = torch.argmax(target[ptr: ptr + set_len]).unsqueeze(0)
            assert target_idx.squeeze().item() <= set_len, "target index exceeds index: " + str(
                target_idx.squeeze().item())

            softmax_logits = F.softmax(logits[ptr: ptr + set_len].view(1, -1), dim=-1)

            assert softmax_logits.shape[-1] == set_len, "softmax not equal"

            loss += loss_fct(softmax_logits, target_idx)

            ptr += set_len
            loss_cnt += 1
        if loss_cnt == 0:
            print("[Model] len=0, logits:", logits.shape, "length:", length)
            loss_cnt = 0.0001

        loss /= loss_cnt

        return {'pred': logits, 'loss': loss}

    def evaluate_step(self, target=None, inputs_embeds=None, input_ids=None, attention_mask=None,
                      token_type_ids=None, length=None):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])

        each_length = length.shape[-1]
        length = length.view(-1, 1)
        target = target.view(-1, 1)

        outputs = self.ptm(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           inputs_embeds=inputs_embeds)

        cls_embedding = outputs[0][:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        # calculate loss based on length
        assert target.sum().item() == 1, "evaluate not consistent"

        loss_fct = CrossEntropyLoss()
        softmax_logits = F.softmax(logits.view(1, -1), dim=-1)
        target_idx = torch.argmax(target).unsqueeze(0)
        loss = loss_fct(softmax_logits, target_idx)

        return {'pred': logits, 'loss': loss}
