import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, BertTokenizer
from torchvision.models import ResNet50_Weights
import numpy as np


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim, dropout=0.4):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # 增加更多层的梯度训练
        for name, param in self.resnet.named_parameters():
            if "layer3" in name or "layer4" in name:  # 解冻更多层
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 更复杂的投影头
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.resnet(x)  # 直接返回ResNet的输出


class TextEncoder(nn.Module):
    def __init__(self, embed_dim, dropout=0.4):  # 降低dropout与图像编码器一致
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # 微调更多BERT层
        for name, param in self.bert.named_parameters():
            if "encoder.layer.9" in name or "encoder.layer.10" in name or "encoder.layer.11" in name:
                param.requires_grad = True  # 解冻最后3层
            else:
                param.requires_grad = False

        # 增强文本特征提取，与图像编码器保持一致
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),  # 使用BatchNorm而不是LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)  # 添加LayerNorm与图像编码器一致
        )

    def forward(self, input_ids, attention_mask):
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS]标记的输出作为文本特征
        return self.projection(outputs.last_hidden_state[:, 0, :])


# 在CLIPModel类中添加可学习的温度参数
class CLIPModel(nn.Module):
    def __init__(self, embed_dim, dropout = 0.5):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim, dropout=dropout)
        self.text_encoder = TextEncoder(embed_dim, dropout=dropout)
        
        # 添加可学习的温度参数，初始值为0.07（OpenAI CLIP使用的值）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # 添加温度参数的上下限约束
        self.logit_scale_min = np.log(1/100)
        self.logit_scale_max = np.log(100)
    
    def forward(self, images, input_ids, attention_mask):
        # 获取图像和文本的嵌入向量
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)

        # 归一化特征向量
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算相似度矩阵
        # 应用温度缩放
        self.logit_scale.data = torch.clamp(
            self.logit_scale.data,
            self.logit_scale_min,
            self.logit_scale_max
        )
        
        # 使用温度参数缩放相似度
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text, image_features, text_features


# 统一的损失函数类
class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07, label_smoothing=0.1, hard_weight=1.0):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.hard_weight = hard_weight
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def forward(self, logits_per_image, logits_per_text):
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # 基本的对比损失
        loss_i2t = self.cross_entropy(logits_per_image, labels)
        loss_t2i = self.cross_entropy(logits_per_text, labels)
        
        # 考虑一对多关系的额外损失
        # 这里我们假设每个图像可能对应多个文本描述
        # 通过增加一个额外的损失项来处理这种情况
        
        # 计算图像和文本特征之间的相似度
        sim_i2t = logits_per_image / self.temperature
        sim_t2i = logits_per_text / self.temperature
        
        # 总损失
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss

