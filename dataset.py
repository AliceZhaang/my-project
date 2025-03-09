import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms

# 添加文本增强器类
class TextAugmenter:
    def __init__(self, p=0.3):
        self.p = p
        
    def __call__(self, text):
        # 简单实现，如果概率小于p，则不进行增强
        if random.random() > self.p:
            return text
            
        # 简单的文本增强策略
        words = text.split()
        
        # 随机删除一些词（最多删除20%的词）
        if len(words) > 5 and random.random() < 0.3:
            num_to_drop = max(1, int(len(words) * 0.2))
            drop_indices = random.sample(range(len(words)), num_to_drop)
            words = [w for i, w in enumerate(words) if i not in drop_indices]
        
        # 随机重复一些词
        if random.random() < 0.2:
            idx = random.randint(0, len(words) - 1)
            words.insert(idx, words[idx])
        
        # 随机交换相邻词的位置
        if len(words) > 1 and random.random() < 0.2:
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)


class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None, max_length=64):
        self.image_dir = image_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.training = 'train' in caption_file.lower()  # 判断是否为训练集
    
        # 读取图像-文本对
        self.image_captions = []
        # 添加图像ID到描述的映射，用于跟踪每张图像的多个描述
        self.image_to_captions = {}
        
        with open(caption_file, 'r', encoding='utf-8') as f:
            # 跳过标题行（通常是 "image,caption"）
            header = f.readline()
            print(f"标题行: {header.strip()}")  # 用于调试
            
            for line_num, line in enumerate(f, 2):  # 从2开始计数，因为1是标题行
                if line.strip():
                    try:
                        img_name, caption = line.strip().split(',', 1)
                        # 验证文件名格式
                        if not any(img_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                            print(f"警告：第{line_num}行的文件名可能格式不正确：{img_name}")
                            continue
                            
                        # 验证图片文件是否存在
                        img_path = os.path.join(image_dir, img_name)
                        if os.path.exists(img_path):
                            self.image_captions.append((img_name, caption))
                            
                            # 提取图像ID（不包括扩展名）
                            img_id = os.path.splitext(img_name)[0]
                            if img_id not in self.image_to_captions:
                                self.image_to_captions[img_id] = []
                            self.image_to_captions[img_id].append(len(self.image_captions) - 1)  # 存储索引
                        else:
                            print(f"警告：找不到图片文件（第{line_num}行）：{img_path}")
                    except ValueError as e:
                        print(f"警告：第{line_num}行格式不正确：{line.strip()}")
                        print(f"错误信息：{str(e)}")
    
        print(f"成功加载 {len(self.image_captions)} 个有效的图像-文本对")
        print(f"共 {len(self.image_to_captions)} 张不同的图像，平均每张图像有 {len(self.image_captions)/len(self.image_to_captions):.1f} 个描述")

        # 设置图像转换
        if transform is None:
            # 增强数据转换
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # 初始化文本增强器
        self.text_augmenter = TextAugmenter()
        
    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.image_captions)
        
    def __getitem__(self, idx):
        img_name, caption = self.image_captions[idx]
        image_path = os.path.join(self.image_dir, img_name)
        
        # 添加调试信息
        if not os.path.exists(image_path):
            print(f"找不到图片: {image_path}")
            print(f"图片名称: {img_name}")
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 处理文本
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'caption': caption
        }


def get_dataloader(image_dir, caption_file, batch_size=32, shuffle=True, vocab_size=None, num_workers=0):
    dataset = Flickr8kDataset(image_dir, caption_file)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # 使用传入的 num_workers 参数
        pin_memory=True
    )
    
    return dataloader
