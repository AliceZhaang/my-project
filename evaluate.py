import numpy
import torch
import torch.nn as nn
from tqdm import tqdm
from model import CLIPModel
from dataset import get_dataloader
import numpy as np
import os 
from torch.serialization import add_safe_globals
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_model(model_path, image_dir, test_caption_file, embed_dim=512, batch_size=32):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    # 加载模型
    model = CLIPModel(embed_dim=embed_dim).to(device)
   
    # 使用 weights_only=False（如果你信任模型文件）
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'成功加载模型: {model_path}')

    # 获取测试数据加载器
    test_loader = get_dataloader(
        image_dir=image_dir,
        caption_file=test_caption_file,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f'测试集大小: {len(test_loader.dataset)} 个样本')

    # 评估指标
    total_loss = 0
    all_image_embeddings = []
    all_text_embeddings = []
    
    print('开始评估...')
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='评估进度'):
            # 将数据移到设备上
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 获取图像和文本的特征
            image_features = model.image_encoder(images)
            text_features = model.text_encoder(input_ids, attention_mask)

            # 归一化特征
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # 计算相似度矩阵
            logits = model.logit_scale.exp() * image_features @ text_features.t()
            
            # 计算损失
            labels = torch.arange(len(logits)).to(device)
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()

            # 保存特征用于计算准确率
            all_image_embeddings.append(image_features.cpu())
            all_text_embeddings.append(text_features.cpu())

    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    print(f'\n测试集平均损失: {avg_loss:.4f}')

    # 计算图像到文本和文本到图像的检索准确率
    all_image_embeddings = torch.cat(all_image_embeddings)
    all_text_embeddings = torch.cat(all_text_embeddings)
    
    # 计算完整的相似度矩阵
    similarity = all_image_embeddings @ all_text_embeddings.t()
    
    # 计算 R@K 指标，考虑每张图像有多个描述
    print('计算检索指标...')
    
    # 获取图像ID到描述索引的映射
    dataset = test_loader.dataset
    image_ids = list(dataset.image_to_captions.keys())
    
    # 创建图像索引到图像ID的映射
    idx_to_image_id = {}
    for i, (img_name, _) in enumerate(dataset.image_captions):
        img_id = os.path.splitext(img_name)[0]
        idx_to_image_id[i] = img_id
    
    # 计算 R@K 指标
    for k in [1, 5, 10]:
        # 图像到文本的检索
        i2t_correct = 0
        unique_images = set(idx_to_image_id.values())
        total_unique_images = len(unique_images)
        
        # 只对唯一图像进行评估
        for img_id in unique_images:
            # 找到该图像的第一个索引
            image_idx = next(i for i, id_ in idx_to_image_id.items() if id_ == img_id)
            correct_indices = dataset.image_to_captions[img_id]
            predictions = similarity[image_idx].topk(k)[1].cpu().numpy()
            if any(idx in predictions for idx in correct_indices):
                i2t_correct += 1
        
        i2t_recall = (i2t_correct / total_unique_images) * 100

        # 文本到图像的检索
        t2i_correct = 0
        for i in range(len(similarity.t())):
            text_img_id = idx_to_image_id[i]
            correct_img_id = text_img_id  # 每个文本只对应一个正确的图像
            predictions = similarity.t()[i].topk(k)[1].cpu().numpy()
            predicted_img_ids = [idx_to_image_id[idx] for idx in predictions]
            if correct_img_id in predicted_img_ids:
                t2i_correct += 1
        
        t2i_recall = t2i_correct / len(similarity.t()) * 100
        
        print(f'R@{k}:')
        print(f'  图像→文本: {i2t_recall:.2f}%')
        print(f'  文本→图像: {t2i_recall:.2f}%')


def main():
    # 测试不同嵌入维度的模型
    image_dir = 'data/flickr8k/images'
    test_caption_file = 'data/flickr8k/test_captions.txt'
    embed_dims = [256, 512]

    for embed_dim in embed_dims:
        print(f'\n评估嵌入维度为 {embed_dim} 的模型')
        model_path = f'checkpoints/best_model_{embed_dim}.pth'
        
        try:
            evaluate_model(
                model_path=model_path,
                image_dir=image_dir,
                test_caption_file=test_caption_file,
                embed_dim=embed_dim
            )
        except Exception as e:
            print(f'评估失败: {str(e)}')


if __name__ == '__main__':
    main()