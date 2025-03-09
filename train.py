import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import CLIPModel, CLIPLoss
from dataset import get_dataloader
from torch.amp import autocast, GradScaler  # 修改导入语句
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


def calculate_retrieval_metrics(model, val_loader, device, k_values=None):
    if k_values is None:
        k_values = [1, 5, 10]
    model.eval()
    
    # 收集所有图像和文本特征
    all_image_features = []
    all_text_features = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Computing metrics'):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 计算特征
            image_features = model.image_encoder(images)
            text_features = model.text_encoder(input_ids, attention_mask)
            
            # 归一化特征
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # 收集特征
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
    
    # 将所有特征连接起来
    all_image_features = torch.cat(all_image_features)
    all_text_features = torch.cat(all_text_features)
    
    # 计算完整的相似度矩阵
    similarity = all_image_features @ all_text_features.t()
    
    # 获取数据集信息
    dataset = val_loader.dataset
    
    # 获取图像ID到描述索引的映射
    image_ids = list(dataset.image_to_captions.keys())
    
    # 创建图像索引到图像ID的映射
    idx_to_image_id = {}
    for i, (img_name, _) in enumerate(dataset.image_captions):
        img_id = os.path.splitext(img_name)[0]
        idx_to_image_id[i] = img_id
    
    # 计算指标
    metrics = {}
    for k in k_values:
        # 图像到文本的检索 - 修改为与evaluate.py一致
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
        metrics[f'R@{k}_i2t'] = i2t_recall
        
        # 文本到图像的检索 - 修改为与evaluate.py一致
        t2i_correct = 0
        for i in range(len(similarity.t())):
            text_img_name, _ = dataset.image_captions[i]
            text_img_id = os.path.splitext(text_img_name)[0]  # 当前文本对应的正确图像ID
            
            predictions = similarity.t()[i].topk(k)[1].cpu().numpy()
            predicted_img_ids = [idx_to_image_id[idx] for idx in predictions]
            
            if text_img_id in predicted_img_ids:
                t2i_correct += 1
        
        t2i_recall = t2i_correct / len(similarity.t()) * 100
        metrics[f'R@{k}_t2i'] = t2i_recall
        
    return metrics


def train(model, train_loader, val_loader, num_epochs, device, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    
    # 修改优化器配置 - 提高初始学习率
    optimizer = optim.AdamW([
        {'params': model.image_encoder.resnet.layer4.parameters(), 'lr': 5e-4},  # 提高学习率
        {'params': model.image_encoder.resnet.fc.parameters(), 'lr': 5e-4},      # 提高学习率
        {'params': model.text_encoder.parameters(), 'lr': 5e-4},                 # 提高学习率
        {'params': [model.logit_scale], 'lr': 1e-3}
    ], weight_decay=0.01)
    
    # 优化学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[5e-4, 5e-4, 5e-4, 1e-3],  # 匹配上面的学习率
        total_steps=num_epochs * len(train_loader),
        pct_start=0.1,  # 10%的步数用于预热
        div_factor=10,  # 初始学习率 = max_lr/10 (提高初始值)
        final_div_factor=1000  # 最终学习率 = max_lr/1000
    )
    
    # 优化混合精度训练设置
    scaler = GradScaler(
        init_scale=2**10,        # 初始缩放因子
        growth_factor=2,         # 增长因子
        backoff_factor=0.5,      # 回退因子
        growth_interval=100      # 增长间隔
    )
    best_loss = float('inf')
    
    # 添加标签平滑
    label_smoothing = 0.15  # 可以调整这个值，通常在0.05-0.2之间
    
    # 添加指标记录
    best_r1 = 0
    metrics_history = {
        'train_losses': [],
        'val_losses': [],
        'R@1_i2t': [],
        'R@1_t2i': [],
        'R@5_i2t': [],
        'R@5_t2i': [],
        'R@10_i2t': [],
        'R@10_t2i': [],
        'learning_rates': []
    }
    
    # 初始化新的损失函数
    criterion = CLIPLoss(
        temperature=0.07,
        label_smoothing=label_smoothing,
        hard_weight=2.0
    ).to(device)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        if 'learning_rates' not in metrics_history:
            metrics_history['learning_rates'] = []
        metrics_history['learning_rates'].append(current_lr)
        print(f"当前学习率: {current_lr:.6f}")

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            images = batch['image'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # 使用混合精度训练 - 已经正确实现
            with autocast(device_type='cuda', dtype=torch.float16):
                # 获取模型输出
                logits_per_image, logits_per_text, image_features, text_features = model(images, input_ids, attention_mask)
                
                # 直接计算损失
                loss = criterion(logits_per_image=logits_per_image, logits_per_text=logits_per_text)

            # 添加梯度裁剪，提高训练稳定性
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 每个batch后更新学习率
            scheduler.step()

            train_loss += loss.item()
            train_steps += 1
            
        avg_train_loss = train_loss / train_steps

        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        all_image_features = []
        all_text_features = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(device, non_blocking=True)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.float16):
                    # 获取特征
                    image_features = model.image_encoder(images)
                    text_features = model.text_encoder(input_ids, attention_mask)
                    
                    # 归一化特征
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    
                    # 计算相似度矩阵
                    logits_per_image = model.logit_scale.exp() * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()
                    
                    # 根据损失函数类型计算验证损失
                    loss = criterion(
                        logits_per_image=logits_per_image,
                        logits_per_text=logits_per_text
                    )

                val_loss += loss.item()
                val_steps += 1
                
                # 收集特征用于计算指标
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())

        # 计算平均验证损失
        avg_val_loss = val_loss / val_steps
        
        # 在验证阶段修改调用方式
        # 计算检索指标
        all_image_features = torch.cat(all_image_features).to(device)
        all_text_features = torch.cat(all_text_features).to(device)
        metrics = calculate_retrieval_metrics(model, val_loader, device)
        
        # 更新最佳模型（基于R@1指标）
        current_r1 = (metrics['R@1_i2t'] + metrics['R@1_t2i']) / 2
        if current_r1 > best_r1:
            best_r1 = current_r1
            # 添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # 先保存到临时文件
                    temp_path = os.path.join(save_dir, 'temp_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'metrics': metrics,
                    }, temp_path)
                    
                    # 使用更安全的方式获取嵌入维度
                    try:
                        embed_dim = model.image_encoder.resnet.fc[-2].out_features
                    except (AttributeError, IndexError):
                        embed_dim = getattr(model, 'embed_dim', 256)
                    
                    # 如果保存成功，再重命名为目标文件
                    final_path = os.path.join(save_dir, f'best_model_{embed_dim}.pth')
                    if os.path.exists(final_path):
                        os.remove(final_path)
                    os.rename(temp_path, final_path)
                    break
                except Exception as e:
                    print(f"保存模型失败，尝试次数 {attempt + 1}/{max_retries}")
                    print(f"错误信息: {str(e)}")
                    if attempt == max_retries - 1:
                        print("达到最大重试次数，跳过保存")
                        
        # 记录指标
        metrics_history['train_losses'].append(avg_train_loss)
        metrics_history['val_losses'].append(avg_val_loss)
        for k, v in metrics.items():
            if k in metrics_history:
                metrics_history[k].append(v)

        # 打印训练信息
        print(f'Epoch {epoch + 1}:')
        print(f'  Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        print('  Retrieval Metrics:')
        for k, v in metrics.items():
            print(f'    {k}: {v:.2f}%' if 'R@' in k or 'NDCG' in k else f'    {k}: {v:.2f}')

        # 修改这部分代码，使用更安全的方式获取嵌入维度
        try:
            # 尝试从模型配置中获取嵌入维度
            embed_dim = model.image_encoder.resnet.fc[-2].out_features
        except (AttributeError, IndexError):
            # 如果上面的方法失败，使用默认值或从模型实例获取
            embed_dim = getattr(model, 'embed_dim', 256)
        
        # 保存训练历史
        torch.save({
            'train_losses': metrics_history['train_losses'],
            'val_losses': metrics_history['val_losses'],
            'r1_i2t': metrics_history['R@1_i2t'],
            'r1_t2i': metrics_history['R@1_t2i'],
            'r5_i2t': metrics_history['R@5_i2t'],
            'r5_t2i': metrics_history['R@5_t2i'],
            'r10_i2t': metrics_history['R@10_i2t'],
            'r10_t2i': metrics_history['R@10_t2i'],
        }, f'training_history_{embed_dim}.pth')
        
        # 绘制训练曲线
        try:
            plot_training_curves(metrics_history, save_dir, embed_dim)
        except Exception as e:
            print(f"绘制训练曲线失败: {str(e)}")


def plot_training_curves(history, save_dir, embed_dim):
    # 创建多个子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    ax1.plot(history['train_losses'], label='Training Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # R@1曲线
    ax2.plot(history['R@1_i2t'], label='Image→Text')
    ax2.plot(history['R@1_t2i'], label='Text→Image')
    ax2.set_title('R@1 Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Recall (%)')
    ax2.legend()
    ax2.grid(True)
    
    # R@5曲线
    ax3.plot(history['R@5_i2t'], label='Image→Text')
    ax3.plot(history['R@5_t2i'], label='Text→Image')
    ax3.set_title('R@5 Curves')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Recall (%)')
    ax3.legend()
    ax3.grid(True)
    
    # R@10曲线
    ax4.plot(history['R@10_i2t'], label='Image→Text')
    ax4.plot(history['R@10_t2i'], label='Text→Image')
    ax4.set_title('R@10 Curves')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.suptitle(f'Training Progress (Embedding Dim: {embed_dim})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_{embed_dim}.png'))
    plt.close()
    
    # 额外绘制学习率曲线
    if 'learning_rates' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['learning_rates'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'lr_schedule_{embed_dim}.png'))
        plt.close()


def main():
    # Enable CUDA error debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'使用GPU: {torch.cuda.get_device_name(0)}')
        # 设置cudnn基准模式以加速训练
        torch.backends.cudnn.benchmark = True
        # Enable deterministic operations for better debugging
        torch.backends.cudnn.deterministic = True
    else:
        print('未检测到GPU，使用CPU训练')

    # 数据集路径
    image_dir = 'data/flickr8k/Images'
    train_caption_file = 'data/flickr8k/train_captions.txt'
    val_caption_file = 'data/flickr8k/val_captions.txt'

    # 实验不同的嵌入维度
    embed_dims = [256, 512]

    try:
        for embed_dim in embed_dims:
            print(f'\nTraining with embedding dimension: {embed_dim}')

            # 初始化模型
            model = CLIPModel(embed_dim=embed_dim, dropout=0.5).to(device)
            
            # 获取 BERT 词表大小
            vocab_size = model.text_encoder.bert.config.vocab_size
            print(f"BERT 词表大小: {vocab_size}")

            # 获取数据加载器，传入 vocab_size
            # 在main函数中
            train_loader = get_dataloader(image_dir, train_caption_file, batch_size=256,  # 增加批量大小
            shuffle=True, vocab_size=vocab_size)
            val_loader = get_dataloader(image_dir, val_caption_file, batch_size=128,
            shuffle=False, vocab_size=vocab_size)

            # 训练模型
            train(model, train_loader, val_loader, num_epochs=30, device=device)
            
    except RuntimeError as e:
        print(f"CUDA Error encountered: {str(e)}")
        print("Try reducing batch size or model size if memory error occurs")
        raise  # Re-raise the exception for full traceback


if __name__ == '__main__':
    main()
