import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from model import CLIPModel
from dataset import get_dataloader
import os
import seaborn as sns
from PIL import Image
import matplotlib.patches as mpatches
import matplotlib
from matplotlib.font_manager import FontProperties
import random
from torch.serialization import add_safe_globals
from numpy._core.multiarray import scalar

# 添加NumPy标量到安全全局变量列表
add_safe_globals([scalar])

matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 使用支持英文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def visualize_alignment(model_path, embed_dim, image_dir, caption_file, num_samples=500):
    """
    可视化CLIP模型中图像和文本特征的对齐情况，优化以匹配评估结果
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型
    model = CLIPModel(embed_dim=embed_dim).to(device)

    # 加载检查点 - 使用与evaluate.py相同的加载方式
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型: {model_path}")
        else:
            print("检查点中没有找到model_state_dict")
            return
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return

    model.eval()

    # 加载数据 - 使用与evaluate.py相同的批次大小
    batch_size = 32
    
    # 修改数据加载方式，使用与evaluate.py相同的方式
    dataloader = get_dataloader(
        image_dir=image_dir,
        caption_file=caption_file,
        batch_size=batch_size,
        shuffle=False,  # 使用与evaluate.py相同的shuffle=False
        num_workers=4   # 使用与evaluate.py相同的workers数
    )
    
    # 获取数据集大小信息
    total_samples = len(dataloader.dataset)
    print(f"数据集总样本数: {total_samples}")
    
    # 限制样本数量，但确保包含足够的样本
    max_samples = min(1000, total_samples) if num_samples is None else min(num_samples, total_samples)
    print(f"使用 {max_samples} 个样本进行可视化")
    
    # 收集特征
    all_image_features = []
    all_text_features = []
    all_image_names = []
    all_captions = []
    
    samples_collected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= max_samples:
                break
            
            # 获取当前批次大小
            current_batch_size = len(batch['image'])
            samples_to_take = min(current_batch_size, max_samples - samples_collected)
            
            # 提取特征
            images = batch['image'][:samples_to_take].to(device)
            input_ids = batch['input_ids'][:samples_to_take].to(device)
            attention_mask = batch['attention_mask'][:samples_to_take].to(device)
            
            # 计算特征
            image_features = model.image_encoder(images)
            text_features = model.text_encoder(input_ids, attention_mask)
            
            # 归一化特征
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # 收集特征和元数据
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            
            # 收集图像名称和描述
            for i in range(samples_to_take):
                idx = samples_collected + i
                if idx < len(dataloader.dataset.image_captions):
                    img_name, caption = dataloader.dataset.image_captions[idx]
                    all_image_names.append(img_name)
                    all_captions.append(caption)
            
            samples_collected += samples_to_take
            print(f"已收集 {samples_collected}/{max_samples} 个样本")
    
    # 合并特征
    image_features = torch.cat(all_image_features).numpy()
    text_features = torch.cat(all_text_features).numpy()
    
    print(f"特征形状 - 图像: {image_features.shape}, 文本: {text_features.shape}")
    
    # 创建图像ID到索引的映射
    image_ids = []
    image_id_to_indices = {}
    
    for i, img_name in enumerate(all_image_names):
        img_id = os.path.splitext(img_name)[0]
        
        if img_id not in image_id_to_indices:
            image_id_to_indices[img_id] = []
            image_ids.append(img_id)
        
        image_id_to_indices[img_id].append(i)
    
    print(f"共有 {len(image_ids)} 个唯一图像")
    
    # 选择唯一图像的特征
    unique_image_indices = []
    for img_id in image_ids:
        if image_id_to_indices[img_id]:
            unique_image_indices.append(image_id_to_indices[img_id][0])
    
    # 限制唯一图像数量，避免t-SNE过慢
    max_unique_images = 100
    if len(unique_image_indices) > max_unique_images:
        print(f"限制唯一图像数量为 {max_unique_images}")
        unique_image_indices = unique_image_indices[:max_unique_images]
    
    unique_image_features = image_features[unique_image_indices]
    
    # 为每个唯一图像选择对应的文本特征
    text_features_by_image = []
    text_to_image_map = []  # 记录每个文本对应的图像索引
    
    for i, idx in enumerate(unique_image_indices):
        img_id = os.path.splitext(all_image_names[idx])[0]
        text_indices = image_id_to_indices[img_id]
        
        # 为每个图像最多选择5个文本描述
        for text_idx in text_indices[:5]:  # 限制每个图像最多5个描述
            if text_idx < len(text_features):
                text_features_by_image.append(text_features[text_idx])
                text_to_image_map.append(i)
    
    text_features_by_image = np.array(text_features_by_image)
    
    print(f"使用 {len(unique_image_features)} 个唯一图像和 {len(text_features_by_image)} 个文本描述进行可视化")
    
    # 使用t-SNE进行降维 - 优化参数
    print("正在使用t-SNE进行降维...")
    features = np.vstack([unique_image_features, text_features_by_image])
    
    # 使用更适合的参数
    perplexity = min(40, len(features) // 4)  # 增加perplexity
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=perplexity,
        early_exaggeration=12.0,  # 增加early_exaggeration
        learning_rate='auto',     # 使用自动学习率
        n_iter=10000,             # 增加迭代次数
        n_iter_without_progress=500,
        metric='cosine'           # 使用余弦距离，与模型使用的相似度度量一致
    )
    features_tsne = tsne.fit_transform(features)
    
    # 分离图像和文本特征
    image_features_tsne = features_tsne[:len(unique_image_features)]
    text_features_tsne = features_tsne[len(unique_image_features):]
    
    # 可视化1：t-SNE散点图 - 使用英文标题和标签
    plt.figure(figsize=(12, 10))
    
    # 使用更明显的颜色和更大的点
    plt.scatter(image_features_tsne[:, 0], image_features_tsne[:, 1], 
                c='blue', label='Image Features', alpha=0.7, s=120, edgecolor='white')
    plt.scatter(text_features_tsne[:, 0], text_features_tsne[:, 1], 
                c='red', label='Text Features', alpha=0.7, s=60, edgecolor='white')
    
    # 只为部分匹配对绘制连线，避免过度拥挤
    max_lines = 200
    if len(text_to_image_map) > max_lines:
        line_indices = random.sample(range(len(text_to_image_map)), max_lines)
    else:
        line_indices = range(len(text_to_image_map))
    
    for i in line_indices:
        img_idx = text_to_image_map[i]
        if i < len(text_features_tsne) and img_idx < len(image_features_tsne):
            plt.plot([image_features_tsne[img_idx, 0], text_features_tsne[i, 0]],
                    [image_features_tsne[img_idx, 1], text_features_tsne[i, 1]],
                    'k-', alpha=0.15)
    
    plt.title('t-SNE Visualization of Image and Text Features (One-to-Many)', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('alignment_tsne.png', dpi=300)
    plt.show()
    
    # 修改可视化2：相似度热力图
    plt.figure(figsize=(15, 8))  # 调整图像比例以适应矩形显示
    
    # 计算图像和文本特征之间的相似度矩阵
    similarity = unique_image_features @ text_features_by_image.T
    
    # 移除坐标轴标签（数据量大时更清晰）
    plt.tick_params(axis='both', which='both', length=0)
    plt.tick_params(labelbottom=False, labelleft=False)
    
    # 使用改进的热力图参数
    im = plt.imshow(similarity, 
                    aspect='auto',  # 自动调整宽高比
                    cmap='RdYlBu_r',  # 使用更好的颜色方案
                    vmin=-0.4,
                    vmax=0.8)
    
    # 添加颜色条并调整其位置和大小
    cbar = plt.colorbar(im, pad=0.01)
    cbar.set_label('Similarity', fontsize=12)
    
    plt.title('Image-Text Similarity Matrix', fontsize=16, pad=20)
    plt.xlabel('Text', fontsize=14, labelpad=10)
    plt.ylabel('Image', fontsize=14, labelpad=10)
    
    # 调整布局以避免裁剪
    plt.tight_layout()
    
    # 保存高质量图像
    plt.savefig('similarity_matrix.png', 
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.show()
    
    # 可视化3：图像-文本对示例
    plt.figure(figsize=(20, 18))
    
    # 随机选择3个图像
    sample_indices = random.sample(range(len(unique_image_indices)), min(3, len(unique_image_indices)))
    
    for i, idx in enumerate(sample_indices):
        img_idx = unique_image_indices[idx]
        img_name = all_image_names[img_idx]
        img_id = os.path.splitext(img_name)[0]
        
        # 显示图像
        plt.subplot(3, 2, i * 2 + 1)
        try:
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            plt.imshow(img)
            plt.title(f"Image {i+1}", fontsize=14, pad=10)
            plt.axis('off')
            
            # 显示文本和相似度
            plt.subplot(3, 2, i * 2 + 2)
            
            # 获取当前图像的所有原始描述
            text_indices = image_id_to_indices[img_id]
            original_captions = [all_captions[idx] for idx in text_indices if idx < len(all_captions)]
            
            if original_captions:
                # 显示原始描述
                plt.text(0.02, 0.98, "Original Descriptions:", fontsize=13, fontweight='bold')
                for j, caption in enumerate(original_captions[:5]):
                    plt.text(0.02, 0.90 - j*0.07, f"{j+1}. {caption}", 
                            wrap=True, fontsize=11)
                
                # 计算相似度并获取Top-5匹配
                img_feature = image_features[img_idx].reshape(1, -1)
                similarities = img_feature @ text_features.T
                
                if similarities.shape[1] > 0:
                    top_matches = np.argsort(similarities[0])[::-1][:5]
                    
                    # 显示Top-5检索结果
                    plt.text(0.02, 0.50, "Top-5 Retrieved Descriptions:", 
                            fontsize=13, fontweight='bold')
                    
                    for j, match_idx in enumerate(top_matches):
                        if match_idx < len(all_captions):
                            match_caption = all_captions[match_idx]
                            similarity_score = similarities[0][match_idx]
                            color = 'green' if match_idx in text_indices else 'red'
                            plt.text(0.02, 0.42 - j*0.07,
                                    f"Top-{j+1} (sim={similarity_score:.3f}): {match_caption}",
                                    wrap=True, fontsize=11, color=color)
        
            plt.axis('off')
        except Exception as e:
            print(f"Error displaying image {i}: {e}")
    
    plt.tight_layout(h_pad=2.0, w_pad=1.0)
    plt.savefig('image_text_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 可视化4：相似度分布 - 使用英文标题和标签
    plt.figure(figsize=(12, 8))
    
    # 计算匹配对和非匹配对的相似度
    matching_similarities = []
    non_matching_similarities = []
    
    for i in range(len(unique_image_features)):
        # 找出属于当前图像的所有文本索引
        matching_indices = [j for j, img_idx in enumerate(text_to_image_map) if img_idx == i]
        non_matching_indices = [j for j, img_idx in enumerate(text_to_image_map) if img_idx != i]
        
        # 收集匹配对的相似度
        for idx in matching_indices:
            if idx < similarity.shape[1]:
                matching_similarities.append(similarity[i, idx])
        
        # 收集非匹配对的相似度（随机采样以避免数据过多）
        if non_matching_indices:
            sample_size = min(len(non_matching_indices), 20)  # 增加采样数量
            sampled_indices = random.sample(non_matching_indices, sample_size)
            for idx in sampled_indices:
                if idx < similarity.shape[1]:
                    non_matching_similarities.append(similarity[i, idx])
    
    # 确保有数据可以绘制
    if matching_similarities and non_matching_similarities:
        matching_similarities = np.array(matching_similarities)
        non_matching_similarities = np.array(non_matching_similarities)
        
        # 计算平均值和标准差
        matching_mean = np.mean(matching_similarities)
        non_matching_mean = np.mean(non_matching_similarities)
        matching_std = np.std(matching_similarities)
        non_matching_std = np.std(non_matching_similarities)
        
        # 绘制分布
        bins = np.linspace(-0.6, 0.9, 30)  # 使用固定的bins范围提高可比性
        
        plt.hist(matching_similarities, bins=bins, color='green', alpha=0.7, 
                 label=f'Matching Pairs (mean={matching_mean:.3f}, std={matching_std:.3f})')
        plt.hist(non_matching_similarities, bins=bins, color='red', alpha=0.7, 
                 label=f'Non-matching Pairs (mean={non_matching_mean:.3f}, std={non_matching_std:.3f})')
        
        plt.title('Similarity Distribution of Matching vs Non-matching Pairs', fontsize=16)
        plt.xlabel('Cosine Similarity', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('similarity_distribution.png', dpi=300)
        plt.show()
        
        # 打印相似度统计信息
        print("\nSimilarity Statistics:")
        print(f"Matching Pairs - Mean: {matching_mean:.4f}, Median: {np.median(matching_similarities):.4f}, Std: {matching_std:.4f}")
        print(f"Non-matching Pairs - Mean: {non_matching_mean:.4f}, Median: {np.median(non_matching_similarities):.4f}, Std: {non_matching_std:.4f}")
        
        # 计算分离度指标
        separation = (matching_mean - non_matching_mean) / ((matching_std + non_matching_std) / 2)
        print(f"Separation Index: {separation:.4f} (higher is better)")
    else:
        print("Not enough data for similarity distribution visualization")
    
    # 可视化5：Top-K准确率图表 - 与评估结果对比
    # 修改Top-K准确率图表部分
    plt.figure(figsize=(10, 6))
    
    # 计算Top-K准确率
    k_values = [1, 5, 10, 20, 50]
    i2t_accuracy = []
    t2i_accuracy = []
    
    # 创建与evaluate.py相同的映射关系
    idx_to_image_id = {}
    for i, img_name in enumerate(all_image_names):
        img_id = os.path.splitext(img_name)[0]
        idx_to_image_id[i] = img_id
    
    # 获取图像ID到描述索引的映射
    image_to_captions = {}
    for i, (img_name, _) in enumerate(dataloader.dataset.image_captions):
        img_id = os.path.splitext(img_name)[0]
        if img_id not in image_to_captions:
            image_to_captions[img_id] = []
        image_to_captions[img_id].append(i)
    
    # 计算完整的相似度矩阵
    full_similarity = image_features @ text_features.T
    
    # 计算图像到文本的准确率，使用与evaluate.py相同的方法
    for k in k_values:
        i2t_correct = 0
        image_ids_used = set()
        
        for i in range(len(image_features)):
            img_id = idx_to_image_id[i]
            if img_id in image_to_captions and img_id not in image_ids_used:
                image_ids_used.add(img_id)
                # 获取当前图像的所有正确描述索引
                correct_indices = image_to_captions[img_id]
                # 获取预测的前k个文本
                predictions = np.argsort(full_similarity[i])[::-1][:k]
                # 如果任何一个正确描述在预测中，就算正确
                if any(idx in predictions for idx in correct_indices if idx < len(text_features)):
                    i2t_correct += 1
        
        if image_ids_used:
            i2t_accuracy.append(i2t_correct / len(image_ids_used) * 100)
        else:
            i2t_accuracy.append(0)
    
    # 计算文本到图像的准确率，使用与evaluate.py相同的方法
    for k in k_values:
        t2i_correct = 0
        total = 0
        
        for i in range(len(text_features)):
            if i < len(all_image_names):
                total += 1
                text_img_id = idx_to_image_id[i]
                # 获取当前文本对应的正确图像ID
                correct_img_ids = [idx_to_image_id[idx] for idx in image_to_captions.get(text_img_id, [])]
                # 获取预测的前k个图像
                predictions = np.argsort(full_similarity.T[i])[::-1][:k]
                predicted_img_ids = [idx_to_image_id[idx] for idx in predictions if idx < len(image_features)]
                # 如果任何一个正确图像ID在预测中，算正确
                if any(img_id in predicted_img_ids for img_id in correct_img_ids):
                    t2i_correct += 1
        
        if total > 0:
            t2i_accuracy.append(t2i_correct / total * 100)
        else:
            t2i_accuracy.append(0)
    
    # 绘制Top-K准确率图表
    plt.plot(k_values, i2t_accuracy, 'o-', color='blue', label='Image→Text')
    plt.plot(k_values, t2i_accuracy, 's-', color='red', label='Text→Image')
    
    # 添加评估结果的参考点
    eval_k = [1, 5, 10]
    eval_i2t = [30.99, 57.83, 69.75]  # 从evaluate.py输出获取
    eval_t2i = [27.95, 46.17, 58.44]  # 从evaluate.py输出获取
    
    plt.plot(eval_k, eval_i2t, 'o', color='darkblue', markersize=10, label='Eval Results (Image→Text)')
    plt.plot(eval_k, eval_t2i, 's', color='darkred', markersize=10, label='Eval Results (Text→Image)')
    
    # 使用纯英文标题
    plt.title('Top-K Retrieval Accuracy', fontsize=16)
    plt.xlabel('K', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks(k_values)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('topk_accuracy.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    model_path = "e:/pycharm/projects/miniclip/checkpoints/best_model_256.pth"
    image_dir = "e:/pycharm/projects/miniclip/data/flickr8k/Images"
    caption_file = "e:/pycharm/projects/miniclip/data/flickr8k/test_captions.txt"
    visualize_alignment(model_path, 256, image_dir, caption_file, num_samples=500)
