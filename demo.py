import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from model import CLIPModel


class ImageTextRetrieval:
    def __init__(self, model_path, embed_dim, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel(embed_dim=embed_dim).to(self.device)

        # 加载模型权重
        checkpoint = torch.load(model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 初始化tokenizer和图像转换
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def encode_text(self, text):
        # 对输入文本进行编码
        encoding = self.tokenizer(
            text,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            text_features = self.model.text_encoder(input_ids, attention_mask)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    def encode_image(self, image_path):
        # 对输入图像进行编码
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.image_encoder(image)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    def search(self, query_text, image_dir, top_k=10):
        # 编码查询文本
        text_features = self.encode_text(query_text)

        # 获取所有图像特征
        image_features_list = []
        image_paths = []

        for img_name in os.listdir(image_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, img_name)
                image_features = self.encode_image(img_path)
                image_features_list.append(image_features)
                image_paths.append(img_path)

        # 计算相似度并获取top-k结果
        image_features = torch.cat(image_features_list)
        similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(top_k)

        return [(image_paths[idx], values[i].item()) for i, idx in enumerate(indices)]


def visualize_results(query, results):
    # 调整图像布局以适应10张图片
    rows = (len(results) + 4) // 5  # 每行最多显示5张图片
    cols = min(5, len(results))
    
    plt.figure(figsize=(15, 3 * rows))
    plt.suptitle(f'Query: {query}', fontsize=16)

    for i, (img_path, score) in enumerate(results):
        plt.subplot(rows, cols, i + 1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f'Score: {score:.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为标题留出空间
    plt.savefig(f'results_{query.replace(" ", "_")}.png')  # 保存结果图像
    plt.show()  # 显示图像


def main():
    # 测试不同嵌入维度的检索效果
    image_dir = 'data/flickr8k/Images'  # 修正图像目录路径

    embed_dims = [256, 512]
    test_queries = [
        'a man walking on the street',
        'a baby is looking at the camera',
        'a bird is flying',
        'sunset over mountains',
    ]

    for embed_dim in embed_dims:
        print(f'\nTesting with embedding dimension: {embed_dim}')
        model_path = f'checkpoints/best_model_{embed_dim}.pth'

        if not os.path.exists(model_path):
            print(f'Model for dimension {embed_dim} not found. Skipping...')
            continue

        retriever = ImageTextRetrieval(model_path, embed_dim)

        for query in test_queries:
            print(f'\nQuery: {query}')
            results = retriever.search(query, image_dir, top_k=10)  # 修改为10个结果
            
            # 打印结果
            for i, (img_path, score) in enumerate(results):
                print(f"  {i+1}. {os.path.basename(img_path)} (Score: {score:.2f})")
            
            # 可视化结果
            visualize_results(query, results)


if __name__ == '__main__':
    main()
