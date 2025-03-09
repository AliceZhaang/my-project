import pandas as pd
import numpy as np

def split_dataset(caption_file, train_ratio=0.8, val_ratio=0.1):
    # 读取原始数据
    df = pd.read_csv(caption_file)
    
    # 获取唯一的图片名称
    unique_images = df['image'].unique()
    np.random.shuffle(unique_images)
    
    # 计算分割点
    n = len(unique_images)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    # 分割数据集
    train_images = unique_images[:train_idx]
    val_images = unique_images[train_idx:val_idx]
    test_images = unique_images[val_idx:]
    
    # 根据图片名称分配数据
    train_df = df[df['image'].isin(train_images)]
    val_df = df[df['image'].isin(val_images)]
    test_df = df[df['image'].isin(test_images)]
    
    # 保存分割后的数据
    train_df.to_csv('data/flickr8k/train_captions.txt', index=False)
    val_df.to_csv('data/flickr8k/val_captions.txt', index=False)
    test_df.to_csv('data/flickr8k/test_captions.txt', index=False)

if __name__ == '__main__':
    split_dataset('data/flickr8k/captions.txt')