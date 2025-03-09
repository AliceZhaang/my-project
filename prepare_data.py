from split_dataset import split_dataset
from utils.text_utils import preprocess_captions

def prepare_dataset():
    # 1. 预处理原始标注文件
    preprocess_captions(
        'data/flickr8k/captions.txt',
        'data/flickr8k/captions_cleaned.txt'
    )
    
    # 2. 分割数据集
    split_dataset('data/flickr8k/captions_cleaned.txt')

if __name__ == '__main__':
    prepare_dataset()