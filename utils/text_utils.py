import re

def clean_caption(caption):
    # 转换为小写
    caption = caption.lower()
    
    # 删除标点符号
    caption = re.sub(r'[^\w\s]', '', caption)
    
    # 删除多余空格
    caption = ' '.join(caption.split())
    
    return caption

def preprocess_captions(caption_file, output_file):
    with open(caption_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过标题行
    header = lines[0]
    processed_lines = [header]
    
    for line in lines[1:]:
        if line.strip():
            img_name, caption = line.strip().split(',', 1)
            clean_cap = clean_caption(caption)
            processed_lines.append(f"{img_name},{clean_cap}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)