import argparse
import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# 导入 SAM 3 模块
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import COLORS, plot_mask, plot_bbox

def parse_args():
    parser = argparse.ArgumentParser(description="SAM 3 批量图片推理脚本")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出结果目录路径")
    parser.add_argument("--prompt", type=str, required=True, help="用于分割的文本提示词 (Text Prompt)")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备 (默认: cuda)")
    return parser.parse_args()

def save_visualization(img, results, save_path, prompt):
    """
    可视化推理结果并保存
    :param img: PIL Image 原图
    :param results: 推理结果字典 (masks, boxes, scores)
    :param save_path: 保存路径
    :param prompt: 提示词
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    
    nb_objects = len(results["scores"])
    img_w, img_h = img.size
    
    # 如果没有检测到对象，打印提示
    if nb_objects == 0:
        plt.text(img_w/2, img_h/2, "No objects detected", 
                 color='red', fontsize=20, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7))
    
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        
        # 绘制 Mask
        mask = results["masks"][i]
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
        plot_mask(mask, color=color)
        
        # 绘制 Bounding Box
        box = results["boxes"][i]
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        
        prob = results["scores"][i].item()
        
        # plot_bbox 需要 img_height, img_width 作为前两个参数
        # 且 box_format="XYXY", relative_coords=False (因为模型输出是绝对坐标)
        plot_bbox(
            img_h, 
            img_w, 
            box, 
            text=f"{prompt} {prob:.2f}", 
            box_format="XYXY", 
            color=color, 
            relative_coords=False
        )

    plt.axis('off')
    plt.title(f"Prompt: {prompt} | Found: {nb_objects} objects")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    args = parse_args()
    
    # 1. 检查输入输出目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 '{args.input_dir}' 不存在。")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. 获取图片列表
    # 支持常见图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in extensions:
        # 使用 glob 递归查找 (这里假设只在当前层级，如需递归可加 recursive=True)
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        # 同时也查找大写后缀
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
    
    # 去重并排序
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"警告: 在 '{args.input_dir}' 中未找到图片文件。")
        return
        
    print(f"找到 {len(image_files)} 张图片，准备开始推理...")
    print(f"提示词: '{args.prompt}'")
    
    # 3. 加载模型
    print("正在加载 SAM 3 模型...")
    try:
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        # 模型默认加载到 CUDA 如果可用，这里显式确认一下
        if args.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保已登录 Hugging Face (huggingface-cli login) 并且网络通畅。")
        return

    # 4. 批量推理循环
    success_count = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            save_path = os.path.join(args.output_dir, f"{basename}_vis.jpg")
            
            # 加载图片
            image = Image.open(img_path).convert("RGB")
            
            # 预处理
            inference_state = processor.set_image(image)
            
            # 文本提示推理
            output = processor.set_text_prompt(
                state=inference_state, 
                prompt=args.prompt
            )
            
            # 提取结果
            # output 包含 'masks', 'boxes', 'scores'
            # masks: [N, 1, H, W], boxes: [N, 4], scores: [N]
            
            # 可视化并保存
            save_visualization(image, output, save_path, args.prompt)
            success_count += 1
            
        except Exception as e:
            print(f"\n处理图片 '{filename}' 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print(f"\n推理完成！成功处理 {success_count}/{len(image_files)} 张图片。")
    print(f"结果已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()
