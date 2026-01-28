import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

# Add current directory to path so we can import sam3 modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def parse_args():
    parser = argparse.ArgumentParser(description="SAM 3 Adapter for OpenIns3D")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output .pt file")
    parser.add_argument("--vocab", type=str, required=True, help="Semicolon separated vocabulary list")
    parser.add_argument("--save_vis_dir", type=str, default=None, help="Directory to save visualization images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()

def save_visualization(image, panoptic_seg, instance_classes, vocab_list, save_path):
    """
    Visualize panoptic segmentation on the image.
    """
    import matplotlib.pyplot as plt
    from sam3.visualization_utils import COLORS
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # panoptic_seg: (H, W), values 1..N
    unique_ids = np.unique(panoptic_seg)
    unique_ids = unique_ids[unique_ids > 0]
    
    # Overlay masks
    for inst_id in unique_ids:
        mask = panoptic_seg == inst_id
        # Use inst_id to pick color
        color = COLORS[(inst_id - 1) % len(COLORS)]
        
        # Simple mask overlay
        img_h, img_w = mask.shape
        color_mask = np.zeros((img_h, img_w, 4))
        color_mask[mask] = list(color) + [0.35] # alpha
        plt.imshow(color_mask)
        
        # Add label text at center
        y, x = np.where(mask)
        if len(y) > 0:
            cy, cx = int(np.mean(y)), int(np.mean(x))
            class_idx = instance_classes[inst_id - 1] # 0-based list, 1-based ID
            label_text = vocab_list[class_idx]
            plt.text(cx, cy, label_text, color='white', fontsize=8, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, pad=0))

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def merge_masks(masks_info, height, width):
    """
    Merge overlapping masks into a panoptic segmentation map using a greedy strategy based on score.
    
    Args:
        masks_info: List of tuples (mask, score, class_id). mask is (H, W) boolean/uint8.
        height: Image height
        width: Image width
        
    Returns:
        panoptic_seg: (H, W) int64, 0 is background, 1..N is instance ID
        instance_to_class: List where index i (0-based) corresponds to instance ID i+1. Value is class_id.
    """
    # Sort by score descending
    masks_info.sort(key=lambda x: x[1], reverse=True)
    
    panoptic_seg = np.zeros((height, width), dtype=np.int64)
    instance_to_class = []
    
    current_instance_id = 1
    
    for mask, score, class_id in masks_info:
        # mask should be boolean or 0/1
        mask_bool = mask > 0
        
        # Find pixels that are part of this mask and currently empty in panoptic_seg
        # We prioritize high confidence masks. 
        # Alternatively, we could overwrite low confidence masks, but "first come first serve" with sorted scores is standard NMS-like behavior for segmentation.
        valid_pixels = np.logical_and(mask_bool, panoptic_seg == 0)
        
        if np.count_nonzero(valid_pixels) > 0:
            panoptic_seg[valid_pixels] = current_instance_id
            instance_to_class.append(class_id)
            current_instance_id += 1
            
    return panoptic_seg, instance_to_class

def main():
    args = parse_args()
    
    # 1. Parse vocab
    # Format: "chair;table;sofa"
    # Note: OpenIns3D passes vocab as a list of strings usually. Here we receive a joined string.
    # We need to map class name back to index in the vocab list provided by OpenIns3D.
    vocab_list = [v.strip() for v in args.vocab.split(";")]
    print(f"Loaded {len(vocab_list)} classes.")

    # 2. Get images
    # OpenIns3D naming convention: image_rendered_angle_{i}.png
    # We should sort them to ensure consistent order with OpenIns3D's glob
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "image_rendered_angle_*.png")))
    
    if not image_files:
        # Fallback to standard png if specific naming not found (for robustness)
        image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
        
    if not image_files:
        print(f"No images found in {args.image_dir}")
        return

    # Check image size from first image
    first_img = Image.open(image_files[0])
    width, height = first_img.size
    num_images = len(image_files)
    
    print(f"Found {num_images} images. Resolution: {width}x{height}")

    # 3. Load Model
    print("Loading SAM3 model...")
    try:
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        if args.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 4. Inference Loop
    panoptic_seg_list = torch.zeros((num_images, height, width), dtype=torch.int64)
    # Max 80 instances per image as per OpenIns3D convention
    labels_tensor = torch.ones((num_images, 80), dtype=torch.int64) * -1
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing Images")):
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Set image (extract features once)
            inference_state = processor.set_image(image)
            
            masks_info = [] # List of (mask, score, class_id)
            
            # Iterate over each class in vocab
            for class_id, class_name in enumerate(vocab_list):
                if not class_name: continue
                
                # Run inference for this prompt
                output = processor.set_text_prompt(
                    state=inference_state, 
                    prompt=class_name
                )
                
                # output['masks'] is (N, 1, H, W) or (N, H, W) depending on version, let's check shapes
                # output['scores'] is (N,)
                
                current_masks = output["masks"]
                current_scores = output["scores"]
                
                if isinstance(current_masks, torch.Tensor):
                    current_masks = current_masks.cpu().numpy()
                if isinstance(current_scores, torch.Tensor):
                    current_scores = current_scores.cpu().numpy()
                    
                # Iterate over detected instances for this class
                for i in range(len(current_scores)):
                    score = current_scores[i]
                    mask = current_masks[i]
                    
                    # SAM3 outputs might be (1, H, W)
                    if mask.ndim == 3:
                        mask = mask[0]
                        
                    masks_info.append((mask, score, class_id))
            
            # Merge masks for this image
            panoptic_seg, instance_classes = merge_masks(masks_info, height, width)
            
            # Store results
            panoptic_seg_list[idx] = torch.from_numpy(panoptic_seg)
            
            # Fill labels tensor
            num_instances = len(instance_classes)
            if num_instances > 0:
                # Truncate if more than 80 instances
                limit = min(num_instances, 80)
                labels_tensor[idx, :limit] = torch.tensor(instance_classes[:limit], dtype=torch.int64)

            # Save visualization if requested
            if args.save_vis_dir:
                os.makedirs(args.save_vis_dir, exist_ok=True)
                vis_path = os.path.join(args.save_vis_dir, os.path.basename(img_path))
                save_visualization(image, panoptic_seg, instance_classes, vocab_list, vis_path)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()

    # 5. Save Results
    print(f"Saving results to {args.output_path}...")
    # Create directory if not exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    results = {
        "panoptic_seg_list": panoptic_seg_list,
        "labels": labels_tensor
    }
    torch.save(results, args.output_path)
    print("Done.")

if __name__ == "__main__":
    main()
