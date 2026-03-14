import torch
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from conv_for_infer import generate_conv
import gc
from PIL import Image
import os
import argparse

# Import the evaluation logic from your separate file
from evaluation import compute_cer, compute_wer, compute_accuracy

def load_pipeline():
    print("Loading FireRed-OCR model with PyTorch SDPA...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "FireRedTeam/FireRed-OCR",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("FireRedTeam/FireRed-OCR")
    return model, processor

def process_and_evaluate(root_path, img_folder_name="images", txt_folder_name="ground_truth"):
    model, processor = load_pipeline()
    
    root_dir = Path(root_path)
    
    total_accuracy = 0.0
    total_cer = 0.0
    total_wer = 0.0
    processed_count = 0
    
    # 1. Traverse the root directory to find subfolders
    for sub_dir in root_dir.iterdir():
        if not sub_dir.is_dir():
            continue
            
        images_dir = sub_dir / img_folder_name
        texts_dir = sub_dir / txt_folder_name
        
        # Check if both required subdirectories exist
        if not images_dir.exists() or not texts_dir.exists():
            print(f"Skipping {sub_dir.name}: Missing images or texts folder.")
            continue
            
        # 2. Iterate through all images in the images folder
        for img_path in images_dir.iterdir():
            # Crucial for Mac users to ignore hidden .DS_Store files
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            print(f"\n==========================================")
            print(f"Processing: {sub_dir.name}/{img_path.name}")
            
            # 3. Check for ground truth FIRST
            txt_path = texts_dir / f"{img_path.stem}.txt"
            if not txt_path.exists():
                print(f"Warning: No ground truth text found for {img_path.name}. Skipping.")
                continue
                
            with open(txt_path, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()

            # 4. Clear memory before starting inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
            # 5. Safely Downscale Massive Images
            max_dimension = 1500
            target_image_path = str(img_path)
            
            with Image.open(img_path) as img:
                width, height = img.size
                if width > max_dimension or height > max_dimension:
                    if width > height:
                        new_w = max_dimension
                        new_h = int(height * (max_dimension / width))
                    else:
                        new_h = max_dimension
                        new_w = int(width * (max_dimension / height))
                    
                    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    target_image_path = f"temp_resized_{img_path.name}"
                    resized_img.save(target_image_path)
                    print(f"   [!] Resized {width}x{height} down to {new_w}x{new_h}")
            
            # 6. Prepare Input
            messages = generate_conv(target_image_path)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            
            # 7. INFERENCE (Done ONLY ONCE, and safely with no_grad)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=8192)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
            # 8. Clean up variables and temporary images immediately
            del inputs
            del generated_ids
            del generated_ids_trimmed
            
            if "temp_resized_" in target_image_path and os.path.exists(target_image_path):
                os.remove(target_image_path)
            
            # 9. Evaluation & Printing
            item_cer = compute_cer(output_text, reference_text)
            item_wer = compute_wer(output_text, reference_text)
            item_acc = compute_accuracy(output_text, reference_text)
            
            total_cer += item_cer
            total_wer += item_wer
            total_accuracy += item_acc
            processed_count += 1
            
            # --- NEW ADDITION: Print the texts before the metrics ---
            print("\n--- TEXT COMPARISON ---")
            print(f"GROUND TRUTH:\n{reference_text}")
            print(f"\nPREDICTION:\n{output_text}")
            print("-----------------------")
            print(f"--> ACC: {item_acc:.2f}% | CER: {item_cer:.4f} | WER: {item_wer:.4f}")

    # 10. Final Report
    if processed_count > 0:
        print("\n==========================================")
        print(f"PIPELINE COMPLETE - {processed_count} FILES PROCESSED")
        print(f"Average Accuracy: {total_accuracy / processed_count:.2f}%")
        print(f"Average CER:      {total_cer / processed_count:.4f}")
        print(f"Average WER:      {total_wer / processed_count:.4f}")
        print("==========================================")
    else:
        print("No matching image/text pairs were found in the specified directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR Evaluation Pipeline")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Relative path to the root data directory"
    )
    args = parser.parse_args()
    
    process_and_evaluate(args.data_dir)


