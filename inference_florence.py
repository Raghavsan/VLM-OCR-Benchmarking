import torch
from pathlib import Path
from PIL import Image
import gc
from transformers import AutoProcessor, AutoModelForCausalLM
import argparse

# Import your evaluation logic
from evaluation import compute_cer, compute_wer, compute_accuracy

def load_florence_pipeline():
    print("Loading Microsoft Florence-2-Large to Mac Metal (MPS)...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "microsoft/Florence-2-large"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    return model, processor, device

def process_and_evaluate(root_path, img_folder_name="images", txt_folder_name="ground_truth"):
    model, processor, device = load_florence_pipeline()
    root_dir = Path(root_path)
    
    total_accuracy = 0.0
    total_cer = 0.0
    total_wer = 0.0
    processed_count = 0
    
    # The specific task token that forces Florence into pure text-extraction mode
    task_prompt = "<OCR>"
    
    for sub_dir in root_dir.iterdir():
        if not sub_dir.is_dir():
            continue
            
        images_dir = sub_dir / img_folder_name
        texts_dir = sub_dir / txt_folder_name
        
        if not images_dir.exists() or not texts_dir.exists():
            continue
            
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            print(f"\n==========================================")
            print(f"Processing: {sub_dir.name}/{img_path.name}")
            
            txt_path = texts_dir / f"{img_path.stem}.txt"
            if not txt_path.exists():
                print(f"Warning: No ground truth text found. Skipping.")
                continue
                
            with open(txt_path, 'r', encoding='utf-8') as f:
                reference_text = f.read().strip()

            # Clear Mac memory
            if device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            
            # Load image (Florence handles dynamic sizing well, but we cap it to save RAM)
            max_dimension = 1500
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_w = max_dimension
                    new_h = int(height * (max_dimension / width))
                else:
                    new_h = max_dimension
                    new_w = int(width * (max_dimension / height))
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                print(f"   [!] Resized down to {new_w}x{new_h}")
            
            # --- FLORENCE-2 INFERENCE BLOCK ---
            inputs = processor(
                text=task_prompt,
                images=image,
                return_tensors="pt"
            ).to(device, torch.float16)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=4096,
                    num_beams=3, # Slight beam search helps Florence correct typos in degraded text
                    do_sample=False
                )
            
            # Decode the raw output
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # Cleanly parse the dictionary output using Microsoft's official method
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height)
            )
            
            output_text = parsed_answer[task_prompt].strip()
            
            del inputs
            del generated_ids
            
            # Evaluation & Printing
            item_cer = compute_cer(output_text, reference_text)
            item_wer = compute_wer(output_text, reference_text)
            item_acc = compute_accuracy(output_text, reference_text)
            
            total_cer += item_cer
            total_wer += item_wer
            total_accuracy += item_acc
            processed_count += 1
            
            print("\n--- TEXT COMPARISON ---")
            print(f"GROUND TRUTH:\n{reference_text}")
            print(f"\nPREDICTION:\n{output_text}")
            print("-----------------------")
            print(f"--> ACC: {item_acc:.2f}% | CER: {item_cer:.4f} | WER: {item_wer:.4f}")

    if processed_count > 0:
        print("\n==========================================")
        print(f"PIPELINE COMPLETE - {processed_count} FILES PROCESSED")
        print(f"Average Accuracy: {total_accuracy / processed_count:.2f}%")
        print(f"Average CER:      {total_cer / processed_count:.4f}")
        print(f"Average WER:      {total_wer / processed_count:.4f}")
        print("==========================================")

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
