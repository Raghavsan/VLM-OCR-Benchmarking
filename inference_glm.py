import torch
from pathlib import Path
from PIL import Image
import gc
from transformers import AutoProcessor, AutoModelForImageTextToText
import argparse

# Import your evaluation logic
from evaluation import compute_cer, compute_wer, compute_accuracy

def load_glm_pipeline():
    print("Loading GLM-OCR model directly to Mac Metal (MPS)...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. FIX: Use AutoProcessor to handle both images and text
    processor = AutoProcessor.from_pretrained("zai-org/GLM-OCR", trust_remote_code=True)
    
    # 2. FIX: Use the correct Vision-Language AutoClass
    model = AutoModelForImageTextToText.from_pretrained(
        "zai-org/GLM-OCR",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device
    )
    
    return model, processor, device

def process_and_evaluate(root_path, img_folder_name="images", txt_folder_name="ground_truth"):
    model, processor, device = load_glm_pipeline()
    root_dir = Path(root_path)
    
    total_accuracy = 0.0
    total_cer = 0.0
    total_wer = 0.0
    processed_count = 0
    
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
            
            # Safely Downscale Massive Images
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
                print(f"   [!] Resized {width}x{height} down to {new_w}x{new_h}")
            
           # 3. FIX: Generate the strict prompt with image placeholders
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": """Extract every single word from this image exactly as it appears.
            
Strict Rules:
1. Reading Order: Read strictly from top-to-bottom, left-to-right. Do not skip any headers, titles, or dates at the very top of the document.
2. No Formatting: Do not format the output as a vertical list, bullet points, or key-value pairs. Output the text as a single, continuous paragraph with single spaces between words.
3. Exact Transcription: Do not group, summarize, or rearrange items. Do not insert dashes or hyphens unless they are physically printed between those specific letters.
4. Character Accuracy: Pay extreme attention to the difference between the letter 'O' and the number '0'.

Begin raw extraction:"""}
                    ]
                }
            ]
            
            # This injects the exact hidden tokens the GLM model architecture expects
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # 4. Process inputs by passing the perfectly formatted text and the image object
            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=4096, temperature=0.0)
            
            # Decode the specific output tokens (ignoring the prompt tokens)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            output_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            del inputs
            del outputs
            
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
