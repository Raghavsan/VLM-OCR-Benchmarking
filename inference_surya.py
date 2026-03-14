import torch
from pathlib import Path
from PIL import Image
import gc
import argparse

# Surya's specific model predictors
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor

# Import your evaluation logic
from evaluation import compute_cer, compute_wer, compute_accuracy

def load_surya_pipeline():
    print("Loading Surya OCR (Foundation, Detection, and Recognition models)...")
    
    # Surya automatically detects your Mac's MPS (Metal) GPU
    found_pred = FoundationPredictor()
    rec_pred = RecognitionPredictor(found_pred)
    det_pred = DetectionPredictor()
    
    return rec_pred, det_pred

def process_and_evaluate(root_path, img_folder_name="images", txt_folder_name="ground_truth"):
    rec_pred, det_pred = load_surya_pipeline()
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
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            
            # Load raw image (Surya handles its own optimal resolution scaling)
            image = Image.open(img_path).convert("RGB")
            
            # --- SURYA INFERENCE BLOCK ---
            # Pass the image array and the detection predictor to the recognition model
            predictions = rec_pred([image], det_predictor=det_pred)
            
            # Extract the raw text from the bounding box predictions
            extracted_lines = []
            
            # predictions[0] corresponds to the first (and only) image in our list
            for line in predictions[0].text_lines:
                extracted_lines.append(line.text)
                
            # Reconstruct the document as a single paragraph with spaces to match evaluation logic
            output_text = " ".join(extracted_lines).strip()
            
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
