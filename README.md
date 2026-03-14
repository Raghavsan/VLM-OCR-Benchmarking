# SOTA Document Extraction & VLM Benchmarking Pipeline

##  Overview
An automated benchmarking suite designed to evaluate state-of-the-art Vision-Language Models (VLMs) and specialized OCR vision agents. This project tests model performance against highly degraded, noisy, and complex document datasets, including cursive handwriting, faded receipts, signposts, and dense text layouts. 

To bridge the gap between raw, degraded pixels and the models' Vision Transformers (ViTs), a custom **OpenCV preprocessing engine** was engineered, significantly boosting recognition accuracy on low-contrast and salt-and-pepper noise images.

---

##  Evaluated Architectures

### 1. GLM-OCR (Generative VLM)
* **Approach:** GLM-OCR is a highly capable Vision-Language Model. Because it is generative, it initially hallucinated complex document structures (e.g., turning abbreviations into vertical lists). This was countered using strict prompt engineering to force continuous left-to-right, top-to-bottom extraction.
* **Libraries:** `torch`, `transformers` (GitHub developer build), `Pillow`.

### 2. Microsoft Florence-2-Large (Vision Agent)
* **Approach:** A specialized spatial vision-task model. Instead of a conversational prompt, the strict `<OCR>` task token was utilized to force the model into a deterministic text-extraction mode, bypassing chatbot-like hallucinations.
* **Libraries:** `torch`, `transformers==4.49.0`, `Pillow`, `timm`, `einops`.
* **Hardware Optimizations:** Applied beam search decoding (`num_beams=3`) to improve typo correction on degraded texts. 

### 3. FireRed-OCR (Qwen3VL Architecture)
* **Approach:** Processed through a Vision Transformer and passed to a language model using a conversational chat template.
* **Hardware Optimizations:** Implemented dynamic Lanczos image downscaling (1500px cap) and PyTorch's Scaled Dot-Product Attention (`sdpa`) to prevent Out-Of-Memory (OOM) errors on Apple Silicon Metal (MPS).

### 4. Surya OCR (Deterministic Pipeline)
* **Approach:** A deterministic, specialized document extraction pipeline utilizing isolated Foundation, Detection, and Recognition models.
* **Optimizations:** Explicitly constrained the recognition model's vocabulary dictionary by passing an English language hint (`[["en"]]`) to prevent hallucinated foreign characters on smudged ink.

---

## OpenCV Image Preprocessing
To maximize accuracy on heavily degraded datasets, an image enhancement layer was applied *before* the images reached the models.

1. **Grayscale Conversion:** Removed distracting color data.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Aggressively darkened faded ink while whitening grayish backgrounds.
3. **Non-Local Means Denoising:** (`cv2.fastNlMeansDenoising`) Smoothed out digital grain and artifact blur without destroying the sharp edges of the typography.

---

## Evaluation Results
Performance was quantified using strictly calculated Character Error Rate (CER), Word Error Rate (WER), and total Accuracy across 36 distinct file groups. 

*Integrating the OpenCV preprocessing engine yielded the highest overall accuracy, pushing GLM-OCR to the top of the benchmark.*

| Model Pipeline | Overall Accuracy | Avg CER | Avg WER |
| :--- | :--- | :--- | :--- |
| **GLM-OCR + OpenCV** | **67.17%** | **0.2779** | **0.3787** |
| GLM-OCR (Base) | 66.20% | 0.2903 | 0.3857 |
| Surya OCR + OpenCV | 56.73% | 0.3582 | 0.5058 |
| Surya OCR (Base) | 54.90% | 0.3709 | 0.5311 |
| Florence-2 | 54.13% | 0.3621 | 0.5401 |

---

## Project Structure

```text
VLM-OCR-Benchmarking/
│
├── .gitignore                
├── README.md                 
│
├── evaluation.py             # Core math/metrics logic (CER, WER, Accuracy)
├── inference.py              # FireRed pipeline
├── inference_florence.py     # Florence-2 pipeline
├── inference_glm.py          # GLM-OCR + OpenCV pipeline
├── inference_surya.py        # Surya OCR pipeline
│
└── sample_data/              # Example directory structure for target images
    └── category_1/
        ├── images/
        │   └── category_1_000.png
        └── ground_truth/
            └── category_1_000.txt

Installation & Usage

Hardware Requirements: The scripts auto-detect Apple Silicon (MPS), CUDA, or fallback to CPU.
Crucial Note: Because these models utilize conflicting dependency architectures, you must use isolated virtual environments.
1. Set up your Data Directory

Ensure your dataset follows the structure outlined in sample_data/. You can pass your custom data directory path at runtime using the --data_dir argument.
2. Running GLM-OCR

GLM requires a bleeding-edge installation of Hugging Face transformers directly from source.
Bash

python3.11 -m venv venv_glm
source venv_glm/bin/activate

pip install git+[https://github.com/huggingface/transformers.git](https://github.com/huggingface/transformers.git)
pip install torch torchvision Pillow opencv-python numpy

python inference_glm.py --data_dir ./your_data_folder

3. Running Florence-2 & Surya OCR

These models require stable, specific versions of transformers.
Bash

python3.11 -m venv venv_stable
source venv_stable/bin/activate

# Install strictly compatible versions
pip install torch torchvision Pillow opencv-python numpy timm einops
pip install transformers==4.56.1 surya-ocr

# Run Florence-2
python inference_florence.py --data_dir ./your_data_folder

# Run Surya
python inference_surya.py --data_dir ./your_data_folder
