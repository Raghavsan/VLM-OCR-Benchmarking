

To tackle the diverse and noisy document extraction dataset (comprising handwriting,
signposts, receipts, and degraded texts), I experimented with different Vision-Language
Models (VLMs) and specialized OCR vision agents.
Model 1: FireRed-OCR (Qwen3VL Architecture)
● Approach: Utilized FireRed-OCR, a generative VLM. The image was processed
through a Vision Transformer and passed to a language model using a
conversational chat template.
● Libraries: torch, transformers, Pillow
● Optimizations: Implemented dynamic image downscaling (capping dimensions at
1500 pixels using Lanczos resampling) to prevent Out-Of-Memory (OOM) errors on
Apple Silicon Metal Performance Shaders (MPS). Used PyTorch's Scaled
Dot-Product Attention for memory-efficient inference.
Model 2: Microsoft Florence-2-Large
● Approach: Recognizing that conversational VLMs can hallucinate formatting on
noisy data, pivoted to Florence 2 large. This is a specialized vision-task model.
Instead of a conversational prompt, used the strict <OCR> task token to force the
model into a deterministic, spatial text-extraction mode.
● Libraries: torch, transformers, Pillow, timm, einops.
● Optimizations: Applied beam search decoding (num_beams=3) to improve typo
correction on degraded texts and utilized Microsoft's native

● post_process_generation method to cleanly parse the spatial outputs.
Model 3: GLM-OCR
● Approach: GLM-OCR is a highly capable Vision-Language Model. However,
because it is generative, it initially hallucinated complex document structures (e.g.,
turning abbreviations into vertical lists). Countered this using strict prompt
engineering, forcing it to extract text in a continuous left-to-right, top-to-bottom
paragraph without inferring formatting.
● Libraries: torch, transformers (GitHub bleeding-edge version), opencv python,
numpy, Pillow.
● Preprocessing (OpenCV): To improve accuracy on low-contrast and degraded
images, implemented an OpenCV preprocessing pipeline. Converting the images to
grayscale, applied CLAHE (Contrast Limited Adaptive Histogram Equalization) to
aggressively darken the text against faded backgrounds, and used Non-Local
Means Denoising (cv2.fastNlMeansDenoising) to smooth out digital grain and blur
without destroying the sharp edges of the letters.





Model 4: Surya OCR
● Approach: Because VLMs inherently guess missing words, tested Surya OCR—a
deterministic, specialized document extraction pipeline consisting of isolated
Foundation, Detection, and Recognition models.
● Base Run vs. Preprocessing: Initially ran Surya on the raw images, but its strict
deterministic nature caused it to struggle with heavy noise and low contrast.
Subsequently applied the exact same OpenCV preprocessing pipeline (CLAHE +
Non-Local Means Denoising) before passing the images to Surya's detection model.
Additionally, explicitly constrained Surya's recognition model by passing an English
language hint ([["en"]]) to prevent it from hallucinating foreign characters on heavily
degraded text blocks.
● Libraries: torch, surya-ocr, transformers(strictly v4.56.1, remove existing version if
any), opencv python, Pillow.
Instructions to Run the Code
Ensure you have Python 3.11 installed. Due to strict library conflicts between the models,
you must use two separate virtual environments (one exclusively for FireRed and one
for all the others)
Update the ROOT_FOLDER_PATH  variable at the bottom of each script to point to your
data directory.
For FireRed:
Install the necessary libraries:
pip install torch torchvision transformers Pillow
File structure should look like:

Create an inference.py file in the folder created by FireRed-OCR. You should have
evaluation.py file in this folder.
Run scripts directly from terminal.
## For Florence2:
Create a separate venv.

Install necessary libraries:
pip install torch torchvision transformers==4.49.0 Pillow timm einops
Make a folder inside alltius_assgn (called zai here). Store evaluation.py,
inference_florence.py here.
Run inference_florence.py from terminal.
For GLM-OCR:
Because GLM-OCR relies on unreleased custom architecture, it requires a bleeding-edge
installation of Hugging Face transformers directly from source.
pip install git+https://github.com/huggingface/transformers.git pip install torch torchvision
Pillow opencv-python numpy
Run the inference_glm.py file stored in zai.
For Surya-OCR:
Surya OCR recently underwent a major update and requires a specific version of
transformers to avoid configuration attribute crashes.
pip install surya-ocr transformers==4.56.1 torch torchvision Pillow opencv-python numpy
Run the inference_surya.py file stored in zai.
## Accuracies:

- FireRed:

## 2. Florence2:
Receipts (Average: 70.56%)
● receipt_000: 79.55%
● receipt_001: 76.92%
● receipt_002: 62.36%
● receipt_003: 54.02%
● receipt_004: 73.33%
● receipt_005: 77.18%


Degraded (Average: 29.36%)
● degraded_000: 65.88%
● degraded_001: 0.00%
● degraded_002: 55.29%
● degraded_003: 16.28%
● degraded_004: 36.61%
● degraded_005: 2.11%
Handwritten (Average: 29.08%)
● handwritten_000: 60.52%
● handwritten_001: 1.52%
● handwritten_002: 28.60%
● handwritten_003: 28.35%
● handwritten_004: 9.68%
● handwritten_005: 45.83%
Scene Text (Average: 90.00%)
● scene_000: 40.00%
● scene_001: 100.00%
● scene_002: 100.00%
● scene_003: 100.00%
● scene_004: 100.00%
● scene_005: 100.00%
Dense Text (Average: 59.41%)
● dense_000: 61.83%
● dense_001: 33.88%
● dense_002: 78.55%
● dense_003: 69.43%
● dense_004: 42.24%
● dense_005: 70.53%
Printed (Average: 46.35%)
● printed_000: 67.61%
● printed_001: 20.77%
● printed_002: 59.46%
● printed_003: 15.64%
● printed_004: 39.59%
● printed_005: 75.04%




## 3. GLM-OCR:
Receipts (Average: 51.68%)
● receipt_000: 50.50%
● receipt_001: 37.44%
● receipt_002: 53.80%
● receipt_003: 52.44%
● receipt_004: 49.76%
● receipt_005: 66.14%
Degraded (Average: 33.41%)
● degraded_000: 83.86%
● degraded_001: 0.00%
● degraded_002: 67.00%
● degraded_003: 15.50%
● degraded_004: 34.08%
● degraded_005: 0.00%
Handwritten (Average: 93.19%)
● handwritten_000: 100.00%
● handwritten_001: 92.93%
● handwritten_002: 100.00%
● handwritten_003: 84.20%
● handwritten_004: 81.98%
● handwritten_005: 100.00%
Scene Text (Average: 100.00%)
● scene_000: 100.00%
● scene_001: 100.00%
● scene_002: 100.00%
● scene_003: 100.00%
● scene_004: 100.00%
● scene_005: 100.00%
Dense Text (Average: 67.28%)

● dense_000: 80.54%
● dense_001: 42.78%
● dense_002: 78.63%
● dense_003: 72.61%
● dense_004: 59.46%
● dense_005: 69.65%
Printed (Average: 51.66%)
● printed_000: 83.86%
● printed_001: 31.92%
● printed_002: 67.00%
● printed_003: 14.76%
● printed_004: 34.08%
● printed_005: 78.31%



## 4. SURYA-OCR:
Receipts (Average: 68.09%)
● receipt_000: 65.47%
● receipt_001: 67.17%
● receipt_002: 71.55%
● receipt_003: 65.50%
● receipt_004: 75.47%
● receipt_005: 63.35%
Degraded (Average: 35.64%)
● degraded_000: 82.57%
● degraded_001: 10.20%
● degraded_002: 62.91%
● degraded_003: 13.21%
● degraded_004: 43.02%
● degraded_005: 1.90%
Handwritten (Average: 86.63%)

● handwritten_000: 100.00%
● handwritten_001: 84.34%
● handwritten_002: 88.96%
● handwritten_003: 75.54%
● handwritten_004: 70.95%
● handwritten_005: 100.00%
Scene Text (Average: 26.67%)
● scene_000: 40.00%
● scene_001: 0.00%
● scene_002: 0.00%
● scene_003: 20.00%
● scene_004: 0.00%
● scene_005: 100.00%
Dense Text (Average: 61.92%)
● dense_000: 79.51%
● dense_001: 36.53%
● dense_002: 74.53%
● dense_003: 69.36%
● dense_004: 41.17%
● dense_005: 70.41%
Printed (Average: 50.48%)
● printed_000: 82.74%
● printed_001: 29.28%
● printed_002: 62.83%
● printed_003: 13.07%
● printed_004: 43.35%
● printed_005: 71.59%



- GLM-OCR+opencv Preprocessing:
Receipts (Average: 54.94%)

● receipt_000: 53.38%
● receipt_001: 54.12%
● receipt_002: 53.80%
● receipt_003: 52.44%
● receipt_004: 49.76%
● receipt_005: 66.14%
Degraded (Average: 35.91%)
● degraded_000: 83.44%
● degraded_001: 15.45%
● degraded_002: 67.00%
● degraded_003: 15.50%
● degraded_004: 34.08%
● degraded_005: 0.00%
Handwritten (Average: 93.19%)
● handwritten_000: 100.00%
● handwritten_001: 92.93%
● handwritten_002: 100.00%
● handwritten_003: 84.20%
● handwritten_004: 81.98%
● handwritten_005: 100.00%
Scene Text (Average: 100.00%)
● scene_000: 100.00%
● scene_001: 100.00%
● scene_002: 100.00%
● scene_003: 100.00%
● scene_004: 100.00%
● scene_005: 100.00%
Dense Text (Average: 67.32%)
● dense_000: 80.54%
● dense_001: 42.78%
● dense_002: 78.63%
● dense_003: 72.61%
● dense_004: 59.70%
● dense_005: 69.65%
Printed (Average: 51.66%)
● printed_000: 83.86%
● printed_001: 31.92%
● printed_002: 67.00%
● printed_003: 14.76%

● printed_004: 34.08%
● printed_005: 78.31%



- SURYA-OCR+opencv Preprocessing:
Receipts (Average: 68.01%)
● receipt_000: 65.86%
● receipt_001: 65.60%
● receipt_002: 65.72%
● receipt_003: 66.64%
● receipt_004: 76.50%
● receipt_005: 64.71%
Degraded (Average: 36.90%)
● degraded_000: 76.42%
● degraded_001: 11.30%
● degraded_002: 62.59%
● degraded_003: 13.24%
● degraded_004: 42.47%
● degraded_005: 15.35%
Handwritten (Average: 86.63%)
● handwritten_000: 100.00%
● handwritten_001: 84.34%
● handwritten_002: 88.96%
● handwritten_003: 75.54%
● handwritten_004: 70.95%
● handwritten_005: 100.00%
Scene Text (Average: 21.67%)
● scene_000: 10.00%
● scene_001: 0.00%
● scene_002: 0.00%
● scene_003: 20.00%

● scene_004: 0.00%
● scene_005: 100.00%
Dense Text (Average: 60.46%)
● dense_000: 79.60%
● dense_001: 37.59%
● dense_002: 74.22%
● dense_003: 69.21%
● dense_004: 42.27%
● dense_005: 59.87%
Printed (Average: 50.55%)
● printed_000: 82.74%
● printed_001: 29.28%
● printed_002: 63.31%
● printed_003: 13.07%
● printed_004: 43.24%
● printed_005: 71.63%






