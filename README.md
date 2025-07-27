# Face Detection and Cropping Pipeline using MTCNN

This repository provides a complete pipeline to detect and crop face regions from images using the MTCNN (Multi-task Cascaded Convolutional Networks) face detector from the facenet_pytorch library.

It is designed specifically for biometric and computer vision research applications where precise face localization and dataset organization are critical. Key features include:

✅ **Robust detection** of faces captured from various devices (mobile phones, DSLR cameras, etc.) and at multiple distances (e.g., 1m, 10m, 50m).

✅ **EXIF orientation correction** for mobile-captured images to ensure consistent face alignment.

✅ **GPU acceleration support**, enabling fast processing of large datasets.

✅ **Cropped face output** saved as high-quality image patches.

✅ **Excel summary generation** that logs detailed metadata.



Whether you're working with surveillance footage, academic face datasets, or real-time image capture scenarios, this pipeline provides an efficient and scalable solution for preparing face crops with traceable metadata and repeatable logic.


![Face Detection Output](assets/example_pipeline.png)

## 🚀 Goal and Output
This project implements an automated **face detection and cropping pipeline** using **MTCNN** from the `facenet_pytorch` library. It works with high-resolution face datasets (from both **Mobile** and **Camera**) and outputs:

- Cropped face images
- An Excel file with detailed metadata:
  - Subject number, image name, camera type, distance
  - Detection status, face box coordinates
  - Image sizes, detection time

---

## 📦 Installation
Install the required dependencies:
```bash
pip install facenet_pytorch
pip install torch torchvision
pip install pandas opencv-python pillow
```

⚠️ **Important**: You **must install `facenet_pytorch`**, which includes MTCNN.

---

## ⚙️ Check GPU Availability
To confirm that PyTorch is using GPU:
```python
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
```

---

## 📸 EXIF Orientation Correction
Many mobile images store orientation data in EXIF metadata. This function corrects it before processing:
```python
def load_image_with_correct_orientation(path):
    img = Image.open(path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except Exception as e:
        print(f"⚠️ EXIF orientation not applied: {e}")
    return img.convert("RGB")
```

This step is **crucial** for face detection to succeed on mobile images.

---

## 📂 Directory Setup
Update paths in `src/face_detection.py` before running:
```python
root_folder = r'path\to\your\image_folder'
output_folder = r'path\to\your\output_folder'
```

Then run:
```bash
python src/face_detection.py
```

---

## 📊 Excel Output Explanation
The output Excel file `face_detection_MTCNN.xlsx` contains:

| Column                | Description                                |
|----------------------|--------------------------------------------|
| subject number       | Extracted from image filename              |
| image name           | Original image filename                    |
| indoor/outdoor       | Parsed from `_in_` or `_out_` in filename  |
| device               | 'Camera' or 'Mobile' based on filename     |
| distance             | e.g., `1m`, `50m`, parsed from filename    |
| category             | `Live`, `Spoof`, or `Props`                |
| detection            | `Detected` or `No Detection`               |
| face box coordinate  | Bounding box (x1, y1, x2, y2)               |
| detection time (s)   | Time in seconds                            |
| input image size     | WxH in pixels                              |
| cropped image size   | WxH of cropped face image                  |

Example image filenames:
- `101_in_c_1m_live.JPG`
- `101_in_c_50m_spoof1_61.JPG`
- `101_in_c_10m_live_propcat2.JPG`

---

## 📁 Repository Structure
```
face-detection-mtcnn-cropping-pipeline/
├── src/
│   └── face_detection.py
├── output/
│   ├── cropped images
│   └── face_detection_MTCNN.xlsx
├── requirements.txt
├── LICENSE
├── .gitignore
├── CITATION.cff
└── README.md
```

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Citation
If you use this codebase in your research, please cite:
```yaml
cff-version: 1.2.0
authors:
  - family-names: "Chowdhury"
    given-names: "Mohammad Zahir Uddin"
    affiliation: "Clarkson University"
    orcid: "https://orcid.org/0009-0009-8194-6669"
title: "Face Detection and Cropping Pipeline using MTCNN"
date-released: 2025-07-27
license: MIT
repository-code: https://github.com/mdzahirdu/face-detection-mtcnn-cropping-pipeline
```
