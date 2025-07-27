import os
import cv2
import pandas as pd
import time
from PIL import Image, ExifTags
import torch
from facenet_pytorch import MTCNN






# === Root directory ===

root_folder = r'path\to\your\image_folder'
output_folder = r'path\to\your\output_folder'
os.makedirs(output_folder, exist_ok=True)

# === Subject filter ===
specific_subjects = None  # or ["Subject_269"]


# === Initialize MTCNN (GPU if available) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

# === Correct EXIF orientation for mobile images ===
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

# === Store results ===
results = []

# === Get subject folders ===
subject_list = os.listdir(root_folder)

for subject_id in subject_list:
    subject_path = os.path.join(root_folder, subject_id)
    if not os.path.isdir(subject_path):
        continue

    if specific_subjects is not None and subject_id not in specific_subjects:
        continue

    print(f"\n🔹 Processing subject: {subject_id}")

    for device_type in ["Camera", "Mobile"]:
        device_path = os.path.join(subject_path, device_type)
        if not os.path.isdir(device_path):
            continue

        for filename in os.listdir(device_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_path = os.path.join(device_path, filename)
            print(f"  ➡️ Processing: {image_path}")

            # === Parse filename ===
            try:
                parts = filename.split("_")
                subject_number = parts[0]
                inout = parts[1]
                indoor_outdoor = "Indoor" if inout.lower() == "in" else "Outdoor"
                device_code = parts[2]
                device_name = "Camera" if device_code == "c" else "Mobile"
                distance = parts[3]
                category = "Unknown"
                for part in parts[4:]:
                    if "propcat" in part.lower():
                        category = "Props"
                        break
                    elif "spoof" in part.lower():
                        category = "Spoof"
                        break
                    elif "live" in part.lower():
                        category = "Live"
            except Exception as e:
                print(f"⚠️ Error parsing filename: {filename} ({e})")
                continue

            # === Load image with EXIF correction ===
            try:
                img = load_image_with_correct_orientation(image_path)
                img_cv = cv2.imread(image_path)
                input_size = f"{img_cv.shape[1]}x{img_cv.shape[0]}" if img_cv is not None else "N/A"
            except Exception as e:
                print(f"⚠️ Error loading image: {filename} ({e})")
                continue

            # === Run MTCNN ===
            start_time = time.time()
            boxes, _ = mtcnn.detect(img)
            end_time = time.time()
            elapsed_time = round(end_time - start_time, 2)

            if boxes is not None:
                x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                cropped_face = img.crop((x1, y1, x2, y2))
                cropped_filename = f"crop_{subject_id}_{device_type}_{filename}"
                cropped_path = os.path.join(output_folder, cropped_filename)
                cropped_face.save(cropped_path)

                cropped_size = f"{cropped_face.width}x{cropped_face.height}"
                detection_status = "Detected"
                box_coords = f"{x1},{y1},{x2},{y2}"
            else:
                detection_status = "No Detection"
                cropped_size = "N/A"
                box_coords = "No face detected"

            # === Record result ===
            results.append({
                "subject number": subject_number,
                "image name": filename,
                "indoor/outdoor": indoor_outdoor,
                "device": device_name,
                "distance": distance,
                "category": category,
                "detection": detection_status,
                "face box coordinate": box_coords,
                "detection time (s)": elapsed_time,
                "input image size": input_size,
                "cropped image size": cropped_size
            })

# === Save Excel ===
df = pd.DataFrame(results)
excel_path = os.path.join(output_folder, "face_detection_MTCNN.xlsx")
df.to_excel(excel_path, index=False)
print(f"\n✅ Excel saved to: {excel_path}")
