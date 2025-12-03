import os
import cv2
from glob import glob

# --- Configuración de rutas ---
BASE_DIR = "/Users/tonycituk/workspace/hand-sign-number-recognition/yolov8_inference/HandConverted/test"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")

def extract_bbox_image(img, x1, y1, x2, y2, clamp=True):
    """
    Extract the subimage contained in the bbox (x1, y1, x2, y2).

    img  : numpy array (OpenCV image, BGR)
    x1,y1: top-left corner (inclusive)
    x2,y2: bottom-right corner (exclusive is fine; we clamp below)

    Returns: cropped image as numpy array.
    """
    h, w = img.shape[:2]

    if clamp:
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w,     x2))  # allow x2 == w
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h,     y2))  # allow y2 == h

    # If bbox is invalid, return None
    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2].copy()
    return crop

# --- Función para convertir YOLO (cx,cy,bw,bh) -> (x1,y1,x2,y2) en píxeles ---
def yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h):
    x_center = cx * img_w
    y_center = cy * img_h
    box_w = bw * img_w
    box_h = bh * img_h

    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    # Clamp para no salirnos de la imagen
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    return x1, y1, x2, y2

# --- Obtener todas las imágenes ---
extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
image_paths = []

for ext in extensions:
    image_paths.extend(glob(os.path.join(IMAGES_DIR, ext)))

image_paths = sorted(image_paths)
print(image_paths)

print(f"Found {len(image_paths)} images")
for i, p in enumerate(image_paths[:5]):
    print(i, "->", p)

if not image_paths:
    raise FileNotFoundError(f"No images found in: {IMAGES_DIR}")

# --- Elige qué índice de imagen quieres mostrar ---
idx = 62   # <--- CAMBIA ESTE ÍNDICE PARA OTRA IMAGEN

if idx < 0 or idx >= len(image_paths):
    raise IndexError(f"idx {idx} fuera de rango, solo hay {len(image_paths)} imágenes")

img_path = image_paths[idx]
print(f"Displaying image {idx}: {img_path}")

# --- Cargar imagen ---
img_og = cv2.imread(img_path)
img = img_og.copy()
if img is None:
    raise RuntimeError(f"Could not open image: {img_path}")

img_h, img_w = img.shape[:2]

# --- Construir ruta del label correspondiente ---
base_name = os.path.splitext(os.path.basename(img_path))[0]  # sin extensión
label_path = os.path.join(LABELS_DIR, base_name + ".txt")

print(f"Using label file: {label_path}")

# --- Leer labels y dibujar cajas (si existe el label) ---
if os.path.exists(label_path):
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue  # línea malformada

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h)

            # Dibujar bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Dibujar id de clase
            cv2.putText(
                img, str(cls_id), (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
else:
    print(f"[WARN] No label file found for this image: {label_path}")

# --- Mostrar imagen ---
cv2.imshow(f"Image {idx} with YOLO BBoxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

crop = extract_bbox_image(img_og, x1, y1, x2, y2)


if crop is not None:
    save_path = "./yolov8_inference/crop.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, crop)
    print(f"Crop guardado en: {save_path}")
else:
    print("BBox inválido, no se pudo recortar.")
