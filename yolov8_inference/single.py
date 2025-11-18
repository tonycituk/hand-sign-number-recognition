import os
import cv2

BASE_DIR = "Hand.v1i.yolov8/train"
images_dir = os.path.join(BASE_DIR, "images")
labels_dir = os.path.join(BASE_DIR, "labels")

# Elige una imagen de ejemplo (puedes cambiar el nombre)
base_name = "3c71bd9e-3d91-4842-8d33-9ca49d7ddd6a_jpg.rf.bcc7c5a82a92951c6cab074351a74b34"

img_path = os.path.join(images_dir, base_name + ".jpg")
label_path = os.path.join(labels_dir, base_name + ".txt")

# --- Cargar imagen ---
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"No pude abrir la imagen: {img_path}")

img_h, img_w = img.shape[:2]

# --- Función para convertir YOLO (cx,cy,bw,bh) a (x1,y1,x2,y2) en píxeles ---
def yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h):
    x_center = cx * img_w
    y_center = cy * img_h
    box_w = bw * img_w
    box_h = bh * img_h

    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    # Opcional: clamp para que no se salgan de la imagen
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    return x1, y1, x2, y2

# --- Leer labels y dibujar cajas ---
if not os.path.exists(label_path):
    raise FileNotFoundError(f"No encontré el label: {label_path}")

with open(label_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])

        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h)

        # Dibujar el bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Dibujar id de clase
        cv2.putText(img, str(cls_id), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# --- Mostrar imagen ---
cv2.imshow("Image with YOLO BBoxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()