import cv2

# --- Config ---
image_path = "Hand.v1i.yolov8/valid/images/3a432004-d9cd-487a-9fab-3d2ef1a52869_jpg.rf.7ef907f88eb389047b4965ff074e1805.jpg"

# --- Load image ---
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"No pude abrir la imagen: {image_path}")

# h = alto, w = ancho
h, w = img.shape[:2]

# ==========================
# CHATGPT AQUI
# ==========================

# Valores normalizados (formato YOLO: cx, cy, w, h)
cx = 0.4609375
cy = 0.3625
bw = 0.140625
bh = 0.16953125

# Pasar de normalizado a píxeles (centro + tamaño)
x_center = cx * w
y_center = cy * h
box_w = bw * w
box_h = bh * h

# Convertir centro + tamaño -> esquinas
x1 = int(x_center - box_w / 2)
y1 = int(y_center - box_h / 2)
x2 = int(x_center + box_w / 2)
y2 = int(y_center + box_h / 2)

# (Opcional) asegurar que quedan dentro de la imagen
x1 = max(0, min(x1, w - 1))
y1 = max(0, min(y1, h - 1))
x2 = max(0, min(x2, w - 1))
y2 = max(0, min(y2, h - 1))

# --- Draw rectangle (bbox) ---
# color BGR (0, 255, 0) = green, thickness=2
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# --- Show image ---
cv2.imshow("Image with BBox", img)
cv2.waitKey(0)
cv2.destroyAllWindows()