import cv2
from ultralytics import YOLO

# --- Carga tu modelo ---
model_path = "/Users/tonycituk/workspace/hand-sign-number-recognition/runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

# --- Abre la cámara ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No pude abrir la cámara.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Predicción ---
    results = model(frame, device="mps")
    r = results[0]

    # --- Dibujar cajas ---
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())

            # Convertir a int
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # --- DIBUJAR BOUNDING BOX ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)

            # --- TEXTO DEL LABEL ---
            class_name = model.names[cls_id]
            text = f"{class_name}: {conf:.2f}"

            # Tamaño del texto
            (tw, th), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, 2
            )

            # Coordenadas del rectángulo de fondo
            tag_x1 = x1
            tag_y1 = y1 - th - 10
            if tag_y1 < 0:
                tag_y1 = y1 + th + 10  # Si está muy arriba, mover abajo

            tag_x2 = x1 + tw + 10
            tag_y2 = y1

            # --- FONDO DEL LABEL ---
            cv2.rectangle(frame,
                          (tag_x1, tag_y1),
                          (tag_x2, tag_y2),
                          (50, 255, 50),
                          thickness=-1)  # sólido

            # --- TEXTO ENCIMA DEL FONDO ---
            cv2.putText(frame,
                        text,
                        (tag_x1 + 5, tag_y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0), 2)

    # Mostrar resultado
    cv2.imshow("YOLO Hand Sign Live", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()