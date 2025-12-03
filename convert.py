import os
import shutil
from pathlib import Path

BASE_ORIG = Path("/Users/tonycituk/workspace/hand-sign-number-recognition/yolov8_inference/Hand.v1i.yolov8")
BASE_NEW  = Path("/Users/tonycituk/workspace/hand-sign-number-recognition/yolov8_inference/HandConverted")

splits = ["train", "valid", "test"]

# Crear estructura destino
for s in splits:
    (BASE_NEW / s / "images").mkdir(parents=True, exist_ok=True)
    (BASE_NEW / s / "labels").mkdir(parents=True, exist_ok=True)

# Lista de clases originales
orig_classes = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'no_gester', 'ok',
    'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop',
    'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted'
]

orig_index_to_name = {i: n for i, n in enumerate(orig_classes)}

# NUEVO MAPEO NUMÉRICO
mapping = {
    "one": 0,
    "two_up": 1,
    "two_up_inverted": 1,
    "peace": 1,
    "peace_inverted": 1,
    "three": 2,
    "three2": 2,
    "four": 3,
    "palm": 4,
    "stop": 4,
    "stop_inverted": 4,
}

def process_split(split):
    print(f"\n▶ Procesando {split}...")
    
    orig_lbl_dir = BASE_ORIG / split / "labels"
    orig_img_dir = BASE_ORIG / split / "images"
    
    new_lbl_dir  = BASE_NEW / split / "labels"
    new_img_dir  = BASE_NEW / split / "images"
    
    for lbl_file in orig_lbl_dir.glob("*.txt"):
        new_lines = []
        keep_file = False

        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                orig_id = int(parts[0])
                cname = orig_index_to_name[orig_id]

                if cname in mapping:
                    new_id = mapping[cname]
                    cx, cy, w, h = parts[1:]
                    new_lines.append(f"{new_id} {cx} {cy} {w} {h}\n")
                    keep_file = True

        if keep_file:
            # Guardar nueva etiqueta
            with open(new_lbl_dir / lbl_file.name, "w") as f:
                f.writelines(new_lines)

            # Copiar imagen correspondiente
            img_name = lbl_file.stem + ".jpg"
            orig_img_path = orig_img_dir / img_name

            if orig_img_path.exists():
                shutil.copy(orig_img_path, new_img_dir / img_name)

process_split("train")
process_split("valid")
process_split("test")

# Crear data.yaml nuevo
yaml_path = BASE_NEW / "data.yaml"
yaml_content = """train: ../train/images
val: ../valid/images
test: ../test/images

nc: 5
names: ['one', 'two', 'three', 'four', 'five']
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("\n✅ Conversión completa.")
print("Nuevo dataset listo en:", BASE_NEW)
print("Nuevo data.yaml creado.")