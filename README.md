## Notas hand-sign-number-recognition
![](./0b3b90f3-2de9-42de-b693-3a71347a64c4_jpg.rf.e28aac7570155b9e40795247fa2ae9f5.jpg)

Descargar Dataset: https://universe.roboflow.com/new-workspace-5ucgu/hand-7hx79

Convertir Dataset
```bash
python3 -m venv venv
source venv/bin/activate
python convert.py
```

Entrenar (Msilicon Mac agregar device=mps)
```bash
yolo train model=yolo11n.pt data=ruta/al/dataset/convertido/data.yaml epochs=3 imgsz=640 
```

Demo
```bash
python camera.py
```

