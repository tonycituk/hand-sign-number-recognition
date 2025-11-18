import os
from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr

api_key = os.getenv("ROBOFLOW_API_KEY")
model = get_model(
    model_id="hand-7hx79/1",   # revisa que coincida EXACTO con lo que sale en Roboflow
    api_key=api_key            # o "rf_tu_key" directo
)
 
image = load_image_bgr("https://www.mashumano.org/images/personas_mayores_canva.png")

results = model.infer(image)[0]
results = sv.Detections.from_inference(results)
annotator = sv.BoxAnnotator(thickness=4)
annotated_image = annotator.annotate(image, results)
annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)
annotated_image = annotator.annotate(annotated_image, results)
sv.plot_image(annotated_image)