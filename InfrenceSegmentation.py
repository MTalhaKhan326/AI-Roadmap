import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

# 1. Initialize
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="UyVGyGW3o5K5CpL7GXUt"
)

IMAGE_PATH = "metal_image.jpg" # Change this to your file name

# 2. Run Segmentation Inference
result = CLIENT.infer(IMAGE_PATH, model_id="structural-condition-segmentation-avgxh/13")

# 3. Load image
image = cv2.imread(IMAGE_PATH)

# 4. Convert results into Supervision Detections (Handles masks automatically)
detections = sv.Detections.from_inference(result)

# 5. Create Mask Annotator (This is the "Highlighter" for the shape)
mask_annotator = sv.MaskAnnotator()
# Optional: Keep the box too
box_annotator = sv.BoxAnnotator()

# 6. Apply the masks to the image
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

# 7. Save
cv2.imwrite("metal_image.jpg", annotated_image)
print("✅ Segmentation complete! Check 'metal_image.jpg'")