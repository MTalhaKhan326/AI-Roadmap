import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

# 1. Initialize Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="UyVGyGW3o5K5CpL7GXUt"
)

IMAGE_PATH = "metal_image.jpg" 

# 2. Run Inference
result = CLIENT.infer(IMAGE_PATH, model_id="structural-condition-segmentation-avgxh/13")

# 3. Load image
image = cv2.imread(IMAGE_PATH)

# 4. Process Detections
detections = sv.Detections.from_inference(result)

# 5. Initialize Annotators
# We use MaskAnnotator for the colorful overlay
mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

# 6. Create the Visualization
# We chain the annotators to see both the Box and the Mask
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

# 7. Save as a RESULT file
cv2.imwrite("segmentation_result.jpg", annotated_image)
print("✅ Segmentation complete! Result saved as 'segmentation_result.jpg'")

# Show it on screen
cv2.imshow("Structural Segmentation", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()