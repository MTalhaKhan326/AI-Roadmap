import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

# 1. Initialize the Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="UyVGyGW3o5K5CpL7GXUt"
)

IMAGE_NAME = "metal_image.jpg"

# 2. Run Detection Inference with a Confidence Threshold
# We set confidence=30 to ensure we see even minor defects
result = CLIENT.infer(IMAGE_NAME, model_id="structural-defects-cmies/2")

# 3. Load and process
image = cv2.imread(IMAGE_NAME)
detections = sv.Detections.from_inference(result)

# 4. Filter Detections (Optional)
# This ensures we only show things the model is at least 40% sure about
detections = detections[detections.confidence > 0.4]

# 5. Create Enhanced Annotators
# We add thickness=4 to make sure the boxes are clearly visible on high-res images
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_scale=1.5)

# 6. Apply Annotations
annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# 7. Save and Verify
if len(detections) == 0:
    print("⚠️ No defects detected. Try lowering the confidence threshold.")
else:
    print(f"✅ Successfully detected {len(detections)} defects!")
    cv2.imwrite("detected_defects.jpg", annotated_image)
    cv2.imshow("Object Detection Result", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()