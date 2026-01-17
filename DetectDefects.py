import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

# 1. Initialize the Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="UyVGyGW3o5K5CpL7GXUt"
)

# 2. Define your image file
IMAGE_NAME = "metal_image.jpg"

# 3. Run Detection Inference (Using your Detection Model ID)
result = CLIENT.infer(IMAGE_NAME, model_id="structural-defects-cmies/2")

# 4. Load the image using OpenCV
image = cv2.imread(IMAGE_NAME)

# 5. Convert results into Supervision Detections
detections = sv.Detections.from_inference(result)

# 6. Create Annotators for Bounding Boxes and Labels
# We removed the MaskAnnotator to focus strictly on detection
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 7. Apply the drawings (Boxes and Class Labels) to the image
annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# 8. Save the final detection result
cv2.imwrite("detected_defects.jpg", annotated_image)

# 9. Feedback and Display
print("✅ Detection complete! Check 'detected_defects.jpg'")
cv2.imshow("Object Detection Result", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()