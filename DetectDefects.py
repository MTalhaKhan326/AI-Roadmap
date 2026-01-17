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

# 3. Run Inference
result = CLIENT.infer(IMAGE_NAME, model_id="structural-defects-cmies/2")


if 'predictions' in result and len(result['predictions']) > 0:
    # Get the unique class names from the objects found in the image
    found_classes = set(p['class'] for p in result['predictions'])
    print(f"✅ In this image, the model found: {found_classes}")
else:
    print("No objects detected in this image, so I can't see the class names yet.")

# 4. Load the image using OpenCV (to draw on it)
image = cv2.imread(IMAGE_NAME)

# 5. Convert results into Supervision Detections
detections = sv.Detections.from_inference(result)

# 6. Create Annotators (The "Pens" that draw the boxes and labels)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 7. Apply the drawings to the image
annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# 8. Save the final result
cv2.imwrite("metal_image.jpg", annotated_image)

# 9. (Optional) Pop up a window to see it now
print("✅ Success! Check 'metal_image.jpg' in your folder.")
cv2.imshow("Defect Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()