import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

# 1. Initialize the Client for OCR
# Using your specific model: ocr_testt/1
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="UyVGyGW3o5K5CpL7GXUt"
)

# 2. Define the path to your patient test report
IMAGE_NAME = "container_image.jpg" 

# 3. Load the image with OpenCV
image = cv2.imread(IMAGE_NAME)

# 4. Run Inference using the OCR model
# This fulfills the "Sensors: OCR scanner" requirement 
result = CLIENT.infer(image, model_id="ocr_testt/1")

# 5. Extract Text and Coordinates
# In your project, only test reports are accessed via OCR API [cite: 154]
# We use sv.Detections to handle the bounding boxes of the detected text
detections = sv.Detections.from_inference(result)

# 6. Initialize Annotators to visualize the OCR results
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Prepare labels (the actual text found by the OCR)
labels = [
    f"{prediction['class']} {prediction['confidence']:.2f}"
    for prediction in result['predictions']
]

# 7. Apply annotations to the report image
annotated_image = box_annotator.annotate(
    scene=image.copy(), 
    detections=detections
)
annotated_image = label_annotator.annotate(
    scene=annotated_image, 
    detections=detections, 
    labels=labels
)

# 8. Log the Analysis
# This fulfills the agent's goal to "analyze test report accurately" [cite: 149]
print("--- OCR Analysis Summary ---")
for label in labels:
    print(f"Detected Data: {label}")

# 9. Save and Show the Result
cv2.imwrite("ocr_result.jpg", annotated_image)
cv2.imshow("MEDICA OCR Scanner", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()