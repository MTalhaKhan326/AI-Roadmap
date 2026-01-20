import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

# 1. Initialize the Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="UyVGyGW3o5K5CpL7GXUt"
)

# 2. Define your image
IMAGE_NAME = "gun_man.jpg" 

# 3. Load and convert to Grayscale
image = cv2.imread(IMAGE_NAME)

# Check if image is color, then convert to gray
if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert back to BGR so the model (which expects 3 channels) can read it
    processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
else:
    processed_image = image

# 4. Run Inference 
result = CLIENT.infer(processed_image, model_id="pose-detection-icfy5/8")

# 5. Convert results into Supervision KeyPoints
keypoints = sv.KeyPoints.from_inference(result)

# 6. Create the Vertex Annotator
kp_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=5)

# 7. Apply the dots (NO KEYWORDS HERE)
# Passing (image, keypoints) directly as positional arguments
annotated_image = kp_annotator.annotate(processed_image.copy(), keypoints)

# 8. Save and Show
cv2.imwrite("keypoints_gray_result.jpg", annotated_image)
cv2.imshow("Keypoint Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()