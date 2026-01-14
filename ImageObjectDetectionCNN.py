from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load a pre-trained professional YOLO model (Nano version for speed)
model = YOLO('yolov8n.pt') 

# You can use any image URL from the internet!
# Example: A busy street or a park scene
image_path = 'football.jpg' 

# Run the detection
results = model(image_path)

# Show the results
for r in results:
    # This automatically draws boxes and labels on the image
    img_with_boxes = r.plot() 
    
    # Convert BGR (OpenCV) to RGB (Matplotlib)
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("Object Detection")
    plt.show()

# Print out what it found
for box in results[0].boxes:
    class_id = int(box.cls[0])
    label = model.names[class_id]
    confidence = float(box.conf[0])
    print(f"Detected: {label} with {confidence:.2f} confidence")
