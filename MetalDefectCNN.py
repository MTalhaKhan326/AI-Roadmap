from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. Load the SPECIALIZED Metal Defect Model
# Note: You can find 'NEU-DET yolov8' weights on GitHub or Roboflow
# For this example, imagine you've downloaded 'metal_defect_best.pt'
model = YOLO('metal_defect_best.pt') 

# 2. Path to your new metal image
image_path = 'my_metal_test.jpg' 

# 3. Run the detection
results = model(image_path)

# 4. Visualize the "Where" (Bounding Boxes)
for r in results:
    # This generates the image with boxes around the defects
    img_with_boxes = r.plot() 
    
    # Convert BGR to RGB for showing it in your notebook/script
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("Metal Quality Inspection: Defect Detection")
    plt.show()

# 5. Get the specific location coordinates
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0] # The "Where" (Coordinates)
    label = model.names[int(box.cls[0])]
    conf = float(box.conf[0])
    
    print(f"FOUND: {label} at Location: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    print(f"Confidence: {conf:.2f}")