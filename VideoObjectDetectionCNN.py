from ultralytics import YOLO
import cv2

# 1. Load the model
model = YOLO('yolov8n.pt') 

# 2. Open the camera
cap = cv2.VideoCapture(0)

# We use a 'Set' because sets only store unique values
unique_person_ids = set()

print("Camera Live. Press 'q' to finish and see the total count.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Use .track() instead of just predicting
    # persist=True tells the model to remember IDs across frames
    results = model.track(frame, persist=True, classes=None) # class 0 is 'person' in YOLO

    for r in results:
        # Draw the boxes with the ID numbers (e.g. "Person 1")
        annotated_frame = r.plot()
        
        # 4. Check if the model found any IDs
        if r.boxes.id is not None:
            # Get the IDs as a list of integers
            ids = r.boxes.id.int().tolist()
            
            # Add these IDs to our set
            for obj_id in ids:
                unique_person_ids.add(obj_id)
        
        # Display the current "Live Count" on the screen
        cv2.putText(annotated_frame, f"Unique People: {len(unique_person_ids)}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("YOLO Real-Time Tracker", annotated_frame)

    # PRESS 'q' TO STOP
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Final Summary
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*30)
print(f"FINAL REPORT: {len(unique_person_ids)} unique people were seen.")
print("="*30)