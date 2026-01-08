import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import requests

# 1. Load a pre-trained model (MobileNet is fast and lightweight)
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)
model.eval() # Set to evaluation mode

# 2. Define the transformation (AI needs images to be a specific size/format)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0) # Create a batch of 1

    # 3. The Forward Pass (The AI "thinks")
    with torch.no_grad():
        output = model(batch_t)

    # 4. Get the result
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    category_id = probabilities.argmax().item()
    label = weights.meta["categories"][category_id]
    confidence = probabilities[category_id].item()

    print(f"Prediction: {label} ({confidence*100:.2f}%)")


# To use: predict_image('your_image.jpg')
# Replace 'test.jpg' with the name of an actual image on your computer
image_to_test = 'image.jpg' 
predict_image(image_to_test)