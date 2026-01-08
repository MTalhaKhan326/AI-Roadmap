import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 1. Load a pre-trained ResNet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 2. Preprocessing (Preparing the image for the layers)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def trace_image_path(img_path):
    input_image = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) 

    print(f"{'#'*10} IMAGE PATHWAY START {'#'*10}\n")
    
    # We use a 'hook' or a simple loop to show the sequence of layers
    current_data = input_batch
    
    # Printing the journey through the main architectural blocks
    with torch.no_grad():
        # Layer 1: The Initial Convolution "Eyes"
        print(f"Step 1: Passing through Input Conv Layer -> Output Shape: {model.conv1(current_data).shape}")
        current_data = model.maxpool(torch.relu(model.bn1(model.conv1(current_data))))
        
        # Layer 2-5: The "Deep" Feature Blocks
        print(f"Step 2: Passing through Layer Block 1 (Finding edges/textures) -> Shape: {model.layer1(current_data).shape}")
        current_data = model.layer1(current_data)
        
        print(f"Step 3: Passing through Layer Block 2 (Finding patterns) -> Shape: {model.layer2(current_data).shape}")
        current_data = model.layer2(current_data)
        
        print(f"Step 4: Passing through Layer Block 3 (Finding parts) -> Shape: {model.layer3(current_data).shape}")
        current_data = model.layer3(current_data)
        
        print(f"Step 5: Passing through Layer Block 4 (Finding objects) -> Shape: {model.layer4(current_data).shape}")
        current_data = model.layer4(current_data)

        # Final Step: Flattening and the "Decision" Layer
        current_data = model.avgpool(current_data)
        current_data = torch.flatten(current_data, 1)
        output = model.fc(current_data)
        
        print(f"\nStep 6: Final Decision Layer (Fully Connected) -> Output: 1000 possible classes")
        
    print(f"\n{'#'*10} IMAGE PATHWAY COMPLETE {'#'*10}")

trace_image_path('image.jpg')