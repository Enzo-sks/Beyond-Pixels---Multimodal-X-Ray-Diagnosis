import torch
import torchvision
from torchvision import transforms
from models.tinyvgg import TinyVGG

def predict_image(model_path, image_path, class_names):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyVGG(3, 10, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image = torchvision.io.read_image(image_path).float() / 255
    transform = transforms.Resize((64, 64))
    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        preds = model(image)
        probs = torch.softmax(preds, dim=1)
        label = probs.argmax(1).item()

    return class_names[label], probs.max().item()