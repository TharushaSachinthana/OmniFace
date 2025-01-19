import torch
from model import OmniFaceModel
from data_preprocessing import prepare_data

def evaluate_model():
    _, val_loader = prepare_data()
    model = OmniFaceModel()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    evaluate_model()
