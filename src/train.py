import torch
from torch.utils.data import DataLoader
from model import OmniFaceModel
from data_preprocessing import prepare_data

def train_model():
    # Prepare data
    train_loader, val_loader = prepare_data()

    # Initialize model
    model = OmniFaceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model_weights.pth")
    print("Training complete.")

if __name__ == "__main__":
    train_model()