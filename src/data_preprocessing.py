from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def prepare_data(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.FakeData(transform=transform)
    val_dataset = datasets.FakeData(transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader