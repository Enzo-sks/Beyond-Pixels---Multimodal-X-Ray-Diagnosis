import torch
from torch import nn
from timeit import default_timer as timer
from models.tinyvgg import TinyVGG
from data.datamodule import ArtishowDataModule

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += (preds.argmax(1) == y).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader)

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y)
            total_loss += loss.item()
            total_acc += (preds.argmax(1) == y).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader)

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = ArtishowDataModule(
        train_dir="Dataset/train",
        test_dir="Dataset/test",
        batch_size=12
    )
    data.setup()

    model = TinyVGG(3, 10, len(data.train_data.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    start = timer()
    for epoch in range(5):
        train_loss, train_acc = train_step(model, data.train_dataloader(), loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, data.test_dataloader(), loss_fn, device)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        print(f"Epoch {epoch} | Train {train_loss:.4f} acc {train_acc:.4f} | Test {test_loss:.4f} acc {test_acc:.4f}")

    end = timer()
    print(f"Training time: {end - start:.2f}s")

    torch.save(model.state_dict(), "tinyvgg_artishow.pth")
    return results

if __name__ == "__main__":
    train_model()