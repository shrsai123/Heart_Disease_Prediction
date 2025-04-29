import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, y_train):
    """Train scikit-learn models."""
    model.fit(X_train, y_train)
    return model

# For PyTorch model
def train_pytorch_model(model, X_train_tensor, y_train_tensor, epochs=100, batch_size=16):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch+1) % 10 == 0:
            avg_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    return model
