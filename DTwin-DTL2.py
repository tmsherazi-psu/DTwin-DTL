# Import required libraries
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# --- Data Preprocessing for Time-Series Dataset ---
def apply_clahe(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_clahe = cv2.merge((l, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)


# Dataset loader with CLAHE enhancement option for image data
def Dataset_loader(DIR, RESIZE, use_clahe=True):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
            img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)
            if use_clahe:
                img = apply_clahe(img)  # Apply CLAHE enhancement
            IMG.append(np.array(img))
    return IMG


# --- Time-Series Data Preprocessing ---
def load_and_preprocess_data(file_path):
    # Load time-series dataset
    df = pd.read_csv(file_path)

    # Handle missing data (for simplicity, fill with forward fill)
    df.fillna(method='ffill', inplace=True)

    # Convert 'date' column to datetime format for time-series analysis
    df['date'] = pd.to_datetime(df['date'])

    # Normalize features (only numerical columns except 'date' and 'oil_temperature')
    features = df.drop(columns=['oil_temperature', 'date'])
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Re-integrate the normalized features with the dataset
    df[features.columns] = normalized_features

    return df


# --- Dataset Loader for Time-Series Data ---
def load_time_series_data():
    df = load_and_preprocess_data("/path/to/data.csv")

    # Extract time-related features such as year, month, day, etc.
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek

    # Target: Oil temperature (e.g., predicting future temperature)
    X = df.drop(columns=['oil_temperature', 'date'])
    y = df['oil_temperature']

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split into training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


# --- Domain Adaptation Model (Using Informer) ---
class InformerModel(nn.Module):
    def __init__(self, input_dim=224, num_classes=1, d_model=512, num_heads=8, num_layers=6):
        super(InformerModel, self).__init__()
        # Assuming Informer is implemented or imported here
        self.informer = Informer(
            input_dim=input_dim,
            output_dim=num_classes,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )

    def forward(self, x):
        output = self.informer(x)
        return output


# --- Label Smoothing Loss ---
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, num_classes=4):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        targets = targets.view(-1)
        log_preds = F.log_softmax(inputs, dim=-1)
        nll_loss = F.nll_loss(log_preds, targets, reduction='none')
        smooth_loss = -log_preds.mean(dim=-1)
        loss = (1 - self.alpha) * nll_loss + self.alpha * smooth_loss
        return loss.mean()


# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10):
    device = next(model.parameters()).device
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if scheduler:
            scheduler.step()

    return train_losses, val_losses, train_accuracies, val_accuracies


# --- Evaluation Function ---
def evaluate_model(model, test_loader):
    y_true, y_pred = [], []

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute MSE, MAE, and MAPE
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')
    return mse, mae, mape


# --- Main Function ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InformerModel(input_dim=224, num_classes=1).to(device)
    criterion = LabelSmoothingCrossEntropy(alpha=0.1, num_classes=4)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.9999)

    X_train, X_test, y_train, y_test = load_time_series_data()

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler=scheduler, epochs=50
    )

    # Plotting Metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title("Accuracy Curve")
    plt.legend()
    plt.show()

    # Evaluate Model Performance
    mse, mae, mape = evaluate_model(model, test_loader)
    print(f'Final Evaluation Metrics -> MSE: {mse}, MAE: {mae}, MAPE: {mape}')


if __name__ == "__main__":
    main()
