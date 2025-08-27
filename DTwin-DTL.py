# Import Required Libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from informer import Informer  # Assuming the Informer model is available


# --- Gradient Reversal Layer (GRL) ---
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.lambda_
        return grad_input, None


# --- Domain Adaptation Model ---
class DomainAdaptationModel(nn.Module):
    def __init__(self, feature_extractor, domain_classifier, lambda_=0.5):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.domain_classifier = domain_classifier
        self.lambda_ = lambda_

    def forward(self, x):
        # Extract features using the feature extractor
        features = self.feature_extractor(x)

        # Apply Gradient Reversal Layer for domain adaptation
        grl_features = GradientReversalLayer.apply(features, self.lambda_)

        # Classify the domain
        domain_output = self.domain_classifier(grl_features)
        return domain_output, features


# --- Informer Model for Time-Series Data ---
class InformerModel(nn.Module):
    def __init__(self, input_dim=224, num_classes=4, d_model=512, num_heads=8, num_layers=6):
        super(InformerModel, self).__init__()
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


# --- Domain Classifier (for domain adaptation) ---
class DomainClassifier(nn.Module):
    def __init__(self, input_dim=256):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 1)  # Binary classification (Source vs Target domain)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# --- Combined Loss Function ---
class DTwinDTLLoss(nn.Module):
    def __init__(self, lambda_=0.5):
        super(DTwinDTLLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, regression_loss, domain_loss):
        # Total loss as a weighted sum of regression and domain loss
        return (1 - self.lambda_) * regression_loss + self.lambda_ * domain_loss


# --- Time-Series Dataset Class ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_column='oil_temperature', date_column='date', window_size=30):

        self.data = data
        self.target_column = target_column
        self.date_column = date_column
        self.window_size = window_size

        # Normalize features (using Min-Max scaling for time-series)
        self.scaler = MinMaxScaler()
        self.features = data.drop(columns=[self.target_column, self.date_column])
        self.features = self.scaler.fit_transform(self.features)

        # Handle missing data (simple forward fill or interpolation)
        self.data[self.target_column] = self.data[self.target_column].fillna(method='ffill')
        self.features = np.nan_to_num(self.features)

        # Prepare data for time-series prediction
        self.X, self.y = self.create_sequences()

    def create_sequences(self):
        X, y = [], []
        for i in range(self.window_size, len(self.data)):
            # Sequence of past `window_size` observations
            X.append(self.features[i - self.window_size:i])
            y.append(self.data[self.target_column].iloc[i])  # Target is oil temperature
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# Load dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing data (for simplicity, fill with forward fill)
    df.fillna(method='ffill', inplace=True)

    # Normalize features (only numerical columns except 'date' and 'oil_temperature')
    features = df.drop(columns=['oil_temperature', 'date'])
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Re-integrate the normalized features with the dataset
    df[features.columns] = normalized_features

    return df


# Example of loading the dataset
file_path = 'path_to_dataset.csv'  # Replace with actual path
df = load_and_preprocess_data(file_path)

# Split dataset into training and testing (80-20 split)
train_size = int(0.8 * len(df))
train_data = df[:train_size]
test_data = df[train_size:]

# Create DataLoader for time-series dataset
train_dataset = TimeSeriesDataset(train_data)
test_dataset = TimeSeriesDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --- Training Function with Learning Rate Scheduler ---
def train_model(model, train_loader, val_loader, regression_criterion, domain_criterion, optimizer, scheduler=None,
                num_epochs=50):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            domain_output, features = model(images)

            # Compute regression loss and domain loss
            regression_loss = regression_criterion(features, labels)  # Using features for regression task
            domain_loss = domain_criterion(domain_output, labels)  # Domain classification task
            total_loss = regression_loss + domain_loss  # Combine both losses

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            _, predicted = torch.max(features.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(dim=1)).sum().item()

        # Update the learning rate if scheduler is defined
        if scheduler:
            scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                domain_output, features = model(images)
                regression_loss = regression_criterion(features, labels)
                domain_loss = domain_criterion(domain_output, labels)
                val_loss += (regression_loss + domain_loss).item()
                _, predicted = torch.max(features.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.argmax(dim=1)).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    return train_losses, val_losses, train_accuracies, val_accuracies


# --- Evaluation Function (MSE, MAE, MAPE) ---
def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            domain_output, features = model(images)
            _, predicted = torch.max(features.data, 1)
            y_true.extend(labels.argmax(dim=1).cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute MSE, MAE, and MAPE
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}')
    return mse, mae, mape


# --- Sensitivity Analysis of Learning Rates ---
def plot_learning_rate_sensitivity():
    # Example of plotting MSE and MAE for different learning rates
    learning_rates = [0.00013, 0.00008, 0.0001]
    mse_values = []
    mae_values = []
    mape_values = []

    for lr in learning_rates:
        # Initialize and train the model for each learning rate
        model = DomainAdaptationModel(InformerModel(), DomainClassifier(input_dim=256))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

        train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model, train_loader, test_loader, nn.MSELoss(), nn.BCELoss(), optimizer, scheduler, num_epochs=50
        )

        mse, mae, mape = evaluate_model(model, test_loader)

        mse_values.append(mse)
        mae_values.append(mae)
        mape_values.append(mape)

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(learning_rates, mse_values, label="MSE")
    plt.title('Learning Rate vs MSE')

    plt.subplot(1, 3, 2)
    plt.plot(learning_rates, mae_values, label="MAE")
    plt.title('Learning Rate vs MAE')

    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, mape_values, label="MAPE")
    plt.title('Learning Rate vs MAPE')

    plt.show()


# --- Main Execution ---
def main():
    train_loader, val_loader = load_data()

    feature_extractor = InformerModel()  # Using Informer as the feature extractor
    domain_classifier = DomainClassifier(input_dim=256)  # Assuming features after extraction

    model = DomainAdaptationModel(feature_extractor, domain_classifier)

    # Loss Functions
    regression_criterion = nn.CrossEntropyLoss()  # For classification task
    domain_criterion = nn.BCELoss()  # Binary Cross Entropy for domain classification

    # Optimizer and Learning Rate Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, regression_criterion, domain_criterion, optimizer, scheduler, num_epochs=50
    )

    # Plotting and Evaluation
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.show()

    mse, mae, mape = evaluate_model(model, val_loader)
    print(f'Final Evaluation Metrics -> MSE: {mse}, MAE: {mae}, MAPE: {mape}')

    # Sensitivity analysis of learning rates
    plot_learning_rate_sensitivity()


if __name__ == "__main__":
    main()
