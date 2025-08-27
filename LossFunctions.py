import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf  # For autocorrelation and partial autocorrelation
from sklearn.preprocessing import MinMaxScaler


# --- Autocorrelation and Partial Autocorrelation Preprocessing ---
def autocorrelation(series, lag=1):
    return acf(series, nlags=lag)[-1]  # Get the autocorrelation at the given lag


def partial_autocorrelation(series, lag=1):
    return pacf(series, nlags=lag)[-1]  # Get the partial autocorrelation at the given lag


# --- Time-Series Data Preprocessing ---
def load_and_preprocess_data(file_path):
    # Load the time-series dataset (e.g., CSV file)
    df = pd.read_csv(file_path)

    # Handle missing data (forward fill as an example)
    df.fillna(method='ffill', inplace=True)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Apply autocorrelation and partial autocorrelation analysis
    df['autocorrelation'] = df['oil_temperature'].apply(lambda x: autocorrelation(x, lag=1))
    df['partial_autocorrelation'] = df['oil_temperature'].apply(lambda x: partial_autocorrelation(x, lag=1))

    # Normalize features (excluding 'date' and 'oil_temperature')
    features = df.drop(columns=['oil_temperature', 'date'])
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Re-integrate the normalized features with the dataset
    df[features.columns] = normalized_features

    return df


# --- Dataset Loader for Time-Series Data ---
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path):
        self.df = load_and_preprocess_data(file_path)
        self.X = self.df.drop(columns=['oil_temperature', 'date']).values
        self.y = self.df['oil_temperature'].values

        # Split data into training and testing sets (80-20 split)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                shuffle=False)

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return torch.tensor(self.X_train[idx], dtype=torch.float32), torch.tensor(self.y_train[idx],
                                                                                  dtype=torch.float32)


# --- Regression Loss ---
class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.mean((inputs - targets) ** 2)


# --- Domain Classification Loss ---
class DomainClassificationLoss(nn.Module):
    def __init__(self):
        super(DomainClassificationLoss, self).__init__()

    def forward(self, inputs, targets):
        # Cross-entropy loss for domain classification
        return F.binary_cross_entropy_with_logits(inputs, targets)


# --- Combined Loss ---
class CombinedLoss(nn.Module):
    def __init__(self, lambda_):
        super(CombinedLoss, self).__init__()
        self.lambda_ = lambda_
        self.regression_loss = RegressionLoss()
        self.domain_loss = DomainClassificationLoss()

    def forward(self, outputs, targets, domain_labels):
        # Compute regression loss
        regression_loss = self.regression_loss(outputs, targets)

        # Compute domain classification loss
        domain_loss = self.domain_loss(outputs, domain_labels)

        # Total loss as a weighted sum of regression and domain loss
        total_loss = (1 - self.lambda_) * regression_loss + self.lambda_ * domain_loss
        return total_loss


# --- DTwin-DTL Model --- (Digital Twin Model)
class DTwinDTL(nn.Module):
    def __init__(self, num_classes=1):  # 1 class for regression (oil temperature)
        super(DTwinDTL, self).__init__()

        # Transformer Encoder for time-series processing
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8), num_layers=6
        )

        # Fully connected layers for regression
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass the time-series data through transformer encoder
        x = x.permute(1, 0, 2)  # Change shape for transformer (seq_len, batch_size, features)
        transformer_output = self.encoder(x)

        # Average pooling across the sequence dimension
        x = transformer_output.mean(dim=0)

        # Fully connected layers for regression
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# --- Agent Class for Training and Evaluation ---
class Agent_DTwinDTL:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(config['device'])
        self.config = config
        self.epoch = 0

    def train(self, data_loader, loss_function):
        self.model.train()
        for epoch in range(self.config['n_epoch']):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = loss_function(outputs, labels, domain_labels=torch.ones(inputs.size(0), 1).to(self.device))

                # Backward pass
                loss.backward()

                # Optimize
                self.optimizer.step()

                running_loss += loss.item()

                if i % self.config['save_interval'] == 0:  # Save model at intervals
                    print(f"Epoch [{epoch + 1}/{self.config['n_epoch']}], Step [{i + 1}], Loss: {loss.item():.4f}")

            # Step the scheduler
            self.scheduler.step()
            print(f'Epoch [{epoch + 1}/{self.config["n_epoch"]}], Average Loss: {running_loss / len(data_loader):.4f}')

    def evaluate(self, data_loader, loss_function):
        self.model.eval()
        total_dice_score = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute Dice score or any other evaluation metrics
                dice_score = self.compute_dice_score(outputs, labels)
                total_dice_score += dice_score

        average_dice_score = total_dice_score / len(data_loader)
        return average_dice_score

    def compute_dice_score(self, outputs, labels):
        smooth = 1e-5
        outputs = torch.sigmoid(outputs)
        intersection = (outputs * labels).sum()
        dice = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
        return dice.item()


# --- Configuration for Training ---
config = {
    'device': 'cuda:0',
    'lr': 1e-3,  # Default Learning rate set to 0.001
    'lr_gamma': 0.9999,
    'n_epoch': 35,  # 50 epochs as per the manuscript
    'batch_size': 32,
    'save_interval': 10,
    'evaluate_interval': 10,
    'optimizer': 'Adam',  # Adam optimizer
    'scheduler': 'CyclicLR',  # CyclicLR scheduler
    'loss_function': 'Combined Loss',  # Combined regression and domain loss
}

# --- Initialize Dataset and Dataloader ---
dataset = TimeSeriesDataset('path_to_your_time_series_data.csv')  # Update with your dataset path
device = torch.device(config['device'])
model = DTwinDTL(num_classes=1).to(device)  # 1 class for regression (oil temperature)

# Optimizer and Scheduler
optimizer = Adam(model.parameters(), lr=config['lr'])
scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, step_size_up=2000, mode='triangular2')

# Loss function
loss_function = CombinedLoss(lambda_=0.5)  # Adjust lambda based on the focus between regression and domain loss

# Initialize the agent for training
train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

# Train the model with different learning rates
learning_rates = [0.00013, 0.00008, 0.0001]

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, step_size_up=2000, mode='triangular2')

    # Re-initialize the agent with the new optimizer
    agent = Agent_DTwinDTL(model, optimizer, scheduler, config)

    # Train the model
    agent.train(train_loader, loss_function)

    # Evaluate the model
    agent.evaluate(test_loader, loss_function)
