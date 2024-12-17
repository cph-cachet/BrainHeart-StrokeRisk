import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
from tqdm import tqdm
import argparse
from dataloader import CustomDataset
from locallead import LocalLeadModel
from sklearn.metrics import mean_squared_error, mean_absolute_error


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--measure', type=str, required=True,
                        choices=['ecg', 'eeg', 'both'], help='Measure type (ecg, eeg, both)')
    args = parser.parse_args()

    path = ''  # new path
    splits = os.path.join('preprocess', 'train_splits.json')
    measure = args.measure
    batch_size = 128

    if measure == 'ecg':
        channels = 1
    elif measure == 'eeg':
        channels = 2
    elif measure == 'both':
        channels = 3
    else:
        raise ValueError(
            "Invalid measure type. Please choose 'ecg', 'eeg', or 'both'.")

    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = os.path.join('checkpoints', f'checkpoints_{measure}')
    print("Making of checkpoint_dir in", checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(splits, 'r') as f:
        splits = json.load(f)
        train_array = splits['train']
        test_array = splits['test']

    total_files = len(os.listdir(path))
    train_dataset = CustomDataset(path, train_array, measure, train=True)
    test_dataset = CustomDataset(path, test_array, measure, train=False)

    # Pass training set normalization parameters to test set
    test_dataset.set_normalization_params(
        train_dataset.age_min,
        train_dataset.age_max,
        train_dataset.data_min,
        train_dataset.data_max
    )

    # Create train and test dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    num_epochs = 50
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LocalLeadModel(num_classes=1, channels=channels)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': [],
        'train_mae': [],
        'val_mae': []
    }

    start_epoch = 0
    best_val_mse = float('inf')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        train_predictions = []
        train_true = []

        for batch_idx, (data, labels, _) in enumerate(tqdm(train_dataloader)):
            data, labels = data.to(device).float(), labels.to(
                device).float()

            outputs = model(data).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_true.extend(labels.cpu().numpy())

        model.eval()
        val_loss = 0
        val_predictions = []
        val_true = []

        with torch.no_grad():
            for data, labels, _ in test_dataloader:
                data, labels = data.to(
                    device).float(), labels.to(device).float()
                outputs = model(data).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_dataloader)
        train_mse = mean_squared_error(train_true, train_predictions)
        train_mae = mean_absolute_error(train_true, train_predictions)

        val_loss = val_loss / len(test_dataloader)
        val_mse = mean_squared_error(val_true, val_predictions)
        val_mae = mean_absolute_error(val_true, val_predictions)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(float(train_mse))
        history['val_mse'].append(float(val_mse))
        history['train_mae'].append(float(train_mae))
        history['val_mae'].append(float(val_mae))

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(
            f'Train Loss: {train_loss:.4f}, Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}')
        print(
            f'Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_val_mse': best_val_mse
        }
        latest_checkpoint_path = os.path.join(
            checkpoint_dir, f'model_checkpoint_{measure}.pth')
        torch.save(checkpoint_dict, latest_checkpoint_path)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint_dict, checkpoint_path)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            print(f"New best model with validation MSE: {best_val_mse:.4f}")
            torch.save(checkpoint_dict, f'best_model_{measure}.pth')

    with open(f'training_history_{measure}.json', 'w') as f:
        json.dump(history, f)


if __name__ == "__main__":
    main()
