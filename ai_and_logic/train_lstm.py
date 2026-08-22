import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
HDF5_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cheating_keypoints.h5'))
MODEL_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cheating_lstm.pt'))

SEQUENCE_LENGTH = 30   # 30-frame sliding windows (~1-2 seconds)
STEP_SIZE = 10         # Shift window by 10 frames to generate overlapping samples
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
NUM_CLASSES = 5        # Classes: 0: Normal, 1: Phone, 2: Head Turn, 3: Paper Pass, 4: Missing
CLASS_NAMES = {
    0: 'Normal',
    1: 'Phone',
    2: 'Head Turn',
    3: 'Paper Pass',
    4: 'Missing'
}


# --- 1. PYTORCH DATASET CLASS ---
class KeypointDataset(Dataset):
    def __init__(self, h5_path, seq_len=30, step_size=10):
        self.samples = []
        self.labels = []

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 dataset file not found at: {h5_path}")

        with h5py.File(h5_path, 'r') as h5_file:
            video_keys = list(h5_file.keys())
            print(f"Loaded {len(video_keys)} video sequence(s) from HDF5.")

            label_counts = {class_id: 0 for class_id in CLASS_NAMES}

            for key in video_keys:
                data = np.array(h5_file[key])  # Shape: (N_frames, 67)
                label = h5_file[key].attrs['label']
                label_counts[int(label)] = label_counts.get(int(label), 0) + 1

                num_frames = data.shape[0]
                if num_frames >= seq_len:
                    # Create overlapping sliding sequence windows
                    for start_idx in range(0, num_frames - seq_len + 1, step_size):
                        window = data[start_idx : start_idx + seq_len]
                        self.samples.append(window)
                        self.labels.append(label)

        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)
        print("Classes present in training data:")
        for class_id, class_name in CLASS_NAMES.items():
            print(f"  {class_id}: {class_name} ({label_counts.get(class_id, 0)} video sequence(s))")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# --- 2. LSTM MODEL ARCHITECTURE ---
class CheatingLSTM(nn.Module):
    def __init__(self, input_size=67, hidden_size=64, num_layers=2, num_classes=5):
        super(CheatingLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, (hn, cn) = self.lstm(x)
        # Take the output of the last sequence step
        last_step_out = out[:, -1, :]
        x = self.fc1(last_step_out)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


# --- 3. TRAINING LOOP ---
def train_model():
    print("=" * 60)
    print("LOADING DATASET FROM HDF5")
    print("=" * 60)

    dataset = KeypointDataset(HDF5_PATH, seq_len=SEQUENCE_LENGTH, step_size=STEP_SIZE)

    if len(dataset) == 0:
        print("[!] No sequences created. Ensure your video extractions have at least 30 frames.")
        return

    print(f"Generated {len(dataset)} sequence samples (Tensor Shape: {dataset.samples.shape}).")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = CheatingLSTM(input_size=67, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n" + "=" * 60)
    print("STARTING LSTM MODEL TRAINING")
    print("=" * 60)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save model weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Model weights saved successfully to: {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_model()