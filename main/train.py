import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from stnet.model import ImprovedSTNet
from stnet.loss import DynamicWeightedLoss
from stnet.config import ModelConfig
from stnet.utils import initialize_model
from stnet.evaluate import evaluate_model

class MockSpatioTemporalDataset(Dataset):
    """Mock dataset mimicking the structure expected by ImprovedSTNet."""
    def __init__(self, csv_file: str, window_size: int = 12, predict_ahead: int = 24):
        df = pd.read_csv(csv_file)
        self.window_size = window_size
        self.predict_ahead = predict_ahead
        self.feature_columns = [
            'Class1_X_center', 'Class1_Y_center', 'Class1_Velocity',
            'Class2_X_center', 'Class2_Y_center', 'Class2_Velocity',
            'Relative_Distance', 'Relative_Angle', 'Angle_Between'
        ]
        self.data = df[self.feature_columns].values.astype(np.float32)
        self.cls_labels = df['process'].values.astype(np.int64)
        self.chase_labels = np.zeros(len(df), dtype=np.float32)
        self.attack_labels = np.zeros(len(df), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data) - self.window_size - self.predict_ahead + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window_end = idx + self.window_size
        return (
            torch.FloatTensor(self.data[idx:window_end]),
            torch.LongTensor([self.cls_labels[window_end - 1]]),
            torch.FloatTensor([self.chase_labels[window_end], self.attack_labels[window_end]])
        )

def main():
    """Train ImprovedSTNet on a mock dataset."""
    # Configuration
    config = ModelConfig(
        input_size=9,
        num_classes=3,
        batch_size=16,
        learning_rate=1e-5,
        epochs=5,
        class_weights=[1.0, 2.0, 3.0]
    )

    # Initialize model and components
    model = initialize_model(config.input_size, config.num_classes, config.hidden_size, config.dropout, config.device)
    criterion = DynamicWeightedLoss(
        class_weights=config.class_weights,
        chase_weight=config.chase_weight,
        attack_weight=config.attack_weight,
        reg_weight=config.reg_weight,
        label_smoothing=config.label_smoothing
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-3)

    # Load dataset
    dataset = MockSpatioTemporalDataset('examples/mock_data.csv')
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for features, cls_labels, reg_labels in loader:
            features = features.to(config.device)
            cls_labels = cls_labels.to(config.device).squeeze()
            reg_labels = reg_labels.to(config.device)

            optimizer.zero_grad()
            cls_pred, reg_pred = model(features)
            loss = criterion(cls_pred, reg_pred, cls_labels, reg_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {total_loss / len(loader):.4f}")

    # Evaluate
    metrics = evaluate_model(model, loader, criterion, config.device)
    print(f"Evaluation - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['cls_accuracy']:.3f}")

if __name__ == "__main__":
    main()