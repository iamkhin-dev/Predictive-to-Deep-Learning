import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.config import SEED, DEVICE, FIGURES_DIR

torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

class TabularMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

class ImageCNN(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

class SequenceModelRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class SequenceModelStatic(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(seq_len, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.fc(x)

def train_mlp_tabular(X_train_t, y_train, X_test_t, y_test, epochs=40):
    tx_train = torch.tensor(X_train_t, dtype=torch.float32)
    ty_train = torch.tensor(y_train, dtype=torch.long)
    tx_test = torch.tensor(X_test_t, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(tx_train, ty_train), batch_size=64, shuffle=True)
    model = TabularMLP(X_train_t.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(tx_test.to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

def train_cnn_experiment(train_loader, validation_loader, test_loader, epochs=10):
    configs = {
        "Baseline": {"use_dropout": False, "weight_decay": 0.0, "early_stop": False, "augmentation": False},
        "Dropout": {"use_dropout": True, "weight_decay": 0.0, "early_stop": False, "augmentation": False},
        "L2_EarlyStop": {"use_dropout": False, "weight_decay": 1e-3, "early_stop": True, "augmentation": False},
        "DataAugmentation": {"use_dropout": False, "weight_decay": 0.0, "early_stop": False, "augmentation": True}
    }
    history = {}
    trained_models = {}
    for name, cfg in configs.items():
        model = ImageCNN(use_dropout=cfg["use_dropout"]).to(
            DEVICE, memory_format=torch.channels_last
        )
        optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=cfg["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        tr_loss, val_loss, tr_acc, val_acc = [], [], [], []
        best_vloss = float("inf")
        best_state = None
        patience, wait = 3, 0
        for _ in range(epochs):
            model.train()
            tot_loss, corr, count = 0.0, 0, 0
            for bx, by in train_loader:
                bx = bx.to(DEVICE, memory_format=torch.channels_last)
                by = by.to(DEVICE)
                if cfg["augmentation"]:
                    batch_size = bx.shape[0]
                    angles = torch.empty(batch_size, device=DEVICE).uniform_(-0.15, 0.15)
                    translations = torch.empty(batch_size, 2, device=DEVICE).uniform_(-0.08, 0.08)
                    cosine = torch.cos(angles)
                    sine = torch.sin(angles)
                    theta = torch.zeros(batch_size, 2, 3, device=DEVICE)
                    theta[:, 0, 0] = cosine
                    theta[:, 0, 1] = -sine
                    theta[:, 0, 2] = translations[:, 0]
                    theta[:, 1, 0] = sine
                    theta[:, 1, 1] = cosine
                    theta[:, 1, 2] = translations[:, 1]
                    grid = F.affine_grid(theta, bx.size(), align_corners=False)
                    bx = F.grid_sample(bx, grid, mode="bilinear", padding_mode="border", align_corners=False)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item() * len(by)
                corr += (out.argmax(1) == by).sum().item()
                count += len(by)
            tr_loss.append(tot_loss / count)
            tr_acc.append(corr / count)
            model.eval()
            v_loss, v_corr, v_count = 0.0, 0, 0
            with torch.no_grad():
                for bx, by in validation_loader:
                    bx = bx.to(DEVICE, memory_format=torch.channels_last)
                    by = by.to(DEVICE)
                    out = model(bx)
                    v_loss += criterion(out, by).item() * len(by)
                    v_corr += (out.argmax(1) == by).sum().item()
                    v_count += len(by)
            cur_vloss = v_loss / v_count
            val_loss.append(cur_vloss)
            val_acc.append(v_corr / v_count)
            if cur_vloss < best_vloss:
                best_vloss = cur_vloss
                if cfg["early_stop"]:
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                wait = 0
            elif cfg["early_stop"]:
                wait += 1
                if wait >= patience:
                    break
        if cfg["early_stop"] and best_state is not None:
            model.load_state_dict(best_state)
        history[name] = {"train_loss": tr_loss, "val_loss": val_loss, "train_acc": tr_acc, "val_acc": val_acc}
        trained_models[name] = model
    plot_cnn_curves(history)
    compare_optimizers(train_loader)
    seq_res = run_sequence_benchmark()
    test_accuracy = {}
    for name, model in trained_models.items():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(DEVICE, memory_format=torch.channels_last)
                by = by.to(DEVICE)
                correct += (model(bx).argmax(1) == by).sum().item()
                total += len(by)
        test_accuracy[name] = round(correct / total, 4)
    return {
        "cnn_test_accuracy": test_accuracy,
        "sequence_mse_comparison": seq_res
    }

def plot_cnn_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    display_names = {
        "Baseline": "Baseline",
        "Dropout": "Dropout regularization",
        "L2_EarlyStop": "L2 regularization with early stopping",
        "DataAugmentation": "Data augmentation"
    }
    for name, data in history.items():
        ep = range(1, len(data["train_loss"]) + 1)
        display_name = display_names.get(name, name)
        ax1.plot(ep, data["train_loss"], linestyle="--", label=f"Training - {display_name}")
        ax1.plot(ep, data["val_loss"], linestyle="-", label=f"Validation - {display_name}")
        ax2.plot(ep, data["train_acc"], linestyle="--", label=f"Training - {display_name}")
        ax2.plot(ep, data["val_acc"], linestyle="-", label=f"Validation - {display_name}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs. validation loss (overfitting and regularization)")
    ax1.legend()
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training vs. validation accuracy")
    ax2.legend()
    plt.savefig(FIGURES_DIR / "regularization_curves.png", bbox_inches="tight")
    plt.close()

def compare_optimizers(train_loader, epochs=5):
    lrs = [0.5, 0.01, 0.0005]
    opt_hist = {}
    for lr in lrs:
        for opt_name in ["SGD_Momentum", "Adam"]:
            model = ImageCNN().to(DEVICE, memory_format=torch.channels_last)
            if opt_name == "SGD_Momentum": optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else: optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            losses = []
            for _ in range(epochs):
                model.train()
                ep_loss, count = 0.0, 0
                for bx, by in train_loader:
                    bx = bx.to(DEVICE, memory_format=torch.channels_last)
                    by = by.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    ep_loss += loss.item() * len(by)
                    count += len(by)
                losses.append(ep_loss / count)
            opt_hist[f"{opt_name}_lr_{lr}"] = losses
    plt.figure(figsize=(9, 5))
    optimizer_labels = {
        key: key.replace("SGD_Momentum", "SGD with momentum").replace("_lr_", " with learning rate ")
        for key in opt_hist
    }
    for k, v in opt_hist.items():
        losses = np.asarray(v, dtype=float)
        losses[~np.isfinite(losses)] = np.nan
        plt.plot(range(1, epochs + 1), np.ma.masked_invalid(losses), marker="o", label=optimizer_labels[k])
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Average loss (log scale)")
    plt.title("Optimizer and learning rate comparison (including divergence)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(FIGURES_DIR / "optimizer_comparison.png", bbox_inches="tight")
    plt.close()

def run_sequence_benchmark():
    seq_len, n_samples = 15, 800
    rng = np.random.RandomState(SEED)
    X_seq = rng.randn(n_samples, seq_len, 1).astype(np.float32)
    y_seq = np.array([[X_seq[i, -1, 0] - 2.0 * X_seq[i, -2, 0] + np.sin(X_seq[i, -3, 0])] for i in range(n_samples)], dtype=np.float32)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=SEED
    )
    tx_train = torch.tensor(X_train_seq).to(DEVICE)
    ty_train = torch.tensor(y_train_seq).to(DEVICE)
    tx_test = torch.tensor(X_test_seq).to(DEVICE)
    ty_test = torch.tensor(y_test_seq).to(DEVICE)
    rnn_model = SequenceModelRNN().to(DEVICE)
    static_model = SequenceModelStatic(seq_len).to(DEVICE)
    opt_rnn = optim.Adam(rnn_model.parameters(), lr=0.01)
    opt_static = optim.Adam(static_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for _ in range(80):
        opt_rnn.zero_grad()
        l_rnn = criterion(rnn_model(tx_train), ty_train)
        l_rnn.backward()
        opt_rnn.step()
        opt_static.zero_grad()
        l_stat = criterion(static_model(tx_train), ty_train)
        l_stat.backward()
        opt_static.step()
    with torch.no_grad():
        rnn_test_mse = criterion(rnn_model(tx_test), ty_test)
        static_test_mse = criterion(static_model(tx_test), ty_test)
    return {
        "rnn_mse": round(float(rnn_test_mse.item()), 4),
        "memoryless_mse": round(float(static_test_mse.item()), 4)
    }