from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torchvision import datasets
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from src.config import SEED, DATA_DIR, FIGURES_DIR, OUTPUTS_DIR


def run_eda(df, target_col, sensitive_col, dataset_source=None, dataset_description=None, split_summary=None):
    def value_counts(series):
        return {
            "Missing" if pd.isna(value) else str(value): int(count)
            for value, count in series.value_counts(dropna=False).items()
        }

    def json_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    descriptive_statistics = df.describe(include="all").transpose().astype(object)
    descriptive_statistics = descriptive_statistics.where(pd.notna(descriptive_statistics), None)
    numeric_correlations = df.select_dtypes(include=[np.number]).corr().round(4).astype(object)
    numeric_correlations = numeric_correlations.where(pd.notna(numeric_correlations), None)
    target_distribution = value_counts(df[target_col])
    sensitive_distribution = value_counts(df[sensitive_col])
    report = {
        "dataset": {
            "source": dataset_source or "Not provided",
            "shape": list(df.shape),
            "sample_count": int(df.shape[0]),
            "feature_count": int(df.shape[1] - 1),
            "class_count": int(df[target_col].nunique(dropna=True)),
            "target_column": target_col,
            "sensitive_attribute": sensitive_col,
            "description": dataset_description or "Not provided"
        },
        "split_shapes": split_summary or {},
        "missing_values": {str(column): int(count) for column, count in df.isna().sum().items()},
        "descriptive_statistics": {
            str(column): {str(stat): json_value(value) for stat, value in values.items()}
            for column, values in descriptive_statistics.to_dict(orient="index").items()
        },
        "target_distribution": target_distribution,
        "sensitive_attribute_distribution": sensitive_distribution,
        "numeric_correlations": {
            str(column): {str(other): json_value(value) for other, value in values.items()}
            for column, values in numeric_correlations.to_dict(orient="index").items()
        }
    }
    report_path = OUTPUTS_DIR / "eda_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as existing_file:
            existing_report = json.load(existing_file)
        if "image_dataset" in existing_report:
            report["image_dataset"] = existing_report["image_dataset"]
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)

    for distribution, filename, title, xlabel in (
        (target_distribution, "eda_target_distribution.png", "Target class distribution", target_col),
        (sensitive_distribution, "eda_sensitive_distribution.png", "Sensitive attribute distribution", sensitive_col)
    ):
        plt.figure(figsize=(8, 5))
        plt.bar(distribution.keys(), distribution.values())
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Sample count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, bbox_inches="tight")
        plt.close()

    if len(numeric_correlations.columns) > 1:
        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(numeric_correlations.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_correlations.columns)), numeric_correlations.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(numeric_correlations.index)), numeric_correlations.index)
        ax.set_title("Numeric feature correlations")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "eda_numeric_correlations.png", bbox_inches="tight")
        plt.close(fig)
    return report

def load_tabular_data():
    dataset = fetch_openml(name="credit-g", version=1, as_frame=True, data_home=str(DATA_DIR))
    df = dataset.frame.copy()
    target_col = "class"
    sensitive_col = "personal_status"
    details = getattr(dataset, "details", {}) or {}
    dataset_source = {
        "provider": "OpenML",
        "dataset_id": details.get("id"),
        "dataset_name": details.get("name", "credit-g"),
        "dataset_url": getattr(dataset, "url", None),
        "original_source_url": details.get("original_data_url")
    }
    df[target_col] = df[target_col].map({"good": 0, "bad": 1}).astype(int)
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    run_eda(
        df,
        target_col,
        sensitive_col,
        dataset_source,
        getattr(dataset, "DESCR", None),
        {"X_train": list(X_train.shape), "X_test": list(X_test.shape), "y_train": [len(y_train)], "y_test": [len(y_test)]}
    )
    return X_train, X_test, y_train, y_test, sensitive_col

def load_image_data(batch_size=128, subset_size=2000, validation_fraction=0.2):
    train_full = datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=True)
    test_full = datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=True)
    rng = np.random.RandomState(SEED)
    selected_train_idx = rng.choice(len(train_full), subset_size, replace=False)
    test_idx = rng.choice(len(test_full), subset_size // 4, replace=False)
    validation_size = max(1, int(round(subset_size * validation_fraction)))
    validation_idx = rng.choice(selected_train_idx, validation_size, replace=False)
    train_idx = np.setdiff1d(selected_train_idx, validation_idx, assume_unique=True)

    def make_tensor_dataset(dataset, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        images = dataset.data.index_select(0, indices).unsqueeze(1).to(torch.float32) / 255.0
        images.sub_(0.2860).div_(0.3530)
        labels = dataset.targets.index_select(0, indices)
        return TensorDataset(images, labels)

    train_dataset = make_tensor_dataset(train_full, train_idx)
    validation_dataset = make_tensor_dataset(train_full, validation_idx)
    test_dataset = make_tensor_dataset(test_full, test_idx)
    report_path = OUTPUTS_DIR / "eda_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as report_file:
            eda_report = json.load(report_file)
    else:
        eda_report = {}
    eda_report["image_dataset"] = {
        "source": "torchvision.datasets.FashionMNIST",
        "original_source": "Fashion-MNIST",
        "source_url": "https://github.com/zalandoresearch/fashion-mnist",
        "original_train_samples": len(train_full),
        "original_test_samples": len(test_full),
        "image_shape": list(train_full.data.shape[1:]),
        "class_count": len(train_full.classes),
        "classes": train_full.classes,
        "selected_train_samples": int(subset_size),
        "train_samples_used": len(train_dataset),
        "validation_samples_used": len(validation_dataset),
        "test_samples_used": len(test_dataset)
    }
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(eda_report, report_file, indent=2, ensure_ascii=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader