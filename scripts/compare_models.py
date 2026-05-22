from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def read_history(path: Path, model_name: str):
    df = pd.read_csv(path)
    best_row = df.loc[df["test_acc"].idxmax()]

    return {
        "model": model_name,
        "best_epoch": int(best_row["epoch"]),
        "best_test_acc": float(best_row["test_acc"]),
        "best_test_loss": float(best_row["test_loss"]),
        "train_acc_at_best": float(best_row["train_acc"]),
        "train_loss_at_best": float(best_row["train_loss"]),
        "final_epoch": int(df["epoch"].iloc[-1]),
        "final_train_acc": float(df["train_acc"].iloc[-1]),
        "final_test_acc": float(df["test_acc"].iloc[-1]),
    }, df


def plot_accuracy_curves(histories, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for model_name, df in histories.items():
        plt.plot(df["epoch"], df["test_acc"], label=f"{model_name} Test Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Model Test Accuracy Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_best_accuracy_bar(summary_df: pd.DataFrame, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.bar(summary_df["model"], summary_df["best_test_acc"])
    plt.xlabel("Model")
    plt.ylabel("Best Test Accuracy")
    plt.title("Best Test Accuracy Comparison")
    plt.ylim(0, 1.0)

    for idx, row in summary_df.iterrows():
        plt.text(
            idx,
            row["best_test_acc"] + 0.01,
            f"{row['best_test_acc']:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    with Path("configs/cifar10.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log_dir = Path(cfg["outputs"]["logs"])
    result_dir = Path(cfg["outputs"]["results"])
    fig_dir = Path(cfg["outputs"]["figures"])

    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "MLP": log_dir / "mlp_history.csv",
        "CNN": log_dir / "cnn_history.csv",
    }

    summaries = []
    histories = {}

    for model_name, path in model_files.items():
        if not path.exists():
            print(f"Warning: missing history file: {path}")
            continue

        summary, history = read_history(path, model_name)
        summaries.append(summary)
        histories[model_name] = history

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(result_dir / "model_comparison.csv", index=False)

    if len(summary_df) >= 2:
        mlp_acc = float(summary_df.loc[summary_df["model"] == "MLP", "best_test_acc"].iloc[0])
        cnn_acc = float(summary_df.loc[summary_df["model"] == "CNN", "best_test_acc"].iloc[0])
        improvement = cnn_acc - mlp_acc
        relative = improvement / mlp_acc

        extra = pd.DataFrame([{
            "mlp_best_acc": mlp_acc,
            "cnn_best_acc": cnn_acc,
            "absolute_improvement": improvement,
            "relative_improvement": relative,
        }])
        extra.to_csv(result_dir / "mlp_vs_cnn_improvement.csv", index=False)

        print("========== MLP vs CNN Improvement ==========")
        print(extra.to_string(index=False))

    plot_accuracy_curves(
        histories,
        fig_dir / "model_test_accuracy_comparison.png",
    )

    plot_best_accuracy_bar(
        summary_df,
        fig_dir / "model_best_accuracy_bar.png",
    )

    print("========== Model Comparison ==========")
    print(summary_df.to_string(index=False))
    print(f"Saved comparison CSV: {(result_dir / 'model_comparison.csv').resolve()}")
    print(f"Saved figures to: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
