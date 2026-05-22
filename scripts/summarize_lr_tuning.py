from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def summarize_history(path: Path, lr: float):
    df = pd.read_csv(path)
    best = df.loc[df["test_acc"].idxmax()]
    final = df.iloc[-1]

    return {
        "learning_rate": lr,
        "history_file": str(path),
        "best_epoch": int(best["epoch"]),
        "best_test_acc": float(best["test_acc"]),
        "best_test_loss": float(best["test_loss"]),
        "train_acc_at_best": float(best["train_acc"]),
        "final_epoch": int(final["epoch"]),
        "final_train_acc": float(final["train_acc"]),
        "final_test_acc": float(final["test_acc"]),
        "final_lr_after_scheduler": float(final["lr"]),
    }, df


def main():
    with Path("configs/cifar10.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log_dir = Path(cfg["outputs"]["logs"])
    result_dir = Path(cfg["outputs"]["results"])
    fig_dir = Path(cfg["outputs"]["figures"])

    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        (0.01, log_dir / "cnn_lr001_30ep_history.csv"),
        (0.05, log_dir / "cnn_lr005_30ep_history.csv"),
        (0.10, log_dir / "cnn_lr01_30ep_history.csv"),
    ]

    summaries = []
    histories = {}

    for lr, path in experiments:
        if not path.exists():
            print(f"Missing: {path}. Skip lr={lr}.")
            continue

        summary, df = summarize_history(path, lr)
        summaries.append(summary)
        histories[lr] = df

    if not summaries:
        raise FileNotFoundError("No learning-rate tuning history files found.")

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values("learning_rate")
    summary_df.to_csv(result_dir / "lr_tuning_summary.csv", index=False)

    print("========== Learning Rate Tuning Summary ==========")
    print(summary_df.to_string(index=False))
    print(f"Saved to: {(result_dir / 'lr_tuning_summary.csv').resolve()}")

    plt.figure(figsize=(8, 5))
    for lr, df in histories.items():
        plt.plot(df["epoch"], df["test_acc"], label=f"lr={lr}")

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Learning Rate Tuning: Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "lr_tuning_accuracy_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    x_labels = [str(lr) for lr in summary_df["learning_rate"]]
    plt.bar(x_labels, summary_df["best_test_acc"])
    plt.xlabel("Initial Learning Rate")
    plt.ylabel("Best Test Accuracy")
    plt.title("Best Test Accuracy under Different Learning Rates")
    plt.ylim(0, 1.0)

    for idx, row in summary_df.reset_index(drop=True).iterrows():
        plt.text(
            idx,
            row["best_test_acc"] + 0.01,
            f"{row['best_test_acc']:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(fig_dir / "lr_tuning_best_accuracy_bar.png", dpi=200)
    plt.close()

    best_row = summary_df.loc[summary_df["best_test_acc"].idxmax()]
    print("========== Best Learning Rate ==========")
    print(best_row.to_string())


if __name__ == "__main__":
    main()
