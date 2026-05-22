from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def summarize_history(path: Path, experiment_name: str, note: str):
    df = pd.read_csv(path)
    best = df.loc[df["test_acc"].idxmax()]
    final = df.iloc[-1]

    return {
        "experiment": experiment_name,
        "note": note,
        "best_epoch": int(best["epoch"]),
        "best_test_acc": float(best["test_acc"]),
        "best_test_loss": float(best["test_loss"]),
        "train_acc_at_best": float(best["train_acc"]),
        "final_epoch": int(final["epoch"]),
        "final_train_acc": float(final["train_acc"]),
        "final_test_acc": float(final["test_acc"]),
        "final_lr": float(final["lr"]),
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
        ("MLP baseline", log_dir / "mlp_history.csv", "MLP, Adam, no augmentation"),
        ("CNN main", log_dir / "cnn_main_history.csv", "CNN, SGD, augmentation, lr=0.1, cosine scheduler"),
        ("CNN no augmentation", log_dir / "cnn_no_aug_history.csv", "CNN, SGD, no augmentation, lr=0.1"),
        ("CNN lr=0.01", log_dir / "cnn_lr001_30ep_history.csv", "CNN, SGD, augmentation, lr=0.01, 30 epochs"),
        ("CNN lr=0.05", log_dir / "cnn_lr005_30ep_history.csv", "CNN, SGD, augmentation, lr=0.05, 30 epochs"),
        ("CNN lr=0.10", log_dir / "cnn_lr01_30ep_history.csv", "CNN, SGD, augmentation, lr=0.10, 30 epochs"),
    ]

    summaries = []
    histories = {}

    for name, path, note in experiments:
        if not path.exists():
            print(f"Missing: {path}. Skip {name}.")
            continue

        summary, df = summarize_history(path, name, note)
        summaries.append(summary)
        histories[name] = df

    if not summaries:
        raise FileNotFoundError("No experiment history files found in outputs/logs.")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(result_dir / "experiment_summary.csv", index=False)

    print("========== Experiment Summary ==========")
    print(summary_df.to_string(index=False))
    print(f"Saved to: {(result_dir / 'experiment_summary.csv').resolve()}")

    plt.figure(figsize=(10, 5))
    for name, df in histories.items():
        plt.plot(df["epoch"], df["test_acc"], label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Experiment Test Accuracy Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "experiment_test_accuracy_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(summary_df["experiment"], summary_df["best_test_acc"])
    plt.xlabel("Experiment")
    plt.ylabel("Best Test Accuracy")
    plt.title("Best Test Accuracy of Different Experiments")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.0)

    for idx, row in summary_df.iterrows():
        plt.text(
            idx,
            row["best_test_acc"] + 0.01,
            f"{row['best_test_acc']:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(fig_dir / "experiment_best_accuracy_bar.png", dpi=200)
    plt.close()

    print(f"Saved figures to: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
