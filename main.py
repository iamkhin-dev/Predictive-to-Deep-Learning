from src.data_loader import load_tabular_data, load_image_data
from src.supervised import run_supervised_pipeline
from src.unsupervised import run_unsupervised_analysis
from src.neural_networks import train_mlp_tabular, train_cnn_experiment
from src.reinforcement_fairness import run_q_learning, audit_fairness, compute_drift_metrics
from src.config import FIGURES_DIR, OUTPUTS_DIR
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import json
import time


def show_gui_dashboard(summary):
    from matplotlib.widgets import Button

    images = [
        ("ROC curves and AUC", "roc_curves.png"),
        ("Bias-variance trade-off", "validation_curve.png"),
        ("Clustering and silhouette", "elbow_silhouette.png"),
        ("Hierarchical dendrogram", "dendrogram.png"),
        ("PCA explained variance", "pca_variance.png"),
        ("PCA 2D clusters", "pca_clusters.png"),
        ("CNN regularization", "regularization_curves.png"),
        ("Optimizer comparison", "optimizer_comparison.png"),
        ("Q-learning rewards", "qlearning_rewards.png"),
        ("Optimal grid policy", "optimal_policy.png"),
        ("Metrics & fairness summary", None)
    ]

    fig = plt.figure(figsize=(14, 9), num="Experiment plots")
    ax_graph = fig.add_axes([0.33, 0.05, 0.65, 0.9])
    buttons = []

    def format_summary_text():
        best_m = summary["best_classic_model"]
        mlp_m = summary["mlp_tabular_model"]
        cnn_m = summary["cnn_and_sequences"]["cnn_test_accuracy"]
        cnn_str = " | ".join(f"{name}: {accuracy}" for name, accuracy in cnn_m.items())
        unsup = summary["unsupervised_metrics"]
        seq_m = summary["cnn_and_sequences"]["sequence_mse_comparison"]
        lines = [
            f"EXPERIMENTAL METRICS SUMMARY (Time: {summary['execution_time_s']} s)",
            "=" * 75,
            f"Best Classical Model : {best_m['name']} (F1: {best_m['f1']} | Acc: {best_m['accuracy']} | AUC: {best_m['roc_auc']})",
            f"Tabular MLP          : F1: {mlp_m['f1']} | Acc: {mlp_m['accuracy']} | Prec: {mlp_m['precision']} | Rec: {mlp_m['recall']} | AUC: {mlp_m['roc_auc']}",
            f"CNN test accuracy    : {cnn_str}",
            f"Unsupervised & PCA   : k={unsup['best_k']} | 90% Var Comp={unsup['n_components_90_var']} | ARI={unsup['ari']} | NMI={unsup['nmi']}",
            f"Sequence MSE         : RNN={seq_m['rnn_mse']} vs Memoryless={seq_m['memoryless_mse']}",
            "-" * 75,
            "SUPERVISED MODELS TEST SET PERFORMANCE:",
        ]
        for name, m in summary["supervised_models_comparison"].items():
            lines.append(f"  {name:<18} | Acc: {m['accuracy']:<6} | Prec: {m['precision']:<6} | Rec: {m['recall']:<6} | F1: {m['f1']:<6} | AUC: {m['roc_auc']:<6}")
        lines.append("-" * 75)
        lines.append("FAIRNESS AUDIT BY SENSITIVE ATTRIBUTE (personal_status):")
        for g, vals in summary["fairness_audit"].items():
            if g == "gaps":
                lines.append(f"  Gaps: Recall={vals['recall_gap']} | Selection={vals['selection_rate_gap']} | Accuracy={vals['accuracy_gap']} | DI={vals['disparate_impact_ratio']}")
            else:
                lines.append(f"  {g:<18} (n={vals['size']:<3}) | Acc: {vals['accuracy']:<6} | Recall: {vals['recall']:<6} | Sel.Rate: {vals['selection_rate']:<6}")
        drift = summary["drift_summary"]
        lines.append("-" * 75)
        lines.append(f"Prediction PSI: {drift.get('prediction_psi', 'Not available')}")
        lines.append(f"Performance drift: {drift.get('performance_drift', 'Not available')}")
        return "\n".join(lines)

    def show_graph(index):
        title, filename = images[index]
        ax_graph.clear()
        ax_graph.axis("off")
        if filename is None:
            ax_graph.text(0.02, 0.96, format_summary_text(), ha="left", va="top", family="monospace", fontsize=9.5, transform=ax_graph.transAxes, bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc"))
        else:
            image_path = FIGURES_DIR / filename
            if image_path.exists(): ax_graph.imshow(mpimg.imread(image_path))
            else: ax_graph.text(0.5, 0.5, "Plot not available", ha="center", va="center", transform=ax_graph.transAxes)
        fig.canvas.draw_idle()

    for index, (title, _) in enumerate(images):
        button_ax = fig.add_axes([0.02, 0.93 - index * 0.080, 0.28, 0.060])
        button = Button(button_ax, title)
        button.on_clicked(lambda _event, selected=index: show_graph(selected))
        buttons.append(button)

    show_graph(0)
    fig._graph_buttons = buttons
    manager = plt.get_current_fig_manager()
    window = getattr(manager, "window", None)
    if window is not None:
        if hasattr(window, "state"): window.state("zoomed")
        elif hasattr(window, "showMaximized"): window.showMaximized()
        elif hasattr(window, "Maximize"): window.Maximize(True)
    plt.show()


def start():
    start_time = time.perf_counter()

    with tqdm(total=7, bar_format="{percentage:3.0f}%|{bar:35}|") as pbar:
        X_train, X_test, y_train, y_test, sensitive_col = load_tabular_data()
        train_loader, validation_loader, test_loader = load_image_data()
        pbar.update(1)

        with threadpool_limits(limits=1):
            supervised_metrics, models, preprocessor = run_supervised_pipeline(X_train, X_test, y_train, y_test)

        best_classic_name = max(supervised_metrics.keys(), key=lambda k: supervised_metrics[k]["cv_best_f1"])
        best_classic_model = models[best_classic_name]
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        pbar.update(1)

        unsupervised_metrics = run_unsupervised_analysis(X_train_trans, y_train)
        pbar.update(1)

        mlp_metrics = train_mlp_tabular(X_train_trans, y_train, X_test_trans, y_test)
        pbar.update(1)

        dl_metrics = train_cnn_experiment(train_loader, validation_loader, test_loader)
        pbar.update(1)

        run_q_learning()
        pbar.update(1)

        fairness_report = audit_fairness(best_classic_model, X_test, y_test, sensitive_col)
        simulated_production_data = X_test.copy()
        drift_report = compute_drift_metrics(
            X_train,
            simulated_production_data,
            best_classic_model,
            y_train,
            y_test
        )
        pbar.update(1)

    elapsed_time = round(time.perf_counter() - start_time, 2)
    clean_supervised = {
        k: {m: (v.tolist() if hasattr(v, "tolist") else v) for m, v in d.items() if m not in ("y_prob", "y_score")}
        for k, d in supervised_metrics.items()
    }
    summary = {
        "supervised_models_comparison": clean_supervised,
        "best_classic_model": {
            "name": best_classic_name,
            "f1": supervised_metrics[best_classic_name]["f1"],
            "accuracy": supervised_metrics[best_classic_name]["accuracy"],
            "roc_auc": supervised_metrics[best_classic_name]["roc_auc"]
        },
        "mlp_tabular_model": mlp_metrics,
        "cnn_and_sequences": dl_metrics,
        "unsupervised_metrics": unsupervised_metrics,
        "fairness_audit": fairness_report,
        "drift_summary": drift_report,
        "execution_time_s": elapsed_time
    }
    with open(OUTPUTS_DIR / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal execution time: {elapsed_time} s")
    show_gui_dashboard(summary)


if __name__ == "__main__":
    start()