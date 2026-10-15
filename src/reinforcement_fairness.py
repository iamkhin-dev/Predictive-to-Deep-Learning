import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
from src.config import SEED, FIGURES_DIR

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = {(1, 1), (2, 2), (3, 1), (1, 3)}
        self.state = self.start
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action_idx):
        move = self.actions[action_idx]
        nxt = (self.state[0] + move[0], self.state[1] + move[1])
        if nxt[0] < 0 or nxt[0] >= self.size or nxt[1] < 0 or nxt[1] >= self.size: nxt = self.state
        if nxt in self.obstacles: nxt = self.state
        self.state = nxt
        if self.state == self.goal: return self.state, 10.0, True
        return self.state, -0.1, False

def run_q_learning(episodes=300, alpha=0.1, gamma=0.9):
    epsilons = [0.01, 0.1, 0.5]
    all_rewards = {}
    best_q_table = None
    env = GridWorld()
    rng = np.random.RandomState(SEED)
    for eps in epsilons:
        q_table = np.zeros((env.size, env.size, len(env.actions)))
        rewards_history = []
        for _ in range(episodes):
            state = env.reset()
            done = False
            tot_reward = 0.0
            steps = 0
            while not done and steps < 100:
                if rng.rand() < eps: action = rng.randint(len(env.actions))
                else: action = int(np.argmax(q_table[state[0], state[1]]))
                next_state, reward, done = env.step(action)
                best_next = np.max(q_table[next_state[0], next_state[1]])
                q_table[state[0], state[1], action] += alpha * (reward + gamma * best_next - q_table[state[0], state[1], action])
                state = next_state
                tot_reward += reward
                steps += 1
            rewards_history.append(tot_reward)
        all_rewards[f"epsilon_{eps}"] = rewards_history
        if eps == 0.1: best_q_table = q_table
    plot_q_learning_results(all_rewards, best_q_table, env)
    return best_q_table

def plot_q_learning_results(all_rewards, q_table, env):
    plt.figure(figsize=(8, 4))
    for label, rewards in all_rewards.items():
        smoothed = np.convolve(rewards, np.ones(10)/10, mode="valid")
        epsilon = label.split("_", 1)[1].replace(".", ",")
        plt.plot(smoothed, label=f"Epsilon = {epsilon}")
    plt.xlabel("Episodes")
    plt.ylabel("Moving average reward")
    plt.title("Q-learning: effect of the exploration parameter (epsilon)")
    plt.legend()
    plt.savefig(FIGURES_DIR / "qlearning_rewards.png", bbox_inches="tight")
    plt.close()
    arrows = ["→", "↓", "←", "↑"]
    grid_policy = np.full((env.size, env.size), " ", dtype=object)
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) == env.goal: grid_policy[r, c] = "G"
            elif (r, c) in env.obstacles: grid_policy[r, c] = "X"
            else: grid_policy[r, c] = arrows[np.argmax(q_table[r, c])]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(np.zeros((env.size, env.size)), cmap="binary")
    for (r, c), val in np.ndenumerate(grid_policy):
        ax.text(c, r, str(val), va="center", ha="center", fontsize=14, fontweight="bold")
    plt.title("Optimal final policy (epsilon = 0.1)")
    plt.savefig(FIGURES_DIR / "optimal_policy.png", bbox_inches="tight")
    plt.close()

def audit_fairness(best_model, X_test, y_test, sensitive_col):
    df_eval = X_test.copy()
    y_pred = best_model.predict(X_test)
    df_eval["y_true"] = y_test
    df_eval["y_pred"] = y_pred
    group_col = df_eval[sensitive_col].astype("string").fillna("Missing").astype(str)
    audit_results = {}
    for g in group_col.unique():
        mask = (group_col == g)
        if mask.sum() == 0: continue
        sub_true = df_eval.loc[mask, "y_true"]
        sub_pred = df_eval.loc[mask, "y_pred"]
        audit_results[g] = {
            "size": int(mask.sum()),
            "accuracy": round(float(accuracy_score(sub_true, sub_pred)), 4),
            "recall": round(float(recall_score(sub_true, sub_pred, zero_division=0)), 4),
            "selection_rate": round(float(np.mean(sub_pred == 1)), 4)
        }
    group_metrics = list(audit_results.values())
    if group_metrics:
        recalls = [values["recall"] for values in group_metrics]
        selection_rates = [values["selection_rate"] for values in group_metrics]
        accuracies = [values["accuracy"] for values in group_metrics]
        highest_selection_rate = max(selection_rates)
        audit_results["gaps"] = {
            "recall_gap": round(max(recalls) - min(recalls), 4),
            "selection_rate_gap": round(highest_selection_rate - min(selection_rates), 4),
            "accuracy_gap": round(max(accuracies) - min(accuracies), 4),
            "disparate_impact_ratio": round(min(selection_rates) / highest_selection_rate, 4) if highest_selection_rate else 1.0
        }
    return audit_results

def compute_drift_metrics(reference_data, current_data, model=None, reference_labels=None, current_labels=None):
    def calculate_psi(expected_counts, actual_counts):
        smoothing = 1e-6
        expected_perc = (expected_counts + smoothing) / (expected_counts.sum() + smoothing * len(expected_counts))
        actual_perc = (actual_counts + smoothing) / (actual_counts.sum() + smoothing * len(actual_counts))
        return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))

    def categorical_psi(expected, actual):
        expected = pd.Series(expected, dtype="string").fillna("<missing>")
        actual = pd.Series(actual, dtype="string").fillna("<missing>")
        categories = sorted(set(expected.unique()) | set(actual.unique()))
        expected_counts = expected.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
        actual_counts = actual.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
        return calculate_psi(expected_counts, actual_counts)

    numeric_drift = {}
    categorical_drift = {}
    for column in reference_data.select_dtypes(include=[np.number]).columns.intersection(current_data.columns):
        if not pd.api.types.is_numeric_dtype(current_data[column]):
            continue
        expected = reference_data[column].dropna().to_numpy(dtype=float)
        actual = current_data[column].dropna().to_numpy(dtype=float)
        if len(expected) and len(actual):
            bins = np.histogram_bin_edges(expected, bins=10)
            bins[0], bins[-1] = -np.inf, np.inf
            expected_counts, _ = np.histogram(expected, bins=bins)
            actual_counts, _ = np.histogram(actual, bins=bins)
            numeric_drift[column] = calculate_psi(expected_counts, actual_counts)

    categorical_columns = reference_data.select_dtypes(include=["object", "category", "bool"]).columns.intersection(current_data.columns)
    for column in categorical_columns:
        expected = reference_data[column]
        actual = current_data[column]
        if len(expected) and len(actual):
            categorical_drift[column] = categorical_psi(expected, actual)

    drift_report = {
        "numeric_feature_psi": numeric_drift,
        "categorical_feature_psi": categorical_drift
    }
    if model is not None:
        reference_predictions = model.predict(reference_data)
        current_predictions = model.predict(current_data)
        drift_report["prediction_psi"] = categorical_psi(reference_predictions, current_predictions)
        if reference_labels is not None and current_labels is not None:
            reference_labels = np.asarray(reference_labels)
            current_labels = np.asarray(current_labels)
            drift_report["performance_drift"] = {
                "reference_accuracy": round(float(accuracy_score(reference_labels, reference_predictions)), 4),
                "current_accuracy": round(float(accuracy_score(current_labels, current_predictions)), 4),
                "accuracy_change": round(float(accuracy_score(current_labels, current_predictions) - accuracy_score(reference_labels, reference_predictions)), 4),
                "reference_f1": round(float(f1_score(reference_labels, reference_predictions, zero_division=0)), 4),
                "current_f1": round(float(f1_score(current_labels, current_predictions, zero_division=0)), 4),
                "f1_change": round(float(f1_score(current_labels, current_predictions, zero_division=0) - f1_score(reference_labels, reference_predictions, zero_division=0)), 4)
            }
    return drift_report