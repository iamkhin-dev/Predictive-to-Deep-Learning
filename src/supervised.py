import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from src.config import SEED, FIGURES_DIR

def build_preprocessor(X_train):
    num_cols = X_train.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_transformer, num_cols), ("cat", cat_transformer, cat_cols)])

def run_supervised_pipeline(X_train, X_test, y_train, y_test):
    preprocessor = build_preprocessor(X_train)
    models = {
        "DecisionTree": (DecisionTreeClassifier(random_state=SEED), {"clf__max_depth": [3, 5, 10, None]}),
        "KNN": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 5, 7, 11]}),
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=SEED), {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "GaussianNB": (GaussianNB(), {}),
        "SVM_RBF": (SVC(kernel="rbf", random_state=SEED), {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto"]}),
        "SVM_Linear": (SVC(kernel="linear", random_state=SEED), {"clf__C": [0.01, 0.1, 1.0]}),
        "RandomForest": (RandomForestClassifier(random_state=SEED), {"clf__n_estimators": [50, 100], "clf__max_depth": [5, 10]}),
        "GradientBoosting": (GradientBoostingClassifier(random_state=SEED), {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.05, 0.1]})
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    fitted_models = {}
    for name, (clf, params) in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        search = GridSearchCV(pipe, params, cv=cv, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        if hasattr(best, "predict_proba"): score = best.predict_proba(X_test)[:, 1]
        elif hasattr(best, "decision_function"): score = best.decision_function(X_test)
        else: score = y_pred
        results[name] = {
            "best_cv_params": search.best_params_,
            "cv_best_f1": round(float(search.best_score_), 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, score)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_score": score
        }
        fitted_models[name] = best
    plot_roc_curves(results, y_test)
    plot_validation_curve(preprocessor, X_train, y_train, cv)
    best_model_name = max(results, key=lambda name: results[name]["cv_best_f1"])
    best_preprocessor = fitted_models[best_model_name].named_steps["prep"]
    return results, fitted_models, best_preprocessor

def plot_roc_curves(results, y_test):
    plt.figure(figsize=(9, 7))
    model_labels = {
        "DecisionTree": "Decision tree",
        "KNN": "K-nearest neighbors",
        "LogisticRegression": "Logistic regression",
        "GaussianNB": "Gaussian naive Bayes",
        "SVM_RBF": "RBF-kernel SVM",
        "SVM_Linear": "Linear SVM",
        "RandomForest": "Random forest",
        "GradientBoosting": "Gradient boosting"
    }
    for name, m in results.items():
        fpr, tpr, _ = roc_curve(y_test, m["y_score"])
        plt.plot(fpr, tpr, label=f"{model_labels.get(name, name)} (AUC = {m['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve comparison")
    plt.legend(loc="lower right")
    plt.savefig(FIGURES_DIR / "roc_curves.png", bbox_inches="tight")
    plt.close()

def plot_validation_curve(preprocessor, X_train, y_train, cv):
    pipe = Pipeline([("prep", preprocessor), ("clf", DecisionTreeClassifier(random_state=SEED))])
    param_range = np.arange(1, 16)
    train_scores, val_scores = validation_curve(pipe, X_train, y_train, param_name="clf__max_depth", param_range=param_range, cv=cv, scoring="f1", n_jobs=-1)
    plt.figure(figsize=(8, 5))
    plt.plot(param_range, np.mean(train_scores, axis=1), label="Training score")
    plt.plot(param_range, np.mean(val_scores, axis=1), label="Cross-validation score")
    plt.xlabel("Maximum depth")
    plt.ylabel("F1 score")
    plt.title("Decision tree validation curve: bias and variance")
    plt.legend()
    plt.savefig(FIGURES_DIR / "validation_curve.png", bbox_inches="tight")
    plt.close()