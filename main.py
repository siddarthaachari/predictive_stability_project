import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

from mealpy.swarm_based import PSO, GWO
from mealpy.evolutionary_based import GA
from mealpy.utils.space import FloatVar

from utils import load_preprocess


# ================================
# Fitness Function
# ================================
def fitness_function(solution):

    # Feature selection mask
    mask = solution > 0.5

    if np.sum(mask) == 0:
        return 1.0

    X_train_selected = X_train[:, mask]
    X_test_selected = X_test[:, mask]

    model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train_selected, y_train)

    pred = model.predict(X_test_selected)

    acc = accuracy_score(y_test, pred)

    return 1 - acc


# ================================
# Run optimizer
# ================================
def run_optimizer(name, optimizer, bounds):

    problem = {
        "obj_func": fitness_function,
        "bounds": bounds,
        "minmax": "min"
    }

    model = optimizer(epoch=20, pop_size=10)

    best = model.solve(problem)

    return best.solution


# ================================
# Plot ROC
# ================================
def plot_roc():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.savefig("results/ROC.png")
    plt.close()


# ================================
# Confusion Matrix
# ================================
def plot_confusion():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred)

    plt.figure()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("results/confusion_matrix.png")
    plt.close()


# ================================
# MAIN
# ================================
def main():

    global X_train, X_test, y_train, y_test

    os.makedirs("results", exist_ok=True)

    datasets = {
        "JM1": "data/jm1.csv",
        "CM1": "data/cm1.csv"
    }

    for name, path in datasets.items():

        print("\nDATASET:", name)

        X_train, X_test, y_train, y_test, feature_names = load_preprocess(path)

        dim = X_train.shape[1]

        bounds = FloatVar(
            lb=(0.0,) * dim,
            ub=(1.0,) * dim
        )

        print("Running PSO")
        pso_sol = run_optimizer("PSO", PSO.OriginalPSO, bounds)

        print("Running GA")
        ga_sol = run_optimizer("GA", GA.BaseGA, bounds)

        print("Running GWO")
        gwo_sol = run_optimizer("GWO", GWO.OriginalGWO, bounds)

        # ROC
        plot_roc()

        # Confusion matrix
        plot_confusion()

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()