from collections import defaultdict

from sklearn.inspection import permutation_importance

splits_file = Path("hpc_splits.pkl")

df = pd.read_csv("data/processed_dataset_wo_metals_w_even_more_qm2.csv")
X = df.drop(["solubility", "smiles", "canon_smiles", "id"], axis=1)
X_filtered = filter_X(X)
y = df["solubility"]

scoring = {"r2": "r2", "MSE": "neg_mean_squared_error"}

for fold_id, model in zip(df["fold_id"], df["estimator"], strict=True):
    frequency: dict[str, int] = defaultdict(int)
    importance_dict: dict[str, list[float]] = defaultdict(list)

    _, X_test, _, y_test = custom_train_test_split(splits_file, fold_id, X_filtered, y)

    PFI = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=20, random_state=9, n_jobs=n_cpus)

    features = X_test.columns
    importances = PFI["r2"].importances_mean

    fi_series = pd.Series(importances, index=features)
