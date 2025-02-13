from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from pathlib import Path
import numpy as np
from ast import literal_eval

part_path = Path("part-2")
raw_path = Path(f"{part_path}/raw")
processed_path = Path(f"{part_path}/processed")
submission_path = Path(f"{part_path}/submission")

# helper
def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)
    
def process_embeddings(embedding_series):
    return np.array([normalize_l2(literal_eval(emb)[:128]) for emb in embedding_series])

# load data
def load_data():
    df_train = pd.read_csv(f"{processed_path}/train-weighted.csv")
    df_test = pd.read_csv(f"{processed_path}/test-weighted.csv")
    return df_train, df_test

# process embeddings
def process_embeddings(df_embs):
    embeddings_a = process_embeddings(df_embs.embedding)
    embeddings_b = process_embeddings(df_embs.embedding_b)
    features = np.hstack([
        embeddings_a,
        embeddings_b
    ])
    return features

# features
features = [
   "size", 
   "size_b", 
   "size_ratio",
   "stars", 
   "stars_b", 
   "stars_ratio",
   "watchers",
   "watchers_b",
   "watchers_ratio",
   "forks", 
   "forks_b", 
   "forks_ratio", 
   "open_issues", 
   "open_issues_b", 
   "issues_ratio",
   "subscribers_count", 
   "subscribers_count_b",  
   "subscribers_ratio",
   "commit_code",
   "commit_code_b",
   "commits_ratio",
   "forked",
   "forked_b",
   "forked_ratio",
   "issue_closed",
   "issue_closed_b",
   "issue_closed_ratio",
   "issue_comment",
   "issue_comment_b",
   "issue_comment_ratio",
   "issue_opened",
   "issue_opened_b",
   "issue_opened_ratio",
   "issue_reopened",
   "issue_reopened_b",
   "issue_reopened_ratio",
   "pull_request_closed",
   "pull_request_closed_b",
   "pull_request_closed_ratio",
   "pull_request_merged",
   "pull_request_merged_b",
   "pull_request_merged_ratio",
   "pull_request_opened",
   "pull_request_opened_b",
   "pull_request_opened_ratio",
   "pull_request_reopened",
   "pull_request_reopened_b",
   "pull_request_reopened_ratio",
   "pull_request_review_comment",
   "pull_request_review_comment_b",
   "pull_request_review_comment_ratio",
   "release_published",
   "release_published_b",
   "release_published_ratio",
   "starred",
   "starred_b",
   "starred_ratio",
   "v_index",
   "v_index_b",
   "v_index_ratio",
   "stars_intersection_v_index",
   "stars_b_intersection_v_index_b",
   "stars_ratio_intersection_v_index_ratio",
   "num_dependents",
   "num_dependents_b",
   "dependency_rank",
   "dependency_rank_b",
   "num_dependents_ratio",
]

def main():
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    import datetime

    # load data
    df_train, df_test = load_data()

    # process embeddings
    loaded = np.load(f'{processed_path}/arrays.npz')
    train_features = loaded['train_features']
    test_features = loaded['test_features']

    # stack features
    train_features = np.hstack([
        train_features,
        df_train[features].to_numpy()
    ])

    test_features = np.hstack([
        test_features,
        df_test[features].to_numpy()
    ])

    # train
    X = train_features
    y = df_train["weight_a"].to_numpy()

    # train model
    lgb_train_data = lgb.Dataset(X, label=y)

    # Define parameters
    params = {
        "objective": "regression",
        "metric": "mse",
        "force_col_wise": True,
        "num_leaves": 100,
    }

    # Perform 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create training and validation datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        # Train model
        model = lgb.train(
            params, 
            train_data, 
            valid_sets=[val_data],
            # num_boost_round=1000,
            # early_stopping_rounds=50,
            # verbose_eval=False
        )

        # Make predictions and calculate MSE
        y_pred = model.predict(X_val)
        mse = np.mean((y_val - y_pred) ** 2)
        cv_scores.append(mse)

    # Calculate mean and std of MSE scores
    cv_scores = np.array(cv_scores)
    mean_mse = cv_scores.mean()
    std_mse = cv_scores.std()

    print(f"Cross-validation MSE: {mean_mse:.4f} (+/- {std_mse:.4f})")

    # Train model on the entire dataset
    model = lgb.train(
        params,
        lgb_train_data,
    )

    X_test = test_features

    test_predictions = model.predict(X_test)
    test_predictions = pd.Series(test_predictions.tolist()).round(6).clip(0.000001, 0.999999)

    # save predictions
    df_submission = df_test[["id"]].copy()  # Select "id" column
    df_submission["pred"] = test_predictions  # Add predictions column

    # Create filename with timestamp and MSE
    filename = f"{submission_path}/submission_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-mse_{mean_mse:.6f}.csv"

    # save predictions
    df_submission.to_csv(filename, index=False)

    print(f"Saved file: {filename}")

if __name__ == "__main__":
    main()
