import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import numpy as np

part_path = Path("part-2")
raw_path = Path(f"{part_path}/raw")
processed_path = Path(f"{part_path}/processed")
submission_path = Path(f"{part_path}/submission")

def stringify_array(arr):
    return "'" + "','".join(arr) + "'"

def remove_str(projects):
    projects = [p.replace('https://github.com/', '') for p in projects]
    return projects

def remove_str_df(df):
    df['project_a'] = df['project_a'].str.replace('https://github.com/', '')
    df['project_b'] = df['project_b'].str.replace('https://github.com/', '')
    return df

def get_dataset():
    if str(part_path) == 'part-1':
        df_train = pd.read_csv(f"{part_path}/train.csv")
        df_test = pd.read_csv(f"{part_path}/test.csv")
        
    elif str(part_path) == 'part-2':
        df_train = pd.read_csv(f"{part_path}/train.csv")
        df_test = pd.read_csv(f"{part_path}/test.csv")
     
    return df_train, df_test

df_train, df_test = get_dataset()

projects = pd.concat([
    df_train['project_a'],
    df_train['project_b'],
    df_test['project_a'],
    df_test['project_b']
]).unique().tolist()

trimed_projects = remove_str(projects)

df_train = pd.read_csv(f"{processed_path}/train-pre-embeddings.csv")
df_test = pd.read_csv(f"{processed_path}/test-pre-embeddings.csv")


# Create quarter to index mapping
quarters = df_train.quarter.unique().tolist() + df_test.quarter.unique().tolist()
quarters.sort()  # Sort chronologically
quarter_to_index = {quarter: idx for idx, quarter in enumerate(quarters, 1)}

# Calculate weights as before
indices = np.arange(1, 36)
weights = 0.5 + ((indices - 1) / 34) * 0.5
weights_list = weights.tolist()

# Create quarter to weight mapping
quarter_to_weight = {quarter: weights_list[idx-1] for quarter, idx in quarter_to_index.items()}

# Example usage:
print("Quarter mappings:")
for quarter, weight in quarter_to_weight.items():
    print(f"{quarter}: {weight:.4f}")
base_features = [
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
# Get the weights for each row based on quarter
weights = df_train['quarter'].map(quarter_to_weight)

# Apply weights to features
for base in base_features:
    # Weight features ending in _a
    df_train[f'{base}'] = df_train[f'{base}'] * weights
    
# Do the same for test data
weights_test = df_test['quarter'].map(quarter_to_weight)
for base in base_features:
    df_test[f'{base}'] = df_test[f'{base}'] * weights_test
    
# Display example to verify
print("Sample of weighted features:")
sample_cols = ['quarter', 'stars', 'stars_b']
print(df_train[sample_cols].head())

df_train.to_csv(f"{processed_path}/train-weighted.csv", index=False)
df_test.to_csv(f"{processed_path}/test-weighted.csv", index=False)