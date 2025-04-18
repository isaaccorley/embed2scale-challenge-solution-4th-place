import numpy as np
import pandas as pd


def create_submission_from_dict(emb_dict: dict[str, np.ndarray]):
    """Assume dictionary has format {hash-id0: embedding0, hash-id1: embedding1, ...}"""
    df_submission = pd.DataFrame.from_dict(emb_dict, orient="index")

    # Reset index with name 'id'
    df_submission.index.name = "id"
    df_submission.reset_index(drop=False, inplace=True)
    return df_submission


def validate_submission(path_to_submission: str, expected_embedding_ids: set, embedding_dim: int = 1024):
    # Load data
    df = pd.read_csv(path_to_submission, header=0)

    # Verify that id is in columns
    if "id" not in df.columns:
        raise ValueError("""Submission file must contain column 'id'.""")

    # Temporarily set index to 'id'
    df.set_index("id", inplace=True)

    # Check that all samples are included
    submitted_embeddings = set(df.index.to_list())
    n_missing_embeddings = len(expected_embedding_ids.difference(submitted_embeddings))
    if n_missing_embeddings > 0:
        raise ValueError(f"""Submission is missing {n_missing_embeddings} embeddings.""")

    # Check that embeddings have the correct length
    if len(df.columns) != embedding_dim:
        raise ValueError(
            f"""{embedding_dim} embedding dimensions, but provided embeddings have {len(df.columns)} dimensions."""
        )

    # Convert columns to float
    try:
        for col in df.columns:
            df[col] = df[col].astype(float)
    except Exception as e:
        raise ValueError(f"""Failed to convert embedding values to float.
    Check embeddings for any not-allowed character, for example empty strings, letters, etc.
    Original error message: {e}""")

    # Check if any NaNs
    if df.isna().any().any():
        raise ValueError("""Embeddings contain NaN values.""")

    # Successful completion of the function
    return True
