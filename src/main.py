import argparse
from src.sidequests.eedi_mmim.src.data_loader import (
    load_train_data,
    load_test_data,
    load_misconception_mapping,
    load_sample_submission
)
from src.sidequests.eedi_mmim.src.mmim_constants import mmim_data_path
from src.sidequests.eedi_mmim.src.visualizations import create_visualizations
from src.sidequests.eedi_mmim.src.mmim_task import MMIMTask

from loguru import logger
import pandas as pd
import numpy as np


def convert_models_to_dataframe(models):
    """Convert a list of Pydantic models to a pandas DataFrame"""
    data = [model.dict() for model in models]
    return pd.DataFrame(data)


def generate_train_statistics(df: pd.DataFrame):
    logger.info("Generating statistics for Training Data")
    logger.info(f"Number of unique Questions: {df['QuestionId'].nunique()}")
    logger.info(f"Number of unique Constructs: {df['ConstructId'].nunique()}")
    logger.info(f"Number of unique Subjects: {df['SubjectId'].nunique()}")
    logger.info("Distribution of Correct Answers:")
    logger.info(df['CorrectAnswer'].value_counts().to_dict())
    logger.info("Number of Misconceptions per Answer Option:")
    misconceptions = ['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']
    for mc in misconceptions:
        logger.info(f"{mc}: {df[mc].notna().sum()}")

    logger.info("Top 5 most common Constructs:")
    logger.info(df['ConstructName'].value_counts().head().to_dict())

    logger.info("Top 5 most common Subjects:")
    logger.info(df['SubjectName'].value_counts().head().to_dict())

    logger.info("Questions with most misconceptions:")
    misconception_counts = df[misconceptions].notna().sum(axis=1)
    top_misconception_questions = misconception_counts.nlargest(5)
    for idx, count in top_misconception_questions.items():
        question = df.loc[idx, 'QuestionText']
        logger.info(f"Question (ID: {df.loc[idx, 'QuestionId']}) with {count} misconceptions: {question[:100]}...")

    logger.info("Average number of misconceptions per question:")
    logger.info(f"{misconception_counts.mean():.2f}")

    logger.info("Distribution of number of misconceptions per question:")
    logger.info(misconception_counts.value_counts().sort_index().to_dict())

    logger.info("Top 5 Constructs with most misconceptions:")
    construct_misconceptions = df[misconceptions].notna().sum().groupby(df['ConstructName']).sum()
    logger.info(construct_misconceptions.nlargest(5).to_dict())

    logger.info("Top 5 Subjects with most misconceptions:")
    subject_misconceptions = df[misconceptions].notna().sum().groupby(df['SubjectName']).sum()
    logger.info(subject_misconceptions.nlargest(5).to_dict())

    logger.info("Correlation between number of misconceptions and subject/construct:")
    df['misconception_count'] = misconception_counts
    subject_size = df['SubjectName'].value_counts()
    construct_size = df['ConstructName'].value_counts()
    subject_mean_misconceptions = df.groupby('SubjectName')['misconception_count'].mean()
    construct_mean_misconceptions = df.groupby('ConstructName')['misconception_count'].mean()

    logger.info(
        f"Correlation with Subject: {subject_mean_misconceptions.corr(subject_size):.2f}"
    )
    logger.info(
        f"Correlation with Construct: {construct_mean_misconceptions.corr(construct_size):.2f}"
    )


def generate_test_statistics(df: pd.DataFrame):
    logger.info("Generating statistics for Test Data")
    logger.info(f"Number of unique Questions: {df['QuestionId'].nunique()}")
    logger.info(f"Number of unique Constructs: {df['ConstructId'].nunique()}")
    logger.info(f"Number of unique Subjects: {df['SubjectId'].nunique()}")
    logger.info("Distribution of Correct Answers:")
    logger.info(df['CorrectAnswer'].value_counts().to_dict())


def generate_misconception_mapping_statistics(df: pd.DataFrame):
    logger.info("Generating statistics for Misconception Mapping")
    logger.info(f"Total Misconceptions: {df['MisconceptionId'].nunique()}")
    misconception_names = df['MisconceptionName'].tolist()
    truncated_names = [name[:50] + '...' if len(name) > 50 else name for name in misconception_names[:10]]
    logger.info(f"First 10 Misconception Names (truncated): {truncated_names}")
    if len(misconception_names) > 10:
        logger.info(f"... and {len(misconception_names) - 10} more misconceptions")


def generate_sample_submission_statistics(df: pd.DataFrame):
    logger.info("Generating statistics for Sample Submission")
    logger.info("Sample Submission Records:")
    logger.info(df.head().to_dict(orient='records'))
    logger.info(f"Total Records: {len(df)}")


def main(visualize=False):
    train_filepath = mmim_data_path / "train.csv"
    test_filepath = mmim_data_path / "test.csv"
    mapping_filepath = mmim_data_path / "misconception_mapping.csv"
    submission_filepath = mmim_data_path / "sample_submission.csv"

    # Load and validate data
    train_data = load_train_data(str(train_filepath))
    test_data = load_test_data(str(test_filepath))
    misconception_mapping = load_misconception_mapping(str(mapping_filepath))
    sample_submission = load_sample_submission(str(submission_filepath))

    # Convert models to DataFrames for statistics
    train_df = convert_models_to_dataframe(train_data)
    test_df = convert_models_to_dataframe(test_data)
    mapping_df = convert_models_to_dataframe(misconception_mapping)
    submission_df = convert_models_to_dataframe(sample_submission)

    # Generate and log statistics
    generate_train_statistics(train_df)
    generate_test_statistics(test_df)
    generate_misconception_mapping_statistics(mapping_df)
    generate_sample_submission_statistics(submission_df)

    # Log the number of records loaded
    logger.info(f"Loaded {len(train_data)} training records")
    logger.info(f"Loaded {len(test_data)} test records")
    logger.info(f"Loaded {len(misconception_mapping)} misconception mappings")
    logger.info(f"Loaded {len(sample_submission)} sample submission records")

    # Create visualizations if the flag is set
    if visualize:
        create_visualizations(train_df, mapping_df)

    # Initialize MMIMTask inputs
    mmim_inputs = []
    for train_item in train_data:
        mmim_input = MMIMTask.Input.from_question(train_item)
        mmim_inputs.append(mmim_input)

    logger.info(f"Created {len(mmim_inputs)} MMIMTask inputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMIM task processing")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    args = parser.parse_args()

    main(visualize=args.visualize)
