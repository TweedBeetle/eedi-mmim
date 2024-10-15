import pandas as pd
import numpy as np
from typing import List
from loguru import logger
from tqdm import tqdm
from models import TrainModel, TestModel, MisconceptionMappingModel, SampleSubmissionModel


# Configure loguru
# logger.add("data_loading.log", rotation="1 MB")


def load_train_data(filepath: str) -> List[TrainModel]:
    logger.info(f"Loading training data from {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.exception("Failed to read train.csv")
        raise

    # Replace NaN with None
    df = df.replace({np.nan: None})

    train_data = []
    errors = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Validating Train Data"):
        try:
            model = TrainModel(**row.to_dict())
            train_data.append(model)
        except Exception as e:
            errors.append((index, str(e)))
            logger.error(f"Validation error at row {index}: {e}")

    if errors:
        logger.warning(f"Encountered {len(errors)} errors while loading train data")
        for index, error in errors[:10]:  # Log first 10 errors
            logger.warning(f"Row {index}: {error}")
        if len(errors) > 10:
            logger.warning(f"... and {len(errors) - 10} more errors")
    
    logger.info(f"Successfully loaded and validated {len(train_data)} training data records")
    return train_data


def load_test_data(filepath: str) -> List[TestModel]:
    logger.info(f"Loading test data from {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.exception("Failed to read test.csv")
        raise

    test_data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Validating Test Data"):
        try:
            model = TestModel(**row.to_dict())
            test_data.append(model)
        except Exception as e:
            logger.error(f"Validation error at row {index}: {e}")
            raise
    logger.info("Successfully loaded and validated test data")
    return test_data


def load_misconception_mapping(filepath: str) -> List[MisconceptionMappingModel]:
    logger.info(f"Loading misconception mapping from {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.exception("Failed to read misconception_mapping.csv")
        raise

    mapping_data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Validating Misconception Mapping"):
        try:
            model = MisconceptionMappingModel(**row.to_dict())
            mapping_data.append(model)
        except Exception as e:
            logger.error(f"Validation error at row {index}: {e}")
            raise
    logger.info("Successfully loaded and validated misconception mapping data")
    return mapping_data


def load_sample_submission(filepath: str) -> List[SampleSubmissionModel]:
    logger.info(f"Loading sample submission data from {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.exception("Failed to read sample_submission.csv")
        raise

    submission_data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Validating Sample Submission Data"):
        try:
            model = SampleSubmissionModel(**row.to_dict())
            submission_data.append(model)
        except Exception as e:
            logger.error(f"Validation error at row {index}: {e}")
            raise
    logger.info("Successfully loaded and validated sample submission data")
    return submission_data


# 2024-09-27 16:18:28.197 | INFO     | src.sidequests.eedi_mmim.src.data_loader:load_sample_submission:101 - Successfully loaded and validated sample submission data
# 2024-09-27 16:18:28.252 | INFO     | __main__:generate_train_statistics:22 - Generating statistics for Training Data
# 2024-09-27 16:18:28.253 | INFO     | __main__:generate_train_statistics:23 - Number of unique Questions: 1869
# 2024-09-27 16:18:28.253 | INFO     | __main__:generate_train_statistics:24 - Number of unique Constructs: 757
# 2024-09-27 16:18:28.253 | INFO     | __main__:generate_train_statistics:25 - Number of unique Subjects: 163
# 2024-09-27 16:18:28.253 | INFO     | __main__:generate_train_statistics:26 - Distribution of Correct Answers:
# 2024-09-27 16:18:28.253 | INFO     | __main__:generate_train_statistics:27 - {<CorrectAnswerEnum.C: 'C'>: 488, <CorrectAnswerEnum.A: 'A'>: 482, <CorrectAnswerEnum.B: 'B'>: 461, <CorrectAnswerEnum.D: 'D'>: 438}
# 2024-09-27 16:18:28.253 | INFO     | __main__:generate_train_statistics:28 - Number of Misconceptions per Answer Option:
# 2024-09-27 16:18:28.254 | INFO     | __main__:generate_train_statistics:31 - MisconceptionAId: 1135
# 2024-09-27 16:18:28.254 | INFO     | __main__:generate_train_statistics:31 - MisconceptionBId: 1118
# 2024-09-27 16:18:28.254 | INFO     | __main__:generate_train_statistics:31 - MisconceptionCId: 1080
# 2024-09-27 16:18:28.254 | INFO     | __main__:generate_train_statistics:31 - MisconceptionDId: 1037
# 2024-09-27 16:18:28.254 | INFO     | __main__:generate_train_statistics:33 - Top 5 most common Constructs:
# 2024-09-27 16:18:28.254 | INFO     | __main__:generate_train_statistics:34 - {'Calculate the square of a number': 14, 'Solve two-step linear equations, with the variable on one side, with all positive integers': 13, 'Factorise a quadratic expression in the form x² + bx + c': 13, 'Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division': 12, 'Identify the order of rotational symmetry of a shape': 12}
# 2024-09-27 16:18:28.254 | INFO     | __main__:generate_train_statistics:36 - Top 5 most common Subjects:
# 2024-09-27 16:18:28.255 | INFO     | __main__:generate_train_statistics:37 - {'Linear Equations': 53, 'Linear Sequences (nth term)': 44, 'BIDMAS': 37, 'Quadratic Equations': 36, 'Area of Simple Shapes': 36}
# 2024-09-27 16:18:28.255 | INFO     | __main__:generate_train_statistics:39 - Questions with most misconceptions:
# 2024-09-27 16:18:28.256 | INFO     | __main__:generate_train_statistics:44 - Question (ID: 1) with 3 misconceptions: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \)...
# 2024-09-27 16:18:28.256 | INFO     | __main__:generate_train_statistics:44 - Question (ID: 2) with 3 misconceptions: Tom and Katie are discussing the \( 5 \) plants with these heights:
# \( 24 \mathrm{~cm}, 17 \mathrm{~...
# 2024-09-27 16:18:28.257 | INFO     | __main__:generate_train_statistics:44 - Question (ID: 3) with 3 misconceptions: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle w...
# 2024-09-27 16:18:28.257 | INFO     | __main__:generate_train_statistics:44 - Question (ID: 5) with 3 misconceptions: James has answered a question on the area of a trapezium and got an answer of \( 54 \).
#
# Behind the ...
# 2024-09-27 16:18:28.257 | INFO     | __main__:generate_train_statistics:44 - Question (ID: 6) with 3 misconceptions: Convert this percentage to a fraction
# \( 62 \% \)...
# 2024-09-27 16:18:28.257 | INFO     | __main__:generate_train_statistics:46 - Average number of misconceptions per question:
# 2024-09-27 16:18:28.257 | INFO     | __main__:generate_train_statistics:47 - 2.34
# 2024-09-27 16:18:28.257 | INFO     | __main__:generate_train_statistics:49 - Distribution of number of misconceptions per question:
# 2024-09-27 16:18:28.257 | INFO     | __main__:generate_train_statistics:50 - {1: 307, 2: 623, 3: 939}
# 2024-09-27 16:18:28.257 | INFO     | __main__:generate_train_statistics:52 - Top 5 Constructs with most misconceptions:
# 2024-09-27 16:18:28.260 | INFO     | __main__:generate_train_statistics:54 - {}
# 2024-09-27 16:18:28.260 | INFO     | __main__:generate_train_statistics:56 - Top 5 Subjects with most misconceptions:
# 2024-09-27 16:18:28.261 | INFO     | __main__:generate_train_statistics:58 - {}
# 2024-09-27 16:18:28.261 | INFO     | __main__:generate_train_statistics:60 - Correlation between number of misconceptions and subject/construct:
# 2024-09-27 16:18:28.264 | INFO     | __main__:generate_train_statistics:67 - Correlation with Subject: -0.04
# 2024-09-27 16:18:28.264 | INFO     | __main__:generate_train_statistics:70 - Correlation with Construct: 0.00
# 2024-09-27 16:18:28.264 | INFO     | __main__:generate_test_statistics:76 - Generating statistics for Test Data
# 2024-09-27 16:18:28.264 | INFO     | __main__:generate_test_statistics:77 - Number of unique Questions: 3
# 2024-09-27 16:18:28.264 | INFO     | __main__:generate_test_statistics:78 - Number of unique Constructs: 3
# 2024-09-27 16:18:28.265 | INFO     | __main__:generate_test_statistics:79 - Number of unique Subjects: 3
# 2024-09-27 16:18:28.265 | INFO     | __main__:generate_test_statistics:80 - Distribution of Correct Answers:
# 2024-09-27 16:18:28.265 | INFO     | __main__:generate_test_statistics:81 - {<CorrectAnswerEnum.A: 'A'>: 1, <CorrectAnswerEnum.D: 'D'>: 1, <CorrectAnswerEnum.B: 'B'>: 1}
# 2024-09-27 16:18:28.265 | INFO     | __main__:generate_misconception_mapping_statistics:85 - Generating statistics for Misconception Mapping
# 2024-09-27 16:18:28.265 | INFO     | __main__:generate_misconception_mapping_statistics:86 - Total Misconceptions: 2587
# 2024-09-27 16:18:28.265 | INFO     | __main__:generate_misconception_mapping_statistics:89 - First 10 Misconception Names (truncated): ['Does not know that angles in a triangle sum to 180...', 'Uses dividing fractions method for multiplying fra...', 'Believes there are 100 degrees in a full turn', 'Thinks a quadratic without a non variable term, ca...', 'Believes addition of terms and powers of terms are...', 'When measuring a reflex angle, gives the acute or ...', 'Can identify the multiplier used to form an equiva...', 'Believes gradient = change in y', 'Student thinks that any two angles along a straigh...', 'Thinks there are 180 degrees in a full turn']
# 2024-09-27 16:18:28.265 | INFO     | __main__:generate_misconception_mapping_statistics:91 - ... and 2577 more misconceptions
# 2024-09-27 16:18:28.265 | INFO     | __main__:generate_sample_submission_statistics:95 - Generating statistics for Sample Submission
# 2024-09-27 16:18:28.265 | INFO     | __main__:generate_sample_submission_statistics:96 - Sample Submission Records:
# 2024-09-27 16:18:28.266 | INFO     | __main__:generate_sample_submission_statistics:97 - [{'QuestionId_Answer': '1869_A', 'MisconceptionId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]}, {'QuestionId_Answer': '1869_B', 'MisconceptionId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]}, {'QuestionId_Answer': '1869_C', 'MisconceptionId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]}, {'QuestionId_Answer': '1870_B', 'MisconceptionId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]}, {'QuestionId_Answer': '1870_C', 'MisconceptionId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]}]
# 2024-09-27 16:18:28.266 | INFO     | __main__:generate_sample_submission_statistics:98 - Total Records: 9
# 2024-09-27 16:18:28.266 | INFO     | __main__:main:126 - Loaded 1869 training records
# 2024-09-27 16:18:28.266 | INFO     | __main__:main:127 - Loaded 3 test records
# 2024-09-27 16:18:28.266 | INFO     | __main__:main:128 - Loaded 2587 misconception mappings
# 2024-09-27 16:18:28.266 | INFO     | __main__:main:129 - Loaded 9 sample submission records