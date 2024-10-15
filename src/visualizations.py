import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import os
from pathlib import Path

from src.constants import PROJECT_ROOT


def create_visualizations(train_df: pd.DataFrame, mapping_df: pd.DataFrame):
    logger.info("Creating visualizations...")

    # Create output directory if it doesn't exist
    output_dir = PROJECT_ROOT / 'out' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style for all plots
    plt.style.use('ggplot')

    # 1. Distribution of misconceptions per question
    plt.figure(figsize=(10, 6))
    misconceptions = ['misconception_a_id', 'misconception_b_id', 'misconception_c_id', 'misconception_d_id']
    misconception_counts = train_df[misconceptions].notna().sum(axis=1)
    sns.histplot(misconception_counts, kde=True)
    plt.title('Distribution of Misconceptions per Question')
    plt.xlabel('Number of Misconceptions')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'misconceptions_distribution.png')
    plt.close()

    # 2. Top 10 Subjects by number of questions
    plt.figure(figsize=(12, 6))
    train_df['subject_name'].value_counts().nlargest(10).plot(kind='bar')
    plt.title('Top 10 Subjects by Number of Questions')
    plt.xlabel('Subject')
    plt.ylabel('Number of Questions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_subjects.png')
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = train_df[['construct_id', 'subject_id'] + misconceptions].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Misconceptions, Constructs, and Subjects')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png')
    plt.close()

    # 4. Word cloud of misconception names
    from wordcloud import WordCloud
    plt.figure(figsize=(12, 8))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
        ' '.join(mapping_df['misconception_name'])
        )
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Misconception Names')
    plt.tight_layout()
    plt.savefig(output_dir / 'misconception_wordcloud.png')
    plt.close()

    logger.info(f"Visualizations created and saved in {output_dir}")
