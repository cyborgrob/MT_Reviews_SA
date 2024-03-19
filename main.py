import pandas as pd
import os
import re


def convert_outscraper_file(csv_filepath):
    """
    Extracts useful columns from an Outscraper output csv file.
    """
    # Create new df using only select columns
    df = pd.read_csv(csv_filepath)
    desired_columns = ["review_datetime_utc", "review_rating", "review_text"]
    df_selected = df[desired_columns]
    # Output new csv
    output_path = os.environ.get("output_path")
    df_selected.to_csv(output_path, index=False)

