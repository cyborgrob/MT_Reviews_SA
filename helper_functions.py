import pandas as pd
from sklearn.model_selection import train_test_split

def remove_blank_terms(file_path):
    """Func to remove entries with no aspects after running annotation tool."""
    # Load CSV file into a df
    df = pd.read_csv(file_path, encoding="unicode_escape")

    # Filter the df to exclude rows where 'aspectTerms' is '[]'
    filtered_df = df[df['aspectTerms'] != '[]']

    # Save the filtered df to a new csv file
    filtered_df.to_csv('filtered_file.csv', index=False)


def train_val_test_split(file_path, train=0.7, random_state=42):
    """Takes a file path of a complete dataset and creates shuffled train, validation, and test splits in a 70/15/15
    split by default."""
    # Load dataset
    data = pd.read_csv(file_path)

    # Split into training set and remaining data
    train_data, remaining_data = train_test_split(data, test_size=(1 - train), random_state=random_state)

    # Now split the remaining data into validation and test sets
    validation_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=random_state)

    # Save these to csv files
    train_data.to_csv('train_data.csv', index=False)
    validation_data.to_csv('validation_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
