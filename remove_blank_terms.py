import pandas as pd

def remove_blank_terms(file_path):
    """Func to remove entries with no aspects after running annotation tool."""
    # Load CSV file into a df
    df = pd.read_csv(file_path, encoding="unicode_escape")

    # Filter the df to exclude rows where 'aspectTerms' is '[]'
    filtered_df = df[df['aspectTerms'] != '[]']

    # Save the filtered df to a new csv file
    filtered_df.to_csv('filtered_file.csv', index=False)
