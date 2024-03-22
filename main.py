import pandas as pd
import os
import re
import requests
from bs4 import BeautifulSoup

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


def scrape_yelp(url):
    """
    Scrapes customer reviews (doesn't include business reply reviews) from a given yelp page. Effective as of March '24.
    :param url: URL of yelp page to scrape reviews from
    :return: list of reviews
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*comment__09f24__D0cxf.*') # If you want all reviews including business replies, replace 'comment__09f24__D0cxf' with just 'comment'. This value was found by inspecting review 'class'
    results = soup.find_all('p', {'class': regex})
    reviews = [result.text for result in results]
    return reviews

print(scrape_yelp('https://www.yelp.com/biz/madtree-brewing-cincinnati-3'))

