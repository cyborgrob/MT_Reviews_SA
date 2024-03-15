import requests

# Define your API credentials
API_KEY = 'API_KEY'

# Define the Yelp Business ID of the business you want to retrieve reviews for
business_id = 'madtree-brewing-cincinnati-3'

# Define the API endpoint
url = f'https://api.yelp.com/v3/businesses/{business_id}/reviews'

# Define the request headers with your API Key
headers = {
    'Authorization': f'Bearer {API_KEY}'
}

# Make the API request
response = requests.get(url, headers=headers)
data = response.json()

# Extract reviews from the response
reviews = data['reviews']

# Process the reviews
for review in reviews:
    review_text = review['text']
    # Process the review text as needed (e.g., sentiment analysis)
    print(review_text)
