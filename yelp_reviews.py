import requests

# Define your API credentials
API_KEY = 'ie0Pg2QoJOxmvFhQjZU_qVq90OfahbgVapcJuWyvVl4v1WlV9ec0jnq2WLeuKb8lCEiXFa4OXtzwvw6FssQQ3g4zw8YvkKv_GgHqBG4_eHVXZh55rfdnbj6I66r0ZXYx'

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
