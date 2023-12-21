from azure.cosmos import CosmosClient
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from collections import Counter

def load_data(cosmos_client, database_name, container_name):
    # Connect to Cosmos DB and retrieve data
    database = cosmos_client.get_database_client(database_name)
    container = database.get_container_client(container_name)

    # Query Cosmos DB to get data
    query = "SELECT c.city, c.lat, c.lng, c.population FROM c"
    data = list(container.query_items(query, enable_cross_partition_query=True))

    return data

def load_reviews_data(cosmos_client, database_name, container_name):
    # Similar to load_data but for reviews
    database = cosmos_client.get_database_client(database_name)
    container = database.get_container_client(container_name)

    # Query Cosmos DB to get reviews data
    query = "SELECT c.score, c.city, c.review FROM c"
    reviews_data = list(container.query_items(query, enable_cross_partition_query=True))

    return reviews_data

def preprocess_reviews(reviews_data):
    # Combine 'title' and 'review' columns into a single text column
    review_text = [f"{review['score']} {review['city']} {review['review']}" for review in reviews_data]
    return review_text

def find_most_popular_words(reviews, num_words):
    # Flatten and tokenize the reviews
    all_words = [word.lower() for review in reviews for word in review.split()]

    # Exclude stopwords (assuming you have a file named "stopwords.txt")
    with open("stopwords.txt", "r") as stopword_file:
        stopwords = stopword_file.read().splitlines()

    filtered_words = [word for word in all_words if word not in stopwords]

    # Find the most common words
    counter = Counter(filtered_words)
    most_common_words = counter.most_common(num_words)

    return most_common_words

def calculate_weighted_average_score(reviews_data, cities_data):
    total_score = 0
    total_weight = 0

    for review in reviews_data:
        city_name = review['city']
        score = review['score']

        # Find the city in cities_data
        city = next((city for city in cities_data if city['city'] == city_name), None)

        if city:
            # Use city population as the weight
            weight = city['population']
            total_score += weight * score
            total_weight += weight

    if total_weight == 0:
        return 0

    return total_score / total_weight

def knn_cluster_reviews(cosmos_client, database_name, container_name_cities, container_name_reviews, classes, k, words):
    # Load data from Cosmos DB
    cities_data = load_data(cosmos_client, database_name, container_name_cities)
    reviews_data = load_reviews_data(cosmos_client, database_name, container_name_reviews)

    # Extract review text for clustering
    review_text = preprocess_reviews(reviews_data)

    # Convert review text into a bag-of-words representation
    vectorizer = CountVectorizer(stop_words='english', max_features=words)
    X = vectorizer.fit_transform(review_text)

    # Normalize the bag-of-words matrix
    X_normalized = normalize(X)

    # Perform KNN clustering
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(X_normalized)
    distances, indices = knn.kneighbors(X_normalized)

    # Placeholder for popular words
    popular_words = {}

    clustered_data = {}
    for cluster_id in range(classes):
        cluster_id_str = f"class_{cluster_id}"
        reviews_in_cluster = [reviews_data[i]['review'] for i in indices[:, cluster_id]]

        # Calculate weighted average score for the cluster
        weighted_average_score = calculate_weighted_average_score(reviews_data, cities_data)

        # Find the most popular words in the cluster
        most_popular_words = find_most_popular_words(reviews_in_cluster, words)
        popular_words[cluster_id_str] = most_popular_words

        clustered_data[cluster_id_str] = {
            "reviews": reviews_in_cluster,
            "weighted_average_score": weighted_average_score
        }

    # Dummy computation time for illustration
    computation_time = 150  # milliseconds

    return clustered_data, popular_words, computation_time
