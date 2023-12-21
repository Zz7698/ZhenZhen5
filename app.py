import json
import math

import re
import time
from KNN import knn_cluster_reviews
from azure.cosmos import CosmosClient, PartitionKey
from flask import Flask, render_template, request, jsonify
import redis

# password is the "Primary" copied in "Access keys"
from numpy import exceptions

redis_passwd = "5gh73csgVzkZM8LgrUrjxQCMUZ8DgTenfAzCaHhTfsM="
# "Host name" in properties
redis_host = "zhenzhen5.redis.cache.windows.net"
# SSL Port
redis_port = 6380

cache = redis.StrictRedis(
    host=redis_host, port=redis_port,
    db=0, password=redis_passwd,
    ssl=True,
)

if cache.ping():
    print("pong")


# Update the connection string to your Cosmos DB
DB_CONN_STR = "AccountEndpoint=https://tutorial-uta-cse6332.documents.azure.com:443/;AccountKey=fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw=="
cosmos_client = CosmosClient.from_connection_string(DB_CONN_STR)
database = cosmos_client.get_database_client("tutorial")

app = Flask(__name__)

us_cities_table = database.get_container_client("us_cities")
amazon_reviews_table = database.get_container_client("reviews")


@app.route('/')
def index():
    return render_template('index.html')


def calculate_time_of_computing():
    # Implement your time calculation logic here
    # For simplicity, you can use the time module
    return int(time.time())


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Read stopwords from file
    with open('static/stopwords.txt', 'r') as file:
        stop_words = set(file.read().split())

    # Remove stopwords
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Join the filtered tokens back to form the preprocessed text
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


# Function to calculate Haversine distance
def haversine_distance(lat1, lng1, lat2, lng2):
    R = 6371  # Earth radius in kilometers

    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c

    return d


def get_cache_key(city):
    return f"closest_cities:{city}"


@app.route('/data/closest_cities', methods=['GET'])
def closest_cities():
    start_time = time.time()

    # Retrieve parameters from the request
    city = request.args.get('city')
    page_size = int(request.args.get('page_size', 50))
    page = int(request.args.get('page', 0))

    # Check if the result is already in the cache
    cache_key = get_cache_key(city)
    cached_result = cache.get(cache_key)
    # If cache hit, include cache_hit in the result
    if cached_result:
        # Deserialize the JSON-formatted string from the cache
        result = json.loads(cached_result)
        start = page * page_size
        end = start + page_size

        response_cities = result[start + 1:end+1]
        total_time = (time.time() - start_time) * 1000
        return jsonify({
            'cities': response_cities,
            'total_time': total_time,
            'cache_hit': True,
        })

    # If not in the cache, perform the query
    else:
        QUERY1 = "SELECT * FROM c WHERE c.city != '{city}'"
        query = list(us_cities_table.query_items(
            query=QUERY1,
            enable_cross_partition_query=True
        ))

        QUERY2 = f"SELECT * FROM c WHERE c.city = '{city}'"
        city_coordinates = list(us_cities_table.query_items(
            query=QUERY2,
            enable_cross_partition_query=True
        ))
        city_coordinates_lat = float(city_coordinates[0]['lat'])
        city_coordinates_lng = float(city_coordinates[0]['lng'])

        cities = [item for item in query]
        filtered_cities = sorted(cities, key=lambda x: haversine_distance(float(x['lat']), float(x['lng']), city_coordinates_lat, city_coordinates_lng))

        # Pagination
        start = page * page_size
        end = start + page_size
        response_cities = filtered_cities[start+1:end+1]

        total_time = (time.time() - start_time) * 1000

        # Cache the result for future requests
        cache.set(cache_key, json.dumps(filtered_cities))

        return jsonify({
            'cities': response_cities,
            'total_time': total_time,
            'cache_hit': False,
        })


@app.route('/data/knn_reviews', methods=['GET'])
def knn_cluster_reviews():
    try:
        classes = int(request.args.get('classes'))
        k = int(request.args.get('k'))
        words = int(request.args.get('words'))

        start_time = time.time()

        # Get clustered data and computation time
        clustered_data, computation_time = knn_cluster_reviews(cosmos_client, "tutorial", "us_cities", "reviews", classes, k, words)

        end_time = time.time()
        response_time = int((end_time - start_time) * 1000)

        response_data = {
            "clustered_data": clustered_data,
            "computation_time": response_time
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/flush_cache', methods=['GET'])
def flush_cache():
    # Implement logic to flush the entire cache
    cache.flushall()
    return jsonify({"message": "Cache flushed successfully"})


if __name__ == '__main__':
    app.run()
