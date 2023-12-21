function searchCities() {
    var city = document.getElementById('cityInput').value;
    var pageSize = document.getElementById('pageSizeInput').value;
    var page = document.getElementById('pageInput').value;

    $.get('/data/closest_cities', { city: city, page_size: pageSize, page: page }, function (data) {
        displayResults(data);
    });
}

function flushCache() {
    $.get('/flush_cache', function () {
        alert('Cache flushed successfully!');
    });
}

function displayResults(data) {
    var resultsDiv = document.getElementById('results');
    var timeDiv = document.getElementById('time');

    // Check if cache is involved
    var cacheTag = data.hasOwnProperty('cache_hit') && data.cache_hit ? 'Cache Hit' : 'Cache Miss';

    // Display time
    timeDiv.innerHTML = '<p>Total Time: ' + data.total_time.toFixed(2) + ' ms</p><p>' + cacheTag + '</p>';

    // Display results in a table
    var tableHtml = '<h2>Results:</h2><table border="1"><tr><th>City</th><th>Latitude</th><th>Longitude</th></tr>';
    data.cities.forEach(function(city) {
        tableHtml += '<tr><td>' + city.city + '</td><td>' + city.lat + '</td><td>' + city.lng + '</td></tr>';
    });
    tableHtml += '</table>';

    resultsDiv.innerHTML = tableHtml;
}

function analyzeReviews() {
    var classes = document.getElementById('classesInput').value;
    var k = document.getElementById('kInput').value;
    var words = document.getElementById('wordsInput').value;

    $.get('/data/knn_reviews', { classes: classes, k: k, words: words }, function (data) {
        displayResults(data);
    });
}
$('document').ready(
    function(){
        $('#btn-popular_v1').click(function() {
            var numberOfClasses = $('#classesInput').val();
            var kValue = $('#kInput').val();
            var wordsLimit = $('#wordsInput').val();
            var queryParams = $.param({
                classes: numberOfClasses,
                k: kValue,
                words: wordsLimit
            });
            window.location.href = "/data/knn_reviews?" + queryParams;
        });
    }
);

