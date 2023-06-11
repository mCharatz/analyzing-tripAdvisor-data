import matplotlib.pyplot as plt
from datetime import datetime
import nltk
import pyLDAvis as pyLDAvis
from gensim import corpora
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import FreqDist, word_tokenize, bigrams, trigrams
from wordcloud import WordCloud
import random
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as greek_stopwords
import gensim.corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import matplotlib.pyplot as plt
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["reviews"]
collection = db["reviews"]

# Fetch reviews from MongoDB
reviews = collection.find()

# Preprocessing steps
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Process review data
review_dates = []
review_ratings = []
review_texts = []

for review in reviews:
    if "review_date" in review:
        review_dates.append(review["review_date"])
    if "review_rating" in review:
        review_ratings.append(review["review_rating"])
    if "review_text" in review:
        review_texts.append(review["review_text"])


# Basic visualizations

# 1. Visualize the number of monthly reviews over time
monthly_reviews = {}

for date in review_dates:
    month_year = date.strftime("%Y-%m")
    monthly_reviews[month_year] = monthly_reviews.get(month_year, 0) + 1

sorted_months = sorted(monthly_reviews.keys())
review_counts = [monthly_reviews[month] for month in sorted_months]

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.lineplot(x=sorted_months, y=review_counts, marker='o', color='steelblue')
plt.xlabel("Month")
plt.ylabel("Number of Reviews")
plt.title("Number of Monthly Reviews")
plt.xticks(rotation=45)

# Specify the x-axis tick interval
interval = max(1, len(sorted_months) // 25)  # Adjust the interval as needed
plt.xticks(np.arange(0, len(sorted_months), interval), [sorted_months[i] for i in range(0, len(sorted_months), interval)])

plt.tight_layout()
plt.show()

# Fetch reviews from MongoDB and convert to a list
reviews = list(collection.find())

# 2. Identify the top-10 rated and bottom-10 rated locations
import matplotlib.pyplot as plt

locations = {}  # Dictionary to store location ratings and counts

for review in reviews:
    location = review.get("business_reviewed")
    rating = review.get("review_rating")
    if location and rating:
        if location not in locations:
            locations[location] = {"count": 1, "rating": rating}
        else:
            locations[location]["count"] += 1
            locations[location]["rating"] += rating

# Filter locations based on minimum number of ratings
min_ratings_threshold = 50
filtered_locations = {
    location: data for location, data in locations.items() if data["count"] >= min_ratings_threshold
}

sorted_locations = sorted(filtered_locations.items(), key=lambda x: (x[1]["count"], x[1]["rating"]), reverse=True)
top_10_rated = sorted_locations[:10]
bottom_10_rated = sorted_locations[-10:]

# Extract location names, average ratings, and counts for visualization
top_locations = [location for location, _ in top_10_rated]
top_ratings = [(data["rating"] / data["count"]) for _, data in top_10_rated]
top_counts = [data["count"] for _, data in top_10_rated]

bottom_locations = [location for location, _ in bottom_10_rated]
bottom_ratings = [(data["rating"] / data["count"]) for _, data in bottom_10_rated]
bottom_counts = [data["count"] for _, data in bottom_10_rated]

print("Top 10 Rated Locations:")
for location, data in top_10_rated:
    print(location, "- Average Rating:", data["rating"] / data["count"], "- Number of Ratings:", data["count"])
print("\nBottom 10 Rated Locations:")
for location, data in bottom_10_rated:
    print(location, "- Average Rating:", data["rating"] / data["count"], "- Number of Ratings:", data["count"])

# Visualize the top-10 rated locations
plt.figure(figsize=(12, 8))
sns.barplot(x=top_ratings, y=top_locations, palette='coolwarm_r')
plt.xlabel('Average Rating')
plt.ylabel('Locations')
plt.title('Top 10 Rated Locations')

# Add data labels to the bars
for i, rating in enumerate(top_ratings):
    plt.text(rating + 0.1, i, f'{rating:.2f}', va='center', color='black')

plt.tight_layout()
plt.show()

# Visualize the bottom-10 rated locations
plt.figure(figsize=(12, 8))
sns.barplot(x=bottom_ratings, y=bottom_locations, palette='coolwarm')
plt.xlabel('Average Rating')
plt.ylabel('Locations')
plt.title('Bottom 10 Rated Locations')

# Add data labels to the bars
for i, rating in enumerate(bottom_ratings):
    plt.text(rating + 0.1, i, f'{rating:.2f}', va='center', color='black')

plt.tight_layout()
plt.show()




# 3. Which locations have the highest increase or decrease in rating over the years?
# Consider calculating the yearly average and percentage change per location.
import matplotlib.pyplot as plt

yearly_ratings = {}

for review in reviews:
    location_id = review.get("business_reviewed")
    rating = review.get("review_rating")
    review_date = review.get("review_date")

    if rating is None or review_date is None:
        continue

    year = review_date.year

    if location_id not in yearly_ratings:
        yearly_ratings[location_id] = [(year, rating)]
    else:
        yearly_ratings[location_id].append((year, rating))

# Calculate yearly average and percentage change for each location
location_changes = {}
for location_id, ratings in yearly_ratings.items():
    sorted_ratings = sorted(ratings, key=lambda x: x[0])
    yearly_avg_ratings = [rating for year, rating in sorted_ratings]
    percentage_changes = [
        (yearly_avg_ratings[i] - yearly_avg_ratings[i-1]) / yearly_avg_ratings[i-1] * 100
        for i in range(1, len(yearly_avg_ratings))
    ]
    location_changes[location_id] = (sorted_ratings, yearly_avg_ratings, percentage_changes)

# Sort locations based on the total percentage change
sorted_locations = sorted(location_changes.items(), key=lambda x: sum(x[1][2]))

# Select locations with the highest increase and decrease in rating
top_increase_locations = [loc for loc in sorted_locations if sum(loc[1][2]) > 0][:10]
top_decrease_locations = [loc for loc in sorted_locations if sum(loc[1][2]) < 0][:10]

# Print the locations with the highest increase and decrease in rating
print("Locations with the Highest Increase in Rating:")
for location_id, changes in top_increase_locations:
    location_name = location_id  # Replace with the code to retrieve location name based on ID
    sorted_ratings, yearly_avg_ratings, percentage_changes = changes
    print(f"Location: {location_name}")
    print("Yearly Average Ratings:", yearly_avg_ratings)
    print("Percentage Changes:", percentage_changes)
    print()

print("\nLocations with the Highest Decrease in Rating:")
for location_id, changes in top_decrease_locations:
    location_name = location_id  # Replace with the code to retrieve location name based on ID
    sorted_ratings, yearly_avg_ratings, percentage_changes = changes
    print(f"Location: {location_name}")
    print("Yearly Average Ratings:", yearly_avg_ratings)
    print("Percentage Changes:", percentage_changes)
    print()

# Plot locations with the highest increase in rating
plt.figure(figsize=(12, 6))
for location_id, changes in top_increase_locations:
    sorted_ratings, yearly_avg_ratings, _ = changes
    years = [year for year, _ in sorted_ratings]
    location_name = location_id  # Replace with the code to retrieve location name based on ID

    plt.plot(years, yearly_avg_ratings, marker='o', label=location_name)

plt.xlabel("Year")
plt.ylabel("Yearly Average Rating")
plt.title("Locations with the Highest Increase in Rating")
plt.legend()
plt.grid(True)
plt.show()

# Plot locations with the highest decrease in rating
plt.figure(figsize=(12, 6))
for location_id, changes in top_decrease_locations:
    sorted_ratings, yearly_avg_ratings, _ = changes
    years = [year for year, _ in sorted_ratings]
    location_name = location_id  # Replace with the code to retrieve location name based on ID

    plt.plot(years, yearly_avg_ratings, marker='o', label=location_name)

plt.xlabel("Year")
plt.ylabel("Yearly Average Rating")
plt.title("Locations with the Highest Decrease in Rating")
plt.legend()
plt.grid(True)
plt.show()





# 4. Visualize the most common words, bi-grams, and tri-grams across all reviews
# through a bar chart or word cloud. Also, visualize the most common words,
# bi-grams, and tri-grams in 5-star versus 1-star reviews.

# Set a random seed for consistent colors in word clouds
random.seed(42)

# Prepare the data
review_texts = [review.get("review_text", "") for review in reviews]

# Tokenize and filter the words
all_words = []
five_star_words = []
one_star_words = []
stop_words = set(stopwords.words("english"))

for text in review_texts:
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    all_words.extend(words)

# Calculate frequency distributions
freq_dist_all = FreqDist(all_words)

# Bar chart of the most common words across all reviews
most_common_words_all = freq_dist_all.most_common(10)
x, y = zip(*most_common_words_all)
x = [word for word in x]

plt.figure(figsize=(12, 6))
plt.barh(x, y, color="teal")  # Horizontal bar chart
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Words", fontsize=12)
plt.title("Most Common Words across All Reviews", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Word cloud of the most common words across all reviews
wordcloud_all = WordCloud(width=800, height=600, background_color="white",
                          colormap="Blues", random_state=42).generate_from_frequencies(freq_dist_all)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_all, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Most Common Words across All Reviews", fontsize=14)
plt.show()

# Calculate frequency distributions for 5-star and 1-star reviews
for review in reviews:
    rating = review.get("review_rating", 0)
    text = review.get("review_text", "")

    if rating == 5:
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in stop_words]
        five_star_words.extend(words)
    elif rating == 1:
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in stop_words]
        one_star_words.extend(words)

# Calculate frequency distributions for 5-star and 1-star reviews
freq_dist_five_star = FreqDist(five_star_words)
freq_dist_one_star = FreqDist(one_star_words)

# Bar chart of the most common words in 5-star reviews
most_common_words_five_star = freq_dist_five_star.most_common(10)
x, y = zip(*most_common_words_five_star)

plt.figure(figsize=(12, 6))
plt.bar(x, y, color="purple")
plt.xlabel("Words", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Most Common Words in 5-star Reviews", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Word cloud of the most common words in 5-star reviews
wordcloud_five_star = WordCloud(width=800, height=600, background_color="white",
colormap="Oranges", random_state=42).generate_from_frequencies(freq_dist_five_star)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_five_star, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Most Common Words in 5-star Reviews", fontsize=14)
plt.show()

# Bar chart of the most common words in 1-star reviews
most_common_words_one_star = freq_dist_one_star.most_common(10)
x, y = zip(*most_common_words_one_star)

plt.figure(figsize=(12, 6))
plt.bar(x, y, color="red")
plt.xlabel("Words", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Most Common Words in 1-star Reviews", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Word cloud of the most common words in 1-star reviews
wordcloud_one_star = WordCloud(width=800, height=600, background_color="white",
colormap="Greens", random_state=42).generate_from_frequencies(freq_dist_one_star)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_one_star, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Most Common Words in 1-star Reviews", fontsize=14)
plt.show()

# Calculate frequency distributions for bi-grams and tri-grams across all reviews
all_bigrams = list(bigrams(all_words))
all_trigrams = list(trigrams(all_words))

freq_dist_bigrams = FreqDist(all_bigrams)
freq_dist_trigrams = FreqDist(all_trigrams)

# Bar chart of the most common bi-grams across all reviews
most_common_bigrams = freq_dist_bigrams.most_common(10)
x, y = zip(*most_common_bigrams)

x = [' '.join(words) for words, _ in x] # Join the words in each bi-gram

plt.figure(figsize=(12, 6))
plt.bar(x, y, color="magenta")
plt.xlabel("Bi-grams", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Most Common Bi-grams across All Reviews", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()



# 5. Identify the 10 fastest growing and 10 fastest shrinking words over time (based on usage frequency)
# Code to calculate the frequency change of words over time goes here
from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize
import string
import matplotlib.pyplot as plt

# Fetch reviews from MongoDB
reviews = collection.find()

# Extract the review text and review date from each document
review_texts = []
review_dates = []
for review in reviews:
    if "review_text" in review:
        review_texts.append(review["review_text"])
    if "review_date" in review:
        review_dates.append(review["review_date"])

# Combine review text and review date into a list of tuples
review_data = list(zip(review_texts, review_dates))

# Sort the review data based on review date
review_data.sort(key=lambda x: x[1])

# Initialize variables
word_frequency = {}
total_words = 0

# Iterate over the review data
for review_text, review_date in review_data:
    # Tokenize the text into words
    words = word_tokenize(review_text)

    # Remove stopwords and punctuation marks
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and word not in string.punctuation]

    # Update word frequencies
    for word in words:
        if word not in word_frequency:
            word_frequency[word] = 0
        word_frequency[word] += 1
        total_words += 1

# Calculate the frequency change of each word
freq_change = {}
for word, frequency in word_frequency.items():
    freq_change[word] = frequency / total_words

# Sort the words based on frequency change
sorted_words = sorted(freq_change.items(), key=lambda x: x[1], reverse=True)

# Get the top 10 growing words
top_growing_words = sorted_words[:10]

# Get the top 10 shrinking words
top_shrinking_words = sorted_words[-10:]

# Print the results
print("Top 10 Growing Words:")
for word, change in top_growing_words:
    print(f"Word: {word}, Frequency: {change}")

print("\nTop 10 Shrinking Words:")
for word, change in top_shrinking_words:
    print(f"Word: {word}, Frequency: {change}")

# Extract the words and frequency change values
words = [word for word, _ in sorted_words]
freq_changes = [change for _, change in sorted_words]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the frequency change values as a bar plot
ax.bar(words, freq_changes, color='blue')

# Set labels and title
ax.set_xlabel('Words')
ax.set_ylabel('Relative Frequency Change')
ax.set_title('Frequency Change of Words Over Time')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()



# 6. Explore and visualize emerging topics from user reviews across time (using techniques like topic modeling or clustering)

# Download the necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the stop words list for English
english_stop_words = set(stopwords.words('english'))

# Define the stop words list for Greek
greek_stop_words = set(stopwords.words('greek'))

# Combine the English and Greek stop words
stop_words = english_stop_words.union(greek_stop_words)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess the reviews and create a list of tokenized texts
texts = []
review_dates = []  # Store the review dates

for review in reviews:
    if 'review_text' in review:
        text = review['review_text']
        review_date = review['review_date']
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stop words and perform lemmatization
        preprocessed_text = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        # Append the preprocessed text to the list of texts
        texts.append(preprocessed_text)
        review_dates.append(review_date)

# Create a dictionary from the tokenized texts
dictionary = gensim.corpora.Dictionary(texts)

# Create a corpus from the tokenized texts
corpus = [dictionary.doc2bow(text) for text in texts]

# Build the LDA model
lda_model = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=10)

# Visualize the topics using the modified LDAvis visualization
topic_data = gensimvis.prepare(lda_model, corpus, dictionary)

# Save the modified LDAvis HTML file
pyLDAvis.save_html(topic_data, 'lda_visualization.html')

# Print the topic model results
print(lda_model.print_topics())

# Show the plot
plt.show()