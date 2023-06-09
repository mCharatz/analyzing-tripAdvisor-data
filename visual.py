import matplotlib.pyplot as plt
from datetime import datetime
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import FreqDist, word_tokenize, bigrams, trigrams
from wordcloud import WordCloud
import random

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

plt.plot(sorted_months, review_counts)
plt.xlabel("Month")
plt.ylabel("Number of Reviews")
plt.title("Number of Monthly Reviews")
plt.xticks(rotation=45)
plt.show()

# Fetch reviews from MongoDB and convert to a list
reviews = list(collection.find())

# 2. Identify the top-10 rated and bottom-10 rated locations
locations = {}  # Dictionary to store location ratings and counts

for review in reviews:
    location = review.get("business_reviewed")  # Use "business_reviewed" instead of "business_name"
    rating = review.get("review_rating")
    if location and rating:
        if location not in locations:
            locations[location] = {"count": 1, "rating": rating}
        else:
            locations[location]["count"] += 1
            locations[location]["rating"] += rating

sorted_locations = sorted(locations.items(), key=lambda x: (x[1]["rating"] / x[1]["count"]), reverse=True)
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
plt.figure(figsize=(10, 6))
plt.barh(np.arange(len(top_locations)), top_ratings, color='green')
plt.yticks(np.arange(len(top_locations)), top_locations)
plt.xlabel('Average Rating')
plt.ylabel('Locations')
plt.title('Top 10 Rated Locations')
plt.tight_layout()
plt.show()

# Visualize the bottom-10 rated locations
plt.figure(figsize=(10, 6))
plt.barh(np.arange(len(bottom_locations)), bottom_ratings, color='red')
plt.yticks(np.arange(len(bottom_locations)), bottom_locations)
plt.xlabel('Average Rating')
plt.ylabel('Locations')
plt.title('Bottom 10 Rated Locations')
plt.tight_layout()
plt.show()


# 2nd visualisation
# Set up the style and color palette
sns.set_style("whitegrid")
colors = sns.color_palette("coolwarm", len(top_locations))

# Visualize the top-10 rated locations
plt.figure(figsize=(10, 6))
plt.barh(np.arange(len(top_locations)), top_ratings, color=colors)
plt.yticks(np.arange(len(top_locations)), top_locations)
plt.xlabel('Average Rating')
plt.ylabel('Locations')
plt.title('Top 10 Rated Locations')

# Add data labels to the bars
for i, rating in enumerate(top_ratings):
    plt.text(rating + 0.1, i, f'{rating:.2f}', va='center', color='black')

# Add a color legend
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
plt.legend(legend_handles, top_locations, loc='lower right', bbox_to_anchor=(1.1, 0.5))

plt.tight_layout()
plt.show()

# Visualize the bottom-10 rated locations
plt.figure(figsize=(10, 6))
plt.barh(np.arange(len(bottom_locations)), bottom_ratings, color=colors[::-1])
plt.yticks(np.arange(len(bottom_locations)), bottom_locations)
plt.xlabel('Average Rating')
plt.ylabel('Locations')
plt.title('Bottom 10 Rated Locations')

# Add data labels to the bars
for i, rating in enumerate(bottom_ratings):
    plt.text(rating + 0.1, i, f'{rating:.2f}', va='center', color='black')

# Add a color legend
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors[::-1]]
plt.legend(legend_handles, bottom_locations, loc='lower right', bbox_to_anchor=(1.1, 0.5))

plt.tight_layout()
plt.show()




# 3
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

# 6. Explore and visualize emerging topics from user reviews across time (using techniques like topic modeling or clustering)
# Code to perform topic modeling or clustering on the review texts goes here
# Additional analysis steps can be added as per your requirements