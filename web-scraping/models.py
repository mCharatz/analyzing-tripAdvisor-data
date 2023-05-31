import datetime
import hashlib
import json
import pymongo
import bs4
from selenium import webdriver


# secondary models
class ReviewType:
    BUSINESS = "business"
    REVIEW = "review"
    REVIEWER = "reviewer"
    URL = "url"


class URLType:
    PAGINATION_URL = "pagination url"
    ACCESS_URL = "access url"


# main models
class Business:
    """
    Business of TripAdvisor
    """

    def __init__(
            self,
            business_name: str = ""
    ):
        self.business_name = business_name

    def to_mongo_document(self) -> dict:
        """
        Serialize the object as a mongo document
        """

        mongo_doc = {
            "unique_id": self.get_object_hash(),
            "type": ReviewType.BUSINESS,
            "business_name": self.business_name
        }
        return mongo_doc

    def get_object_hash(self) -> str:
        """
        Calculate the hash of the business object,
        it will be used as a custom ID in MongoDB to avoid duplicate insertions
        """
        obj_json = json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
        hash_object = hashlib.sha256(obj_json.encode())
        return hash_object.hexdigest()


class Reviewer:
    """
    Reviewer of TripAdvisor
    """

    def __init__(
            self,
            reviewer_username: str = "",
            reviewer_name: str = ""
    ):
        self.reviewer_username = reviewer_username
        self.reviewer_name = reviewer_name

    def to_mongo_document(self) -> dict:
        """
        Serialize the object as a mongo document
        """
        mongo_doc = {
            "unique_id": self.get_object_hash(),
            "type": ReviewType.REVIEWER,
            "reviewer_username": self.reviewer_username,
            "reviewer_name": self.reviewer_username
        }
        return mongo_doc

    def get_object_hash(self) -> str:
        """
        Calculate the hash of the reviewer object,
        it will be used as a custom ID in MongoDB to avoid duplicate insertions
        """
        obj_json = json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
        hash_object = hashlib.sha256(obj_json.encode())
        return hash_object.hexdigest()


class Review:
    """
    Review of a Business of TripAdvisor
    """

    def __init__(
            self,
            business: Business = Business(),
            reviewer: Reviewer = Reviewer(),
            review_date: datetime = None,
            visit_data: datetime = None,
            review_title: str = "",
            review_text: str = "",
            review_rating: float = -1,
    ):
        self.business = business
        self.reviewer = reviewer
        self.review_date = review_date
        self.visit_data = visit_data
        self.review_title = review_title
        self.review_text = review_text
        self.review_rating = review_rating

    def to_mongo_document(self) -> dict:
        """
        Serialize the object as a mongo document
        """

        mongo_doc = {
            "unique_id": self.get_object_hash(),
            "type": ReviewType.REVIEW,
            "business_reviewed": self.business.business_name,
            "reviewer_username": self.reviewer.reviewer_username,
            "review_date": self.review_date,
            "review_title": self.review_title,
            "review_text": self.review_text,
            "review_rating": self.review_rating,
            "visit_data": self.visit_data

        }
        return mongo_doc

    def get_object_hash(self) -> str:
        """
        Calculate the hash of the review object,
        it will be used as a custom ID in MongoDB to avoid duplicate insertions
        """
        obj_json = json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
        hash_object = hashlib.sha256(obj_json.encode())
        return hash_object.hexdigest()


class URL:
    """
    URL of TripAdvisor (pagination url or access url)
    """

    def __init__(
            self,
            url: str,
            url_type: str,
            object_type: ReviewType or str,
            page_of_pagination: int = 0  # 0 = not a pagination url
    ):
        self.url = url
        self.url_type = url_type
        self.object_type = object_type
        self.page_of_pagination = page_of_pagination

    def to_mongo_document(self) -> dict:
        """
        Serialize the object as a mongo document
        """

        mongo_doc = {
            "unique_id": self.get_object_hash(),
            "type": ReviewType.URL,
            "url": self.url,
            "url_type": self.url_type,
            "object_type": self.object_type,
            "page_of_pagination": self.page_of_pagination
        }
        return mongo_doc

    def get_object_hash(self) -> str:
        """
        Calculate the hash of the url object,
        it will be used as a custom ID in MongoDB to avoid duplicate insertions
        """
        obj_json = json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
        hash_object = hashlib.sha256(obj_json.encode())
        return hash_object.hexdigest()


# functionality models
class TripAdvisorMongoClient:
    def __init__(
            self,
            connection_string: str = "",
            database: pymongo.database.Database or str = "",
            urls_collection: pymongo.collection.Collection or str = "",
            reviews_collection: pymongo.collection.Collection or str = "",

    ):
        self.connection_string = connection_string
        self.mongo_client = pymongo.MongoClient(self.connection_string)
        self.database = self.mongo_client[database]
        self.urls_collection = self.database[urls_collection]
        self.reviews_collection = self.database[reviews_collection]

    def insert_business(self, business: Business) -> bool:
        if self.reviews_collection.find_one({"unique_id": business.get_object_hash()}) is not None:
            return False  # document exist
        self.reviews_collection.insert_one(business.to_mongo_document())
        return True

    def insert_review(self, review: Review) -> bool:
        if self.reviews_collection.find_one({"unique_id": review.get_object_hash()}) is not None:
            return False  # document exist
        self.reviews_collection.insert_one(review.to_mongo_document())
        return True

    def insert_reviewer(self, reviewer: Reviewer) -> bool:
        if self.reviews_collection.find_one({"unique_id": reviewer.get_object_hash()}) is not None:
            return False  # document exist
        self.reviews_collection.insert_one(reviewer.to_mongo_document())
        return True

    def insert_url(self, url: URL) -> bool:
        if self.urls_collection.find_one({"unique_id": url.get_object_hash()}) is not None:
            return False  # document exist
        self.urls_collection.insert_one(url.to_mongo_document())
        return True


class TripAdvisorWebScrapper():
    def __init__(
            self,
            web_driver: webdriver = None,
            trip_advisor_mongo_client: TripAdvisorMongoClient = None

    ):
        if web_driver is None:
            web_driver = webdriver.Chrome()
        self.web_driver = web_driver
        self.trip_advisor_mongo_client = trip_advisor_mongo_client
        self.TRIP_ADVISOR_URL = 'https://www.tripadvisor.com'

    def get_and_store_paginated_business_urls(self, start_url: str) -> list[str]:

        next_page_arrow_class = "BrOJk u j z _F wSSLS tIqAi unMkR"

        def find_next_page_arrow(current_page):
            next_or_prev_arrows = current_page.find_all("a", class_=next_page_arrow_class)
            for element in next_or_prev_arrows:
                if element['aria-label'] == "Next page":
                    return element
            return None

        paginated_url_list = []
        current_page_pagination = 1

        # initialize the web driver and store the start URL
        self.web_driver.get(start_url)
        url_object = URL(
            url_type=URLType.PAGINATION_URL,
            object_type=ReviewType.BUSINESS,
            url=start_url,
            page_of_pagination=current_page_pagination
        )
        self.trip_advisor_mongo_client.insert_url(url_object)
        paginated_url_list.append(start_url)

        while True:
            try:
                # parse the current page
                current_page = bs4.BeautifulSoup(self.web_driver.page_source, 'html.parser')

                # find the next page arrow
                next_page_arrow = find_next_page_arrow(current_page)

                if not next_page_arrow:
                    break  # No 'Next page' arrow found, (end of pagination)

                # get the URL for the next page
                new_url = self.TRIP_ADVISOR_URL + next_page_arrow['href']

                # store the new url to mongo
                current_page_pagination += 1
                url_object = URL(
                    url_type=URLType.PAGINATION_URL,
                    object_type=ReviewType.BUSINESS,
                    url=self.web_driver.current_url,
                    page_of_pagination=current_page_pagination
                )
                self.trip_advisor_mongo_client.insert_url(url_object)
                paginated_url_list.append(self.web_driver.current_url)

                # load the next page
                self.web_driver.get(new_url)

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                break

        return paginated_url_list

    def get_and_store_paginated_review_urls(self, start_url: str) -> list[str]:
        return list[str]

    def get_and_store_business_urls(self, paginated_business_urls: list[str]) -> list[str]:
        return list[str]

    def get_and_store_reviews_urls(self, paginated_reviews_urls: list[str]) -> list[str]:
        return list[str]


url = "https://www.tripadvisor.com/Attractions-g189473-Activities-a_allAttractions.true" \
      "-Thessaloniki_Thessaloniki_Region_Central_Macedonia.html"
client = TripAdvisorMongoClient(
    connection_string="mongodb://localhost:27017",
    database="local",
    urls_collection="urls",
    reviews_collection="reviews"
)
scraper = TripAdvisorWebScrapper(trip_advisor_mongo_client=client)

paginated_business_urls = scraper.get_and_store_paginated_business_urls(url)
business_urls = scraper.get_and_store_business_urls(paginated_business_urls)

# business = Business(business_name="Agios Dimitrios")
# print(business.get_object_hash())
# print(business.to_mongo_document())
# reviewer = Reviewer(reviewer_username="tom", reviewer_name="jerry")
# item = Review(reviewer=reviewer, business=business)
# url = URL(url_type=URLType.ACCESS_URL, object_type=MongoReviewCollectionType.BUSINESS, url="this is a url")
#
# review = Review(
#     business=business,
#     reviewer=reviewer
# )
# print(review.to_mongo_document())
# client = TripAdvisorMongoClient(
#     connection_string="mongodb://localhost:27017",
#     database="local",
#     urls_collection="urls",
#     reviews_collection="reviews"
# )
# print(type(client.database))
# print(type(client.urls_collection))
# print(type(client.reviews_collection))
#
# print(client.insert_review(review))
# print(client.insert_reviewer(reviewer))
# print(client.insert_url(url))
#
# scraper = TripAdvisorWebScrapper(trip_advisor_mongo_client=client)
