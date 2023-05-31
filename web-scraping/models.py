import datetime
import hashlib
import json
import pymongo


# secondary models
class MongoReviewCollectionType:
    BUSINESS = "business"
    REVIEW = "review"
    REVIEWER = "reviewer"
    URL = "url"


class URLType:
    PAGINATION_URL = "pagination url"
    ACCESS_URL = "access url"


class ObjectType:
    BUSINESS = "business"
    REVIEW = "review"


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
            "type": MongoReviewCollectionType.BUSINESS,
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
            "type": MongoReviewCollectionType.REVIEWER,
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
            "type": MongoReviewCollectionType.REVIEW,
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
            url_type: URLType,
            object_type: ObjectType,
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
            "type": MongoReviewCollectionType.URL,
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
            connection_string: str,
            database: pymongo.database.Database,
            urls_collection: pymongo.collection.Collection or str,
            reviews_collection: pymongo.collection.Collection or str,

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

    def insert_review(self, review: Review):
        if self.reviews_collection.find_one({"unique_id": review.get_object_hash()}) is not None:
            return False  # document exist
        self.reviews_collection.insert_one(review.to_mongo_document())
        return True
    def insert_reviewer(self, reviewer: Reviewer):
        if self.reviews_collection.find_one({"unique_id": reviewer.get_object_hash()}) is not None:
            return False  # document exist
        self.reviews_collection.insert_one(reviewer.to_mongo_document())
        return True

    def insert_url(self, url: URL):
        if self.urls_collection.find_one({"unique_id": url.get_object_hash()}) is not None:
            return False  # document exist
        self.urls_collection.insert_one(url.to_mongo_document())
        return True


business = Business(business_name="Agios Dimitrios")
print(business.get_object_hash())
print(business.to_mongo_document())
reviewer = Reviewer(reviewer_username="tom", reviewer_name="jerry")
item = Review(reviewer=reviewer, business=business)
url = URL(url_type=URLType.ACCESS_URL,object_type=ObjectType.BUSINESS,url="this is a url")


review = Review(
    business=business,
    reviewer=reviewer
)
print(review.to_mongo_document())
client = TripAdvisorMongoClient(
    connection_string="mongodb://localhost:27017",
    database="local",
    urls_collection="urls",
    reviews_collection="reviews"
)
print(type(client.database))
print(type(client.urls_collection))
print(type(client.reviews_collection))

print(client.insert_review(review))
print(client.insert_reviewer(reviewer))
print(client.insert_url(url))

