import datetime
import hashlib
import json
class Business:
    """
    Business of TripAdvisor
    """

    def __init__(
            self,
            business_name: str = None,
            trip_advisor_url: str = None,
    ):
        self.business_name = business_name
        self.trip_advisor_url = trip_advisor_url

    def to_mongo_document(self) -> dict:
        """
        Serialize the object as a mongo document
        """
        return self

    def get_business_hash(self):
        """
        Calculate the hash of the business object,
        it will be used as a custom ID in MongoDB to avoid dublicate insertions
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
            reviewer_username: str = None,
            review_date: datetime = None,
            visit_data: str = None,
            review_text: str = None,
            review_rating: int = None,
    ):
        self.business = business
        self.reviewer_username = reviewer_username
        self.review_date = review_date
        self.visit_data = visit_data
        self.review_text = review_text
        self.review_rating = review_rating


    def to_mongo_document(self):
        """
        Serialize the object as a mongo document
        """
        return self

    def get_review_hash(self):
        """
        Calculate the hash of the business object,
        it will be used as a custom ID in MongoDB to avoid dublicate insertions
        """
        obj_json = json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
        hash_object = hashlib.sha256(obj_json.encode())
        return hash_object.hexdigest()

business = Business(business_name="Agios Dimitrios")
item = Review(reviewer_username="Tom3asd",business=business)
item2 = Review(reviewer_username="Tom",review_text="test")
item3 = Review(reviewer_username="Tom")
item4 = Review(reviewer_username="Tom")
print(item.get_review_hash())
print(item2.get_review_hash())
print(item3.get_review_hash())
print(item4.get_review_hash())
