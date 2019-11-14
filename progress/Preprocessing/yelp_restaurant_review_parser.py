import json
from pathlib import Path

data_folder = Path("./p-progress/Data/Train/")

business_file = data_folder / 'yelp_academic_dataset_business.json'
review_file = data_folder / 'yelp_academic_dataset_review.json'
positive_file = data_folder / 'yelp_positive_reviews'
negative_file = data_folder / 'yelp_negative_reviews'
neutral_file = data_folder / 'yelp_neutral_reviews'

business_data = []
review_data = []

with open(business_file, encoding="utf-8") as f:
    for line in f:
        business_data.append(json.loads(line))

with open(review_file, encoding="utf-8") as f:
    for line in f:
        review_data.append(json.loads(line))

is_restaurant = {}
for business in business_data:
    b_id = business["business_id"]
    if business["categories"] == None:
        continue
    categories = business["categories"].replace(" ", "").split(",")
    if "Restaurants" in categories:
        is_restaurant[b_id] = True

positive_reviews = []
negative_reviews = []
neutral_reviews = []

for review in review_data:
    b_id = review["business_id"]
    review["text"] = review["text"].replace("\n", " ")
    if b_id in is_restaurant:
        if review["stars"] > 3:
            positive_reviews.append(review["text"])
        elif review["stars"] < 3:
            negative_reviews.append(review["text"])
        else:
            neutral_reviews.append(review["text"])

def writeToFile(fp, text_list): 
    with open(fp, 'w+', encoding="utf-8") as f:
        for text in text_list:
            f.write(text)

writeToFile(positive_file, positive_reviews)
writeToFile(negative_file, negative_reviews)
writeToFile(neutral_file, neutral_reviews)

