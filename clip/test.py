import requests
import os
import dotenv

dotenv.load_dotenv()

url = os.environ["API_URL"]

data = {
    "texts": ["photo of sea"],
    "images": ["https://brewadzddngodunjkxfx.supabase.co/storage/v1/object/public/images/1aadad69-5451-4039-902f-e294c965d260.png"],
}

response = requests.post(
    url, json=data, headers={"Authorization": "Bearer " + os.environ["AUTH_TOKEN"]}
)

def dot_product(list1, list2):
    return sum(x * y for x, y in zip(list1, list2))

def magnitude(vector):
    return sum(x ** 2 for x in vector) ** 0.5

def cosine_similarity(list1, list2):
    dot_prod = dot_product(list1, list2)
    mag1 = magnitude(list1)
    mag2 = magnitude(list2)
    return dot_prod / (mag1 * mag2)

if response.status_code == 200:
    print("Success!")
    result = response.json()
    print(result)
    image = result["images"][0]["embedding"]
    text = result["texts"][0]["embedding"]
    cosine_similarity = cosine_similarity(image, text)
    print("Cosine Similarity: ", cosine_similarity)

else:
    print("Error:", response.status_code)
