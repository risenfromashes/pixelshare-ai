import requests
import os
import dotenv

dotenv.load_dotenv()

url = os.environ["API_URL"]

data = {
    "texts": ["Photo of a rose"],
    "images": ["https://images.pixelshare.site/Rosa_Precious_platinum.jpg"],
}

response = requests.post(
    url, json=data, headers={"Authorization": "Bearer " + os.environ["AUTH_TOKEN"]}
)

if response.status_code == 200:
    print("Success!")
    print(response.json())
else:
    print("Error:", response.status_code)
