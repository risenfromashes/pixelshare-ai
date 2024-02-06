import requests

url = "https://risenfromashes--clip-square-dev.modal.run"

# data = {"url": "https://images.pixelshare.site/mega.png"}
data = {"text": "A very beautiful girl"}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Success!")
    print(response.json())
else:
    print("Error:", response.status_code)
