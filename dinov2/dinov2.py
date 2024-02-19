import os
from modal import Secret, Stub, build, enter, method, Image, web_endpoint
from typing import Dict
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

image = Image.from_registry(
    "nvidia/cuda:12.3.1-base-ubuntu22.04", add_python="3.11"
).pip_install(
    "pillow",
    "requests",
    "transformers",
    "torch",
)

stub = Stub("dinov2")


@stub.cls(
    timeout=60 * 5,
    container_idle_timeout=60 * 10,
    image=image,
)
class DinoV2:
    @build()
    @enter()
    def load_model(self):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.model = AutoModel.from_pretrained("facebook/dinov2-large")

    @method()
    def embed_image(self, url):
        import requests
        from PIL import Image

        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states[0][0].tolist()


auth_scheme = HTTPBearer()


@stub.function(secrets=[Secret.from_name("web-auth-token")])
@web_endpoint(method="POST")
def encode(data: Dict, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    images = []
    if data.get("images") is not None:
        for url in data["images"]:
            images.append({"url": url, "embedding": DinoV2().embed_image.remote(url)})

    return {"images": images}
