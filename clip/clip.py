from modal import Stub, build, enter, method, Image, web_endpoint
from typing import Dict

image = Image.from_registry(
    "nvidia/cuda:12.3.1-base-ubuntu22.04", add_python="3.11"
).pip_install(
    "pillow",
    "requests",
    "transformers",
    "open_clip_torch",
    "torch",
)

stub = Stub("clip")


@stub.cls(
    timeout=60 * 5,
    container_idle_timeout=60,
    image=image,
)
class Clip:
    @build()
    @enter()
    def load_model(self):
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @method()
    def embed_text(self, text):
        import open_clip
        import torch

        with torch.no_grad(), torch.cuda.amp.autocast():
            text = self.tokenizer(text)
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.tolist()

            return text_features

    @method()
    def embed_image(self, url):
        import requests
        import torch
        from PIL import Image

        image = self.preprocess(
            Image.open(requests.get(url, stream=True).raw)
        ).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)[0]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.tolist()

            return image_features


@stub.function()
@web_endpoint(method="POST")
def square(data: Dict):
    if data.get("url") is not None:
        return {"embedding": Clip().embed_image.remote(data["url"])}
    elif data.get("text") is not None:
        return {"embedding": Clip().embed_text.remote(data["text"])}
    return {"error": "invalid request"}
