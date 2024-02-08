import os

import numpy as np
import torch
import uvicorn
from config import ArtClassifierConstants
from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import transforms as T

from initial_model_utils import init_model

config = ArtClassifierConstants()

device = torch.device("cpu")

model = init_model(device, num_classes=len(config.id2label), pretrained=False)
model.load_state_dict(torch.load(config.model_weights_path, map_location=device))
model.eval()

trans = T.Compose(
    [
        T.Resize(size=(config.image_size, config.image_size)),
        T.Lambda(lambda x: np.array(x, dtype="float32") / 255),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

app = FastAPI()


@app.post("/predict")
def get_art_label(image: UploadFile):
    """Prediction endpoint

    Parameters
    ----------
    image : UploadFile
        Image to predict
    """
    with torch.no_grad():
        image = Image.open(image.file).convert("RGB")
        transformed_image = torch.unsqueeze(trans(image), 0)
        outputs = model(transformed_image)
        _, preds = torch.max(outputs, 1)
        pred_pos = preds.cpu().numpy()[0]

    return {"sub_category": config.id2label[pred_pos]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
