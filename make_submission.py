"""Script for generation competition's submission.
All constants remains with competition rules (cpu usage)
"""

import numpy as np
import torch
from torchvision import transforms

from src.initial_model_utils import init_model
from src.train_utils import ArtDataset

MODEL_WEIGHTS = "./data/weights/resnet50_tl_68.pt"
TEST_DATASET = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"

if __name__ == "__main__":
    device = torch.device("cpu")
    model = init_model(device, num_classes=35)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    img_size = 224
    trans = transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.Lambda(lambda x: np.array(x, dtype="float32") / 255),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ard_data = ArtDataset(TEST_DATASET, transform=trans)
    batch_size = 16
    num_workers = 4
    testloader = torch.utils.data.DataLoader(
        ard_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    all_image_names = [item.split("/")[-1] for item in ard_data.files]
    all_preds = []

    with torch.no_grad():
        for idx, (images, _) in enumerate(testloader, 0):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())

    with open(SUBMISSION_PATH, "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_image_names, all_preds):
            f.write(f"{name}\t{cl_id}\n")
