import numpy as np
from scipy.io import savemat

data = np.load("breastmnist.npz")

savemat("breastmnist.mat", {
    "train_images": data["train_images"],
    "train_labels": data["train_labels"],
    "val_images": data["val_images"],
    "val_labels": data["val_labels"],
    "test_images": data["test_images"],
    "test_labels": data["test_labels"],
})

print("breastmnist.mat created successfully!")
