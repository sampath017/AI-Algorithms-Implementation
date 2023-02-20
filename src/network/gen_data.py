from pathlib import Path
import zipfile
import cv2

import matplotlib.pyplot as plt
import numpy as np

import requests


def download_and_extract(url='https://nnfs.io/datasets/fashion_mnist_images.zip', target_dir="data", exist_ok=False):
    if not Path(target_dir).exists() or exist_ok:
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        response = requests.get(url)
        file_path = Path(target_dir) / Path(url).name
        with open(file_path, "wb") as f:
            print(f'Downloading {url} and saving file.')
            f.write(response.content)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            print('Unzipping images...')
            zip_ref.extractall(target_dir)

        file_path.unlink()
        print('Done!')
    else:
        print("Folder already exists")


def load_mnist(dataset, target_dir="data"):
    data_dir = Path(target_dir) / dataset
    labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    X = []
    y = []

    for label in labels:
        label_dir = data_dir / label
        for file in label_dir.iterdir():
            image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):

    download_and_extract()

    X_train, y_train = load_mnist('train', path)
    X_test, y_test = load_mnist('test', path)

    return X_train, y_train, X_test, y_test
