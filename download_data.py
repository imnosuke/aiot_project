"""Download and extract the cats/dogs dataset (idempotent, stdlib only).

Source: https://github.com/guilhermedom/resnet50-transfer-learning-cats-and-dogs/tree/main/data/raw
Result: $DATA_DIR/{cats,dogs}_{training,testing}/{cats,dogs}/*.jpg
"""
import os
import sys
import time
import urllib.request
import zipfile

DATA_DIR = os.environ.get("DATA_DIR", "data")
BASE_URL = ("https://raw.githubusercontent.com/guilhermedom/"
            "resnet50-transfer-learning-cats-and-dogs/main/data/raw/")
PARTS = ["cats_training", "cats_testing", "dogs_training", "dogs_testing"]


def download(url, dest, retries=3):
    for attempt in range(1, retries + 1):
        try:
            print(f"  -> {url} (attempt {attempt})", flush=True)
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  ! {e}", flush=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for part in PARTS:
        sub = part.split("_")[0]  # cats / dogs
        target = os.path.join(DATA_DIR, part, sub)
        if os.path.isdir(target) and any(f.endswith(".jpg") for f in os.listdir(target)):
            print(f"[skip] {part} already present")
            continue
        print(f"[get ] {part}", flush=True)
        zip_path = os.path.join(DATA_DIR, part + ".zip")
        download(BASE_URL + part + ".zip", zip_path)
        if not zipfile.is_zipfile(zip_path):
            sys.exit(f"{zip_path} is not a valid zip")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(os.path.join(DATA_DIR, part))  # archive contains cats/ or dogs/
        os.remove(zip_path)
        n = len([f for f in os.listdir(target) if f.endswith(".jpg")])
        print(f"[done] {part}: {n} images")
    print("Dataset ready.")


if __name__ == "__main__":
    main()
