"""Build the image folders used by train.py from the Pascal VOC annotations.

The "Bikes Helmets" dataset is annotated for object detection (one box per rider, labelled
"With Helmet" or "Without Helmet"). The classifier in this project needs one label per
image, so an image is labelled `no_helmet` if it contains at least one "Without Helmet"
box and `helmet` otherwise. Images are split 80 / 20 into train / test with a fixed seed.

Usage (from the project root, after downloading the dataset to data/raw/helmet_voc):
    python scripts/prepare_data.py
"""
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

RAW = Path("data/raw/helmet_voc")
OUT = Path("data/helmet_detection")
TEST_SIZE = 0.2
SEED = 42


def image_label(xml_path):
    labels = {obj.findtext("name", "").strip().lower() for obj in ET.parse(xml_path).getroot().iter("object")}
    if not labels:
        return None
    return "no_helmet" if "without helmet" in labels else "helmet"


def main():
    samples = []
    for xml_path in sorted((RAW / "annotations").glob("*.xml")):
        label = image_label(xml_path)
        image = RAW / "images" / (xml_path.stem + ".png")
        if label and image.exists():
            samples.append((image, label))

    random.Random(SEED).shuffle(samples)
    n_test = round(len(samples) * TEST_SIZE)
    split = {"test": samples[:n_test], "train": samples[n_test:]}

    for subset, items in split.items():
        for image, label in items:
            target = OUT / subset / label
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target / image.name)

    summary = {s: dict(Counter(label for _, label in items)) for s, items in split.items()}
    Path("results").mkdir(exist_ok=True)
    Path("results/data_split.json").write_text(
        json.dumps(
            {
                "rule": "no_helmet if any 'Without Helmet' box, else helmet",
                "seed": SEED,
                "counts": summary,
                "files": {s: sorted(i.name for i, _ in items) for s, items in split.items()},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
