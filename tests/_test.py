import os
from src.utils.fc import load_config_json
from src.services.prediction_classify import classify_image

paths, database, allowed_ext = load_config_json(os.getcwd())

for root, _, files in os.walk(paths["unclassified_dir"]):
    for file in files:
        if os.path.splitext(file)[1].lower() in allowed_ext:
            img_path = os.path.join(root, file)
            print(f"Processing {img_path}")
            try:
                classify_image(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# evaluate_model()