import os
import json

# 이미지 파일을 바이너리로 열기
def open_image_binary(image_path):
    with open(image_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data

# config.json 생성
def create_config_json(base_dir, paths=None, database=None, allowed_extensions=None):
    # default value when no args
    default_paths = {
        "dataset_dir": os.path.join(base_dir, r"storage\dataset"),
        "unclassified_dir": os.path.join(base_dir, r"storage\unclassified"),
        "classified_dir": os.path.join(base_dir, r"storage\classified")
    }
    default_database = {
        "name": "noDG",
        "model_name": "my_trained_model",
        "user": "root",
        "password": "<PASSWORD>",
        "host": "localhost",
        "port": 3306
    }
    default_allowed_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".raw",
        ",ico",
        ".pdf",
        ".tiff"
    ]
    
    config_data = {
        "paths": paths if paths is not None else default_paths,
        "database": database if database is not None else default_database,
        "allowed_extensions": allowed_extensions if allowed_extensions is not None else default_allowed_extensions
    }
    
    config_dir = os.path.join(base_dir, "config.json")
    with open(config_dir, "w", encoding="utf-8") as file:
        json.dump(config_data, file, indent=4, ensure_ascii=False)
    print(f"Config saved to {config_dir}")

# config.json 읽기
def load_config_json(base_dir):
    config_path = os.path.join(base_dir, "config.json")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        print(f"Config loaded from {config_path}")

        # 반환할 값
        paths = config.get("paths", {})
        database = config.get("database", {})
        allowed_extensions = config.get("allowed_extensions", [])

        return paths, database, allowed_extensions

    except FileNotFoundError:
        print(f"Error: config.json not found in {base_dir}")
        return {}, {}, []
    except json.JSONDecodeError:
        print("Error: config.json is not a valid JSON file.")
        return {}, {}, []
