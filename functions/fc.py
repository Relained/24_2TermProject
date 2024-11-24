import os
import json

# 이미지 파일을 바이너리로 열기
def open_image_binary(image_path):
    with open(image_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data

# 디렉터리 구조를 인식하여 매핑 테이블 및 세분류의 개수 반환
def create_mapping_and_class_count(directory):

    stack = [(directory, "")]
    mapping_table = {}
    class_count = 0
    class_id = 0

    while stack:
        current_dir, current_path = stack.pop()
        entries = sorted(os.listdir(current_dir))

        has_subfolders = False
        for entry in entries:
            entry_path = os.path.join(current_dir, entry)
            if os.path.isdir(entry_path):
                # 현재 폴더 경로 업데이트
                sub_path = os.path.join(current_path, entry)
                stack.append((entry_path, sub_path))
                has_subfolders = True

        if not has_subfolders:  # 하위 폴더가 없으면 세분류로 간주
            mapping_table[class_id] = os.path.join(current_path)
            class_count += 1
            class_id += 1

    return mapping_table, class_count

# config.json 생성
def create_config_json(base_dir, paths=None, database=None, allowed_extensions=None):
    # default value when no args
    default_paths = {
        "dataset_dir": os.path.join(base_dir, r"storage\dataset"),
        "unclassified_dir": os.path.join(base_dir, r"storage\unclassified"),
        "classified_dir": os.path.join(base_dir, r"storage\classfied")
    }
    default_database = {
        "name": "noDG",
        "model_name": "my_trained_model.h5",
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

