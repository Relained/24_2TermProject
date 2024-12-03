import os
import json

from PIL import Image

# config.json 생성
def create_config_json(path=None, database_setting=None):
    # default value when no args
    default_path = os.getcwd()
    default_database_setting = {
        "name": "<DATABASE NAME>",
        "user": "<USERNAME>",
        "password": "<PASSWORD>",
        "host": "localhost",
        "port": 3306
    }

    config_data = {
        "path": path if path is not None else default_path,
        "database_setting": database_setting if database_setting is not None else default_database_setting,
    }

    config_file_path = os.path.join(path if path is not None else default_path, 'config.json')

    with open(config_file_path, "w", encoding="utf-8") as file:
        json.dump(config_data, file, indent=4, ensure_ascii=False)
    print(f"Config saved to {config_file_path}")

# config.json 읽기
# path, database_setting 반환
def load_config_json(path=None):
    # default path value
    main_path = path if path is not None else os.getcwd()
    config_path = os.path.join(main_path, 'config.json')

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        print(f"Config loaded from {config_path}")

        # 반환할 값
        path = config.get("path")
        database_setting = config.get("database_setting", {})

        return path, database_setting

    except FileNotFoundError:
        print(f"Error: config.json not found in {config_path}")
        return None, None
    except json.JSONDecodeError:
        print("Error: config.json is not a valid JSON file.")
        return None, None

# 이미지 파일을 바이너리로 열기
def open_image_binary(image_path):
    with open(image_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data

# 손상된 이미지 로드 오류 방지
def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # 파일 검증
        return True
    except (IOError, SyntaxError) as e:
        print(f"Corrupted image file skipped: {image_path} {e}")
        return False

# 폴더 내 모든 하위 폴더를 포함하여 GIF 및 WEBP 이미지를 PNG로 변환
# 변환된 파일들의 경로 리스트를 반환
def convert_images(folder_dir):
    converted_files = []
    valid_extensions = {'.gif', '.webp'}

    for root, dirs, files in os.walk(folder_dir):
        filtered_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]

        for file in filtered_files:
            file_path = os.path.join(root, file)
            output_path = os.path.join(root, os.path.splitext(file)[0] + '.png')

            # 이미지 검증
            if not validate_image(file_path):
                continue

            if file.lower().endswith('.gif'):
                try:
                    with Image.open(file_path) as img:
                        if img.is_animated:
                            img.seek(0)
                        img.convert("RGB").save(output_path, "PNG")
                        print(f"Converted GIF {file} to PNG format.")
                        converted_files.append(output_path)
                except Exception as e:
                    print(f"Failed to process GIF {file}: {e}")

            elif file.lower().endswith('.webp'):
                try:
                    with Image.open(file_path) as img:
                        if getattr(img, "is_animated", False):
                            img.seek(0)
                        img.convert("RGB").save(output_path, "PNG")
                        print(f"Converted WEBP {file} to PNG format.")
                        converted_files.append(output_path)
                except Exception as e:
                    print(f"Failed to process WEBP {file}: {e}")

    return converted_files

# storage 폴더 생성
def setup_storage(path):
    os.makedirs(os.path.join(path, 'storage'), exist_ok=True)
    os.makedirs(os.path.join(path, 'storage/classified'), exist_ok=True)
    os.makedirs(os.path.join(path, 'storage/dataset'), exist_ok=True)
    os.makedirs(os.path.join(path, 'storage/unclassified'), exist_ok=True)
    print(f"Storage path set to {path}.")