import os
import shutil
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from functions.fc import load_config_json, create_mapping_and_class_count
from functions.db import insert_image

paths, database, _ = load_config_json(os.getcwd() + r"\config.json")
mapping_table, _ = create_mapping_and_class_count(paths["dataset_dir"])

# 모델 불러오기
model = load_model("DeepL_TranferL/models/" + database["model_name"])
print(f"Model '{database["model_name"]}' loaded successfully.")

# 이미지 전처리 함수 정의 (cv2 사용)
def preprocess_image(img_path):
    try:
        # 이미지 형식에 따른 처리
        if img_path.lower().endswith(".gif"):
            # GIF 파일의 첫 번째 프레임만 사용
            cap = cv2.VideoCapture(img_path)
            ret, img = cap.read()  # 첫 번째 프레임을 읽음
            cap.release()
            if not ret:
                raise ValueError("Could not load GIF image")
        
        elif img_path.lower().endswith(".webp"):
            # WEBP 파일 읽기
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not load WEBP image")
        
        else:
            # 다른 포맷의 이미지를 일반적으로 로드
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Could not load image")

        # 이미지가 투명 채널을 포함하면 RGB로 변환
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 이미지 크기 조정 및 정규화
        img = cv2.resize(img, (224, 224))  # 크기 조정
        img_array = img / 255.0  # 0-1 범위로 정규화
        img_array = np.expand_dims(img_array, axis=0)  # 모델 입력 형식에 맞게 배치 차원 추가
        return img_array

    except Exception as e:
        print(f"Error loading image: {img_path}")
        print(e)
        return None

# 예측 및 분류
def classify_image(img_path, connection):
    img = preprocess_image(img_path)

    # 모델 예측
    if img is not None:
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        img_name = os.path.basename(img_path)

        # 클래스 레이블 가져오기 (매핑 테이블 사용)
        class_label = mapping_table.get(predicted_class, f"class_{predicted_class}")

        # 예측된 폴더로 이미지 이동
        dest_dir = os.path.join(paths["classified_dir"], class_label)
        os.makedirs(dest_dir, exist_ok=True)  # 폴더가 없으면 생성
        shutil.move(img_path, os.path.join(dest_dir, img_name))
        print(f"{img_name} -> {class_label}")

        # DB에 데이터 insert
        insert_image()

