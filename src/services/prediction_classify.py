import os
import shutil
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from src.utils.fc import load_config_json
from src.services.make_model import CustomDataset

# 설정 로드
paths, database, _ = load_config_json(os.getcwd())

train_dataset = CustomDataset(paths["dataset_dir"])
class_count = len(train_dataset.class_to_name)  # 클래스 개수 계산

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 사용 중인 장치 출력
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
else:
    print("Using CPU.")

# 모델 정의
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# 커스텀 FC 레이어 추가 (저장된 모델과 일치시킴)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, class_count),
    nn.Softmax(dim=1)
)
model_path = "DeepL_TranferL/models/" + database["model_name"]
state_dict = torch.load(model_path, map_location=device)

# state_dict 로드 (strict=False)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()  # 평가 모드 설정
print(f"Model '{database['model_name']}' loaded successfully.")


# 이미지 전처리 함수 정의 (cv2 및 PyTorch transforms 사용)
def preprocess_image(img_path):
    try:
        # 이미지 형식에 따른 처리
        if img_path.lower().endswith(".gif"):
            cap = cv2.VideoCapture(img_path)
            ret, img = cap.read()  # 첫 번째 프레임을 읽음
            cap.release()
            if not ret:
                raise ValueError("Could not load GIF image")

        elif img_path.lower().endswith(".webp"):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not load WEBP image")

        else:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Could not load image")

        # 투명 채널을 포함한 이미지 처리
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # PyTorch 전처리 파이프라인
        preprocess = transforms.Compose([
            transforms.ToPILImage(),  # OpenCV 이미지 -> PIL 이미지 변환
            transforms.Resize((224, 224)),  # 크기 조정
            transforms.ToTensor(),  # 텐서 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
        ])
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가
        return img_tensor.to(device)

    except Exception as e:
        print(f"Error loading image: {img_path}")
        print(e)
        return None


# 예측 및 분류 함수
def classify_image(img_path, connection=None):
    # 이미지 전처리
    img_tensor = preprocess_image(img_path)

    if img_tensor is not None:
        with torch.no_grad():
            # 모델 예측
            outputs = model(img_tensor)
            _, predicted_class = torch.max(outputs, 1)  # 가장 높은 점수의 클래스 선택
            predicted_class = predicted_class.item()

        img_name = os.path.basename(img_path)

        # 클래스 레이블 가져오기 (폴더 이름 매핑)
        class_label = train_dataset.class_to_name.get(predicted_class, f"class_{predicted_class}")

        # 예측된 폴더로 이미지 이동
        dest_dir = os.path.join(paths["classified_dir"], class_label)
        os.makedirs(dest_dir, exist_ok=True)  # 폴더 생성
        output_path = os.path.join(dest_dir, img_name)
        shutil.move(img_path, output_path)
        print(f"Image '{img_name}' has been classified as '{class_label}' and moved to '{dest_dir}'.")

        # DB에 데이터 삽입 (선택적으로 활성화 가능)
        # insert_image(connection, predicted_class, output_path)


def evaluate_model(test_dir=paths["dataset_dir"], batch_size=32):

    # 데이터 전처리 (테스트용)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지 크기 조정 (모델에 맞게 변경 가능)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 기준 정규화
    ])

    # 테스트 데이터셋 로드
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # 일부 데이터만 테스트에 사용 (test_fraction 비율만큼 나누기)
    test_size = int(len(test_dataset) * 0.2)
    _, test_subset = random_split(test_dataset, [len(test_dataset) - test_size, test_size])

    # DataLoader 생성
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 그래디언트 계산 비활성화
        for images, labels in test_loader:
            # 데이터와 라벨을 디바이스로 이동
            images, labels = images.to(device), labels.to(device)

            # 모델 예측
            outputs = model(images)

            # 손실 계산
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)  # 배치 크기만큼 가중치 적용

            # 정확도 계산
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 평균 손실 및 정확도 계산
    avg_loss = total_loss / total
    accuracy = correct / total
    # 결과 출력
    print(f"Normalized Loss (0~1): {avg_loss:.4f}")
    print(f"Normalized Accuracy (0~1): {accuracy:.4f}")
    return avg_loss, accuracy