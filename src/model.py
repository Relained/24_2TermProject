import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from src.utils import load_config_json, convert_images

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 설정 및 데이터 경로 로드
path, _ = load_config_json()
processed_dataset_dir = os.path.join(path, "storage/processed")  # 변환된 이미지 저장 경로
dataset_dir = os.path.join(path, "storage/dataset")
classified_dir = os.path.join(path, "storage/classified")

# 모델 선택
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# 사용자 정의 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_name = {}

        # 디렉터리 순회하며 이미지 경로 및 레이블 수집
        class_idx = 0
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_name[class_idx] = class_name
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path):
                        self.image_paths.append(file_path)
                        self.labels.append(class_idx)
                class_idx += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 데이터 로드 및 전처리
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 사용자 정의 데이터셋 로드
train_dataset = CustomDataset(dataset_dir, transform=data_transforms['train'])
val_dataset = CustomDataset(dataset_dir, transform=data_transforms['val'])
class_count = len(train_dataset.class_to_name)  # 클래스 개수 계산

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
}

# 모델 수정: 출력 레이어를 클래스 수에 맞게 변경
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, class_count),
    nn.LogSoftmax(dim=1)
)

# 학습 및 검증 손실과 정확도를 시각화
def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # 손실 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(path, "storage/_loss.png"))

    # 정확도 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(path, "storage/_accuracy.png"))

# CPU 혹은 CUDA 중 하나를 선택
# torch.device 반환
def select_device():
    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 사용 중인 장치 출력
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
    else:
        print("Using CPU.")
    return device

# 이미지 전처리 함수
# 이미지를 처리하여 Pytorch 전처리 파이프라인에 전송
def preprocess_image(img_path, device):
    try:
        # Pillow를 사용해 이미지 읽기
        if img_path.lower().endswith(".gif"):
            with Image.open(img_path) as img:
                img.seek(0)  # 첫 번째 프레임 읽기

        elif img_path.lower().endswith(".webp"):
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # 투명 채널 제거

        else:
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # 일반 이미지

        # PyTorch 전처리 파이프라인
        preprocess = transforms.Compose([
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

def make_model():
    # 변환 함수 실행 (변환된 파일 리스트 반환)
    print("Converting images...")
    copied_img = convert_images(dataset_dir)

    # 사용할 장치 선택
    device = select_device()
    device_model = model.to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(device_model.parameters(), lr=0.0001)

    # 학습 루프
    num_epochs = 10
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                device_model.train()
            else:
                device_model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = device_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # 기록 저장
            if phase == 'train':
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # 학습 진행 시각화
    plot_training_history(history)

    # 학습 완료된 모델 저장
    torch.save(device_model.state_dict(), os.path.join(path, "storage/custom_model.pth"))
    print("Custom Model saved to storage.")

    # 변환된 이미지 복사본 삭제
    print("Cleaning up converted images...")
    for file in copied_img:
        try:
            os.remove(file)
            print(f"Deleted {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

    print("Training complete.")

# 학습된 state_dict 를 로드
def load_model():
    # state_dict 로드
    model.load_state_dict(torch.load(os.path.join(path, "storage/custom_model.pth"), weights_only=True))
    device = select_device()
    device_model = model.to(device)
    device_model.eval()
    print("Model loaded successfully.")

    return device_model

# 클래스 예측 및 이미지 분류
def classify_image(device_model, img_path):
    device = next(device_model.parameters()).device

    # 이미지 전처리
    img_tensor = preprocess_image(img_path, device)

    if img_tensor is not None:
        with torch.no_grad():
            # 모델 예측
            outputs = device_model(img_tensor)
            _, predicted_class = torch.max(outputs, 1)  # 가장 높은 점수의 클래스 선택
            predicted_class = predicted_class.item()

        img_name = os.path.basename(img_path)

        # 클래스 레이블 가져오기 (폴더 이름 매핑)
        class_label = train_dataset.class_to_name.get(predicted_class, f"class_{predicted_class}")

        # 예측된 폴더로 이미지 이동
        dest_dir = os.path.join(classified_dir, class_label)
        os.makedirs(dest_dir, exist_ok=True)  # 폴더 생성
        output_path = os.path.join(dest_dir, img_name)
        shutil.move(img_path, output_path)
        print(f"Image '{img_name}' classified as '{class_label}' and moved to '{dest_dir}'.")


# 폴더의 이미지 연속 분류
def classify_from_folder(device_model, folder_path):
    # 폴더 내 이미지 파일 수집
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    if not image_paths:
        print("No valid images found in the folder.")
        return

    for img_path in image_paths:
        classify_image(device_model, img_path)