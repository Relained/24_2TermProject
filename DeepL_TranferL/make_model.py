import os
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from functions.fc import load_config_json
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 설정 및 데이터 경로 로드
paths, database, _ = load_config_json(os.getcwd())
processed_dataset_dir = os.getcwd() + r"\processed_data"  # 변환된 이미지 저장 경로


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


# 손상된 이미지 로드 오류 방지
def validate_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 파일 검증
        return True
    except (IOError, SyntaxError) as e:
        print(f"Corrupted image file skipped: {file_path}")
        return False

# GIF 및 WEBP 이미지를 PNG로 변환 (OpenCV 사용)
def convert_images(input_dir):
    converted_files = []
    valid_extensions = {'.gif', '.webp'}

    for root, dirs, files in os.walk(input_dir):
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

# 전이 학습을 사용하여 ResNet50 모델 구성 및 학습
# 전이 학습을 사용하여 ResNet50 모델 구성 및 학습
def make_model():
    # 변환 함수 실행 (변환된 파일 리스트 반환)
    print("Converting images...")
    copied_img = convert_images(paths["dataset_dir"])

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
    train_dataset = CustomDataset(paths["dataset_dir"], transform=data_transforms['train'])
    val_dataset = CustomDataset(paths["dataset_dir"], transform=data_transforms['val'])
    class_count = len(train_dataset.class_to_name)  # 클래스 개수 계산

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
    }

    # ResNet50 모델 불러오기 (사전 학습된 가중치 사용)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # 모델 수정: 출력 레이어를 클래스 수에 맞게 변경
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, class_count),
        nn.Softmax(dim=1)
    )

    # GPU 사용 여부 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 사용 중인 장치 출력
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
    else:
        print("Using CPU.")

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 학습 루프
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # 학습 완료된 모델 저장
    os.makedirs("DeepL_TranferL/models", exist_ok=True)
    torch.save(model.state_dict(), f"DeepL_TranferL/models/{database['model_name']}")
    print(f"Model saved as '{database['model_name']}'")

    # 변환된 이미지 복사본 삭제
    print("Cleaning up converted images...")
    for file in copied_img:
        try:
            os.remove(file)
            print(f"Deleted {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

    print("Training complete.")