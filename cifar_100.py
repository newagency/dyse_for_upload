import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import numpy as np


def load_cifar100_subset(batch_size=64, samples_per_class=50, num_workers=2):
    """
    CIFAR-100 train 데이터셋에서 각 클래스별 'samples_per_class'만 골라
    Subset을 만든 뒤, DataLoader를 반환한다.
    100개 클래스를 모두 포함하되, 샘플 수를 줄여 학습 속도를 높이고 메모리를 절약할 수 있음.

    Args:
        batch_size (int): 미니배치 크기
        samples_per_class (int): 각 클래스당 몇 개 이미지를 사용할지
        num_workers (int): 데이터 로딩에 사용할 서브 프로세스 수

    Returns:
        DataLoader: 클래스별로 샘플이 제한된 Subset
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))  # CIFAR-100 정규화 통계값
    ])

    # Train셋 로드 (전체 50k 이미지)
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    # 클래스별 인덱스를 수집
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)

    # 각 클래스에서 samples_per_class만큼 샘플링
    subset_indices = []
    for label, indices in class_indices.items():
        if len(indices) > samples_per_class:
            chosen = np.random.choice(indices, samples_per_class, replace=False)
        else:
            chosen = indices  # 클래스 내 이미지가 적다면 전부 사용
        subset_indices.extend(chosen)

    # Subset으로 새로운 Dataset 구성
    subset_dataset = Subset(train_dataset, subset_indices)

    # DataLoader 생성
    train_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return train_loader


def create_resnet18_cifar100():
    """
    torchvision.models.resnet18을 가져와서,
    CIFAR-100(클래스 100개)에 맞도록 마지막 FC 레이어를 수정.
    """
    import torchvision.models as models
    model = models.resnet18(weights=None)  # weights=None -> From scratch
    model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-100 (100 classes)
    return model


def train_subset_model(epochs=10, batch_size=64, samples_per_class=50, lr=0.001, num_workers=2):
    """
    클래스별로 일정 샘플만 포함하는 CIFAR-100 Subset을 이용해 ResNet-18 모델을 학습.
    학습 완료 후 모델 가중치를 .pth 파일로 저장.

    Args:
        epochs (int): 학습 epoch 수
        batch_size (int): 배치 사이즈
        samples_per_class (int): 각 클래스별 샘플 개수
        lr (float): Learning Rate
        num_workers (int): DataLoader num_workers
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available(), "gpu")
    # Subset Dataloader 준비
    train_loader = load_cifar100_subset(batch_size=batch_size,
                                        samples_per_class=samples_per_class,
                                        num_workers=num_workers)

    # CIFAR-100 Validation/Test Dataloader (전체 사용)
    # 실제 코드에서는 test_loader도 subset 아님. 여기서는 'train=False'로 불러옴
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 모델 생성
    model = create_resnet18_cifar100().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation(Test) accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100.0 * correct / total
        print(f"[Epoch {epoch}/{epochs}] Loss: {avg_train_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # 모델 저장
    torch.save(model.state_dict(), f"resnet18_cifar100_subset_{samples_per_class}.pth")
    print(f"Model saved to resnet18_cifar100_subset_{samples_per_class}.pth")


if __name__ == "__main__":
    # 예시 실행
    # 100개 클래스를 모두 포함하되, 각 클래스당 50개만 사용 -> 총 5,000개 이미지
    train_subset_model(epochs=10, batch_size=64, samples_per_class=20, lr=0.001, num_workers=2)
