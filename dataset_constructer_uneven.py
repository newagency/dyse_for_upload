import os
import json
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from datetime import datetime
from collections import defaultdict


def create_5_subsets_cifar100(output_dir='./subsets', base_name='cifar100_subset'):
    """
    CIFAR-100을 5개 서브셋으로 분할:
      - subset_4: 모든 클래스에서 100장씩 균등하게 (각 클래스 100장 => 총 10000장)
      - subset_0~3: 남은 400장을 4개 서브셋에 무작위(불균등) 분배

    각 서브셋을 (.pt, .json) 형태로 저장하고,
    메타데이터에는 class_distribution, usage_count=0, creation_time 등을 기록.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) CIFAR-100 로드 (train)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    cifar100_train = CIFAR100(root='./data_cifar100', train=True, download=True, transform=transform)

    # 2) 클래스별 인덱스 수집
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(cifar100_train):
        class_indices[label].append(idx)

    # 서브셋 인덱스 저장용 (subset_0..subset_4)
    subset_indices_list = [[] for _ in range(5)]  # 0~3: 불균등, 4: 균등

    # 3) 각 클래스에 대해 분배 로직
    #    - shuffle -> 첫 100장 -> subset_4 (균등)
    #    - 나머지 400장을 subset_0..3에 무작위 분배
    for c in range(100):
        indices = class_indices[c]
        random.shuffle(indices)

        # 균등 서브셋(마지막 subset_4)에 100장 할당
        uniform_part = indices[:100]
        subset_indices_list[4].extend(uniform_part)

        # 나머지 400장은 subset_0..3에 불균등 분배
        leftover = indices[100:]
        # leftover 길이는 400
        # 예: random partition for leftover among subsets 0..3
        # 방법1) 임의의 비율 [a0,a1,a2,a3] 합=400
        # 여기서는 간단히 leftover를 무작위로 섞은 뒤 임의크기로 4등분
        random.shuffle(leftover)

        # 예를 들어 [subset_0, subset_1, subset_2, subset_3] 각각 임의개수
        # 한 가지 접근: generate 3 random cut points in range(0..400), sort them, slice
        cut_points = sorted(random.sample(range(1, 400), 3))
        # 예: cut_points=[x, y, z] (x<y<z)
        # leftover[:x] -> subset_0
        # leftover[x:y] -> subset_1
        # leftover[y:z] -> subset_2
        # leftover[z:] -> subset_3

        s0 = leftover[:cut_points[0]]
        s1 = leftover[cut_points[0]:cut_points[1]]
        s2 = leftover[cut_points[1]:cut_points[2]]
        s3 = leftover[cut_points[2]:]

        subset_indices_list[0].extend(s0)
        subset_indices_list[1].extend(s1)
        subset_indices_list[2].extend(s2)
        subset_indices_list[3].extend(s3)

    # 4) 이제 subset_indices_list[0..4] 각각에 대해 .pt + .json 생성
    for subset_id in range(5):
        subset_indices = subset_indices_list[subset_id]

        # 실제 텐서로 로딩
        loader = DataLoader(subset_indices, batch_size=len(subset_indices), shuffle=False,
                            collate_fn=lambda batch: collate_cifar_indices(batch, cifar100_train))
        images_tensor, labels_tensor = None, None
        for images, labels in loader:
            images_tensor = images
            labels_tensor = labels
            break

        # 저장 파일명
        pt_filename = f"{base_name}_{subset_id}.pt"
        pt_filepath = os.path.join(output_dir, pt_filename)

        torch.save({"images": images_tensor, "labels": labels_tensor}, pt_filepath)

        # 클래스 분포 계산
        class_dist = {}
        for lab in labels_tensor:
            c = int(lab.item())
            class_dist[c] = class_dist.get(c, 0) + 1

        creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subset_name = f"{base_name}_{subset_id}"

        metadata = {
            "subset_name": subset_name,
            "num_samples": len(subset_indices),
            "class_distribution": class_dist,  # {class_label: count}
            "usage_count": 0,
            "creation_time": creation_time,
            "info": f"subset_{subset_id}: {'uniform' if subset_id == 4 else 'non-uniform'} distribution"
        }

        json_filename = f"{base_name}_{subset_id}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        with open(json_filepath, 'w') as jf:
            json.dump(metadata, jf, indent=4)

        print(f"[Subset {subset_id}] Saved {pt_filename} + {json_filename} | samples={len(subset_indices)}")


def collate_cifar_indices(batch, dataset):
    """
    batch: list of indices for the original dataset
    dataset: the CIFAR-100 dataset object
    Return: (images, labels) tensor
    """
    indices = batch  # entire batch is the list of indices
    images_list = []
    labels_list = []
    for idx in indices:
        img, lab = dataset[idx]
        images_list.append(img.unsqueeze(0))  # [1, C, H, W]
        labels_list.append(lab)
    images_tensor = torch.cat(images_list, dim=0)  # [N, 3, 32, 32]
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    return images_tensor, labels_tensor


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    create_5_subsets_cifar100(output_dir='./subsets_cifar100', base_name='cifar100_subset')
    print("Done. Created 5 subsets: 4 non-uniform, 1 uniform (subset_4).")


if __name__ == "__main__":
    main()
