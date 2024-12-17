# server.py
import os
import json
import boto3
import tempfile
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from flask import Flask, request, jsonify

app = Flask(__name__)
s3 = boto3.client('s3')  # Ensure your EC2 IAM Role has S3 read/write access

MODEL_FILE_CACHE = None  # global variable to store path of uploaded model in memory

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    ds_type = request.args.get('type', 'cifar100')
    if ds_type == 'cifar100':
        bucket = 'dyse-all'
        prefix = 'subsets_cifar100/'
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        subsets_info = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json'):
                local_json = download_temp(bucket, key)
                with open(local_json, 'r') as f:
                    meta = json.load(f)
                filtered_meta = {
                    "subset_name": meta.get("subset_name"),
                    "num_samples": meta.get("num_samples"),
                    "class_distribution": meta.get("class_distribution"),
                    "usage_count": meta.get("usage_count", 0),
                    "info": meta.get("info", "N/A")
                }
                subsets_info.append(filtered_meta)
        return jsonify(subsets_info)
    else:
        return jsonify([])

#@app.route('/api/estimate_time', methods=['POST'])
#def estimate_time():
#    global MODEL_FILE_CACHE
#    model_file = request.files.get('modelFile', None)
#    subsets_str = request.form.get('subsets', '[]')
#    subsets = json.loads(subsets_str)
#
#    with tempfile.TemporaryDirectory() as tmpdir:
#        if model_file:
#            local_model = os.path.join(tmpdir, 'uploaded_model.pth')
#            model_file.save(local_model)
#            MODEL_FILE_CACHE = local_model
#
#        combined_ds = combine_and_sample(subsets, tmpdir)
#
#        # NTK 기반 훈련 시간 예측 알고리즘 구현
#        num_samples = sum([sub['num_samples'] for sub in subsets])
#        learning_rate = 0.001
#        momentum = 0.9
#        threshold = 0.01
#        estimated_steps = estimate_ntk_time(num_samples, learning_rate, momentum, threshold)
#        estimated_time_minutes = round(estimated_steps * 0.5 / 60, 2)  # mock time conversion
#
#    return jsonify({"estimated_time": f"{estimated_time_minutes} minutes"})
#
#def estimate_ntk_time(samples, lr, momentum, epsilon):
#    # Pseudo Code 기반 구현
#    elr = lr / (1 - momentum)
#    steps = int((samples / elr) * epsilon * 10)  # Simplified NTK-based steps estimation
#    return steps

@app.route('/api/start_training', methods=['POST'])
def start_training():
    global MODEL_FILE_CACHE
    subsets_str = request.form.get('subsets', '[]')
    subsets = json.loads(subsets_str)
    epochs = int(request.form.get('epochs', '5'))
    batch_size = int(request.form.get('batch_size', '64'))

    if MODEL_FILE_CACHE is None:
        return jsonify({"error": "No model file in cache"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        combined_ds = combine_and_sample(subsets, tmpdir)
        train_loader = DataLoader(combined_ds, batch_size=batch_size, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load the model from MODEL_FILE_CACHE
        model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 100)
        state_dict = torch.load(MODEL_FILE_CACHE, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        final_loss = 0.0
        for e in range(epochs):
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
            final_loss = running_loss / len(train_loader)
            print(f"Epoch[{e + 1}/{epochs}] - Loss={final_loss:.4f}")

        # Evaluate on a test set (not shown, just mock)
        final_acc = 0.88  # mock

    return jsonify({"success": True, "final_acc": final_acc, "final_loss": final_loss})


def combine_and_sample(subsets, tmpdir):
    """
    Download each subset .pt from S3, then do class distribution alignment.
    For simplicity, we just Concat them without sampling logic.
    Modify to implement uniform sampling if needed.
    """
    all_ds = []
    for sub in subsets:
        pt_s3_uri = sub['pt']  # e.g. s3://dyse-all/subsets_cifar100/cifar100_subset_0.pt
        local_pt = os.path.join(tmpdir, os.path.basename(pt_s3_uri))
        download_from_s3(pt_s3_uri, local_pt)
        ds = load_pt_dataset(local_pt)
        # TODO: uniform sampling by class distribution
        # sub["class_distribution"] can guide sampling
        all_ds.append(ds)
    combined_dataset = ConcatDataset(all_ds)
    return combined_dataset


def load_pt_dataset(pt_file):
    data_dict = torch.load(pt_file)
    images = data_dict['images']
    labels = data_dict['labels']
    return TensorDataset(images, labels)


def download_temp(bucket, key):
    import tempfile
    local_file = tempfile.NamedTemporaryFile(delete=False)
    local_file.close()
    s3.download_file(bucket, key, local_file.name)
    return local_file.name


def download_from_s3(s3_uri, local_path):
    # parse s3://bucket/key
    assert s3_uri.startswith("s3://")
    no_prefix = s3_uri[len("s3://"):]
    bucket, key = no_prefix.split('/', 1)
    s3.download_file(bucket, key, local_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=22)
