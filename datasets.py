import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import random
import pickle
import gc  # 🚨 导入垃圾回收模块，防止 Killed

proxy = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

import open_clip

model_path = '/root/autodl-tmp/EEG2Vision/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.safetensors'
model_config_path = '/root/autodl-tmp/EEG2Vision/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_config.json'
model_type = 'ViT-H-14'

vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type,
    pretrained=model_path,
    precision='fp32',
    device=device
)

import json

config_path = "data_config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

data_path = config["data_path"]
img_directory_training = config["img_directory_training"]
img_directory_test = config["img_directory_test"]
features_path = config["features_path"]


class EEGDataset(Dataset):
    """
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """

    def __init__(self, data_path, exclude_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes=None,
                 pictures=None, val_size=None):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject
        self.val_size = val_size
        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)

        self.data, self.labels, self.text, self.img = self.load_data()

        self.data = self.extract_eeg(self.data, time_window)

        if self.classes is None and self.pictures is None:

            text_features_filename = os.path.join(
                f'{model_type}_text_features_train.pt') if self.train else os.path.join(
                f'{model_type}_text_features_test.pt')  # 文本特征文件名
            img_features_filename = os.path.join(
                f'{model_type}_img_features_train.pt') if self.train else os.path.join(
                f'{model_type}_img_features_test.pt')  # 图像特征文件名
            depth_features_filename = os.path.join(
                f'{model_type}_depth_features_train.pt') if self.train else os.path.join(
                f'{model_type}_depth_features_test.pt')  # 深度图像特征文件名

            if os.path.exists(text_features_filename):
                self.text_features = torch.load(text_features_filename)['text_features']
            else:
                self.text_features = self.Textencoder(self.text)
                torch.save({'text_features': self.text_features.cpu()}, text_features_filename)

            if os.path.exists(img_features_filename):
                self.img_features = torch.load(img_features_filename)['img_features']
            else:
                self.img_features = self.ImageEncoder(self.img)
                torch.save({'img_features': self.img_features.cpu()}, img_features_filename)  # 保存图像特征

            if os.path.exists(depth_features_filename):
                self.depth_features = torch.load(depth_features_filename)['depth_features']
            else:
                raise FileNotFoundError(f"{depth_features_filename} not found.")

        else:
            self.text_features = self.Textencoder(self.text)
            self.img_features = self.ImageEncoder(self.img)

    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []

        if self.train:
            text_file_path = os.path.join('/root/autodl-tmp/EEG2Vision/Retrieval/texts/this_is_a_description_texts',
                                          'img_to_text_training.pkl')
        else:
            text_file_path = os.path.join('/root/autodl-tmp/EEG2Vision/Retrieval/texts/this_is_a_description_texts',
                                          'img_to_text_test.pkl')

        if os.path.exists(text_file_path):
            with open(text_file_path, 'rb') as f:
                texts = pickle.load(f)
        else:
            print(f"Warning: {text_file_path} not found. No text descriptions loaded.")

        if self.train:
            directory = img_directory_training
            img_directory = img_directory_training
        else:
            directory = img_directory_test
            img_directory = img_directory_test

        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()

        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()

        if self.classes is not None and self.pictures is not None:
            images = []
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if
                                  img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            images = []
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if
                                  img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            images = []
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
        else:
            print("Error")

        print("self.subjects", self.subjects)
        print("exclude_subject", self.exclude_subject)
        
        for subject in self.subjects:
            if self.train:
                if subject == self.exclude_subject:
                    continue
                file_name = 'preprocessed_eeg_training.npy'

                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)

                # 🚨 使用 .clone() 避免内存引用原巨大数组
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach().clone()
                times = torch.from_numpy(data['times']).detach()[50:].clone()
                ch_names = data['ch_names']

                n_classes = 1654
                samples_per_class = 10

                if self.classes is not None and self.pictures is not None:
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * 1 + p
                        if start_index < len(preprocessed_eeg_data):
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + 1].clone()
                            labels = torch.full((1,), c, dtype=torch.long).detach()
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)

                elif self.classes is not None and self.pictures is None:
                    for c in self.classes:
                        start_index = c * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[
                                                      start_index: start_index + samples_per_class].clone()
                        labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                else:
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[
                                                      start_index: start_index + samples_per_class].clone()
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

            else:
                if subject == self.exclude_subject or self.exclude_subject == None:
                    file_name = 'preprocessed_eeg_test.npy'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = np.load(file_path, allow_pickle=True)
                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach().clone()
                    times = torch.from_numpy(data['times']).detach()[50:].clone()
                    ch_names = data['ch_names']
                    n_classes = 200

                    samples_per_class = 1

                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:  
                            continue
                        start_index = i * samples_per_class  
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index + samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach() 
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0).clone()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)  
                else:
                    continue
            
            # ==========================================
            # 🚨 核心防 Killed 修改：释放当前 Subject 的庞大内存！
            # ==========================================
            del preprocessed_eeg_data
            del data
            gc.collect()
            print(f"[Memory Safe] {subject} loaded and raw cache cleared.")

        # ==========================================
        # 🚨 终极防 Killed 修改：边出栈边拼接 (Pop & Free)
        # ==========================================
        print(f"开始拼接 {len(data_list)} 个数据块，进入零峰值低内存模式...")
        
        # 1. 提取用于 view 的 shape 模板
        shape_template = data_list[0].shape
        
        # 2. 预先计算总长度并分配【空内存】(不产生复制峰值)
        total_data_len = sum(t.shape[0] for t in data_list)
        data_tensor = torch.empty((total_data_len, *shape_template[1:]), dtype=data_list[0].dtype)
        
        # 3. 边出栈边赋值，保证内存里只有一份完整数据！
        current_idx = 0
        while len(data_list) > 0:
            # pop(0) 会将元素从 list 中永远移除，内存立刻被释放！
            t = data_list.pop(0)  
            size = t.shape[0]
            data_tensor[current_idx : current_idx + size] = t
            current_idx += size
            
        # 彻底清空残留
        gc.collect()
        
        # 兼容你原本的维度重塑逻辑
        if self.train:
            data_tensor = data_tensor.view(-1, *shape_template[2:])
        else:
            data_tensor = data_tensor.view(-1, *shape_template)
            
        print("数据拼接完成！开始处理标签...")
        
        # 对 label 也做同样的低内存拼接
        total_label_len = sum(t.shape[0] for t in label_list)
        label_tensor = torch.empty((total_label_len, *label_list[0].shape[1:]), dtype=label_list[0].dtype)
        
        current_idx = 0
        while len(label_list) > 0:
            t = label_list.pop(0)
            size = t.shape[0]
            label_tensor[current_idx : current_idx + size] = t
            current_idx += size
            
        gc.collect()

        if self.train:
            label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                unique_values = torch.tensor(lis)
                mapping = {val.item(): index for index, val in enumerate(unique_values)}
                label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
        else:
            pass

        self.times = times
        self.ch_names = ch_names

        print(f"✅ Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")

        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)
        extracted_data = eeg_data[..., indices]

        return extracted_data

    def Textencoder(self, text):
        batch_size = 32
        text_features_list = []

        for i in range(0, len(text), batch_size):
            batch_texts = text[i:i + batch_size]
            text_inputs = torch.cat([open_clip.tokenize(t) for t in batch_texts]).to(device)

            with torch.no_grad():
                text_features = vlmodel.encode_text(text_inputs)

            text_features = F.normalize(text_features, dim=-1).detach()
            text_features_list.append(text_features.cpu())

        all_text_features = torch.cat(text_features_list, dim=0)

        if self.train:
            reshaped_features = all_text_features.view(-1, 10, 1024)
            aggregated_features = reshaped_features.mean(dim=1)
            final_features = aggregated_features
        else:
            final_features = all_text_features.view(-1, 1024)

        print(f"Text features shape: {final_features.shape}")

        return final_features

    def ImageEncoder(self, images):
        batch_size = 20
        image_features_list = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(
                device)

            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

            image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)

        return image_features

    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        x = self.data[index]
        label = self.labels[index]

        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes) * 1 * 80
                index_n_sub_train = len(self.classes) * 10 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (10 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes) * 1 * 80
                index_n_sub_train = len(self.classes) * 1 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)

        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]
        depth_features = self.depth_features[img_index]
        
        return x, label, text_features, img_features, depth_features, img_index

    def __len__(self):
        return self.data.shape[0]  


if __name__ == "__main__":
    data_path = data_path
    train_dataset = EEGDataset(data_path, subjects=['sub-01'], train=True)
    test_dataset = EEGDataset(data_path, subjects=['sub-01'], train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    i = 80 * 1 - 1
    x, label, text_features, img_features, depth_features, img_index = test_dataset[i]
    print(f"Index {i}, Label: {label}")