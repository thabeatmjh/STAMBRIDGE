import os
import re
import csv
import math
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 假设这些是你本地的依赖文件
from newmodel import RouteModel
from eegdatasets_leaveone import EEGDataset
from util import wandb_logger
from loss import ClipLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 工具函数
# ==========================================
def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group()) - 1
    return 0

def make_class_prototypes(img_feats, samples_per_class=10):
    """
    提取每个类别的原型 (Prototypes)，保证类别对应的 Image 特征唯一且标准化。
    """
    N, D = img_feats.shape
    assert N % samples_per_class == 0, f"img_feats length {N} not divisible by {samples_per_class}"
    n_classes = N // samples_per_class
    proto = img_feats.view(n_classes, samples_per_class, D).mean(dim=1)
    proto = F.normalize(proto, dim=-1)
    return proto

# ==========================================
# 训练循环 (Train Loop)
# ==========================================
def train_model(sub, model, loader, optimizer, device, config, epoch):
    model.train()
    loss_fn = ClipLoss()
    total_loss = 0.0
    batch_cnt = 0

    for eeg, labels, txt, img in loader:
        eeg = eeg.to(device)
        labels = labels.to(device)
        txt = txt.to(device).float()
        img = img.to(device).float()

        batch_size = eeg.size(0)
        subject_id = extract_id_from_string(sub)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)

        optimizer.zero_grad()
        
        # 前向传播 (严格匹配精简后的 RouteModel 返回值)
        z_eeg1, (eeg_img, eeg_text, eeg_dep), semantic_logits, \
        _, _, _, _, _, _, _, _, f1, f2, _ = model(eeg, subject_ids, img, txt)

        # 1. Main Loss (InfoNCE 直接对齐)
        l_img_main = loss_fn(z_eeg1, img, model.logit_scale_img)
        
        # 2. Bridge Loss (构建正则化的中间特征目标)
        img_drop = F.dropout(img.detach(), p=0.4, training=True)
        loss_bridge = loss_fn(f2, img_drop, model.logit_scale_img)

        # 3. Distillation Loss (分阶段蒸馏，辅助头向脱离计算图的桥接目标学习)
        loss_distill = loss_fn(eeg_img, f2.detach(), model.logit_scale_img)

        # 动态权重调度 (Staged Distillation Strategy)
        lambda_0 = 0.99
        lambda_1 = 0.5
        lambda_2 = 0.2 * min(1.0, (epoch - 5) / 15.0) if epoch >= 5 else 0.0

        total_loss_batch = (
            lambda_0 * l_img_main +
            lambda_1 * loss_bridge +
            lambda_2 * loss_distill
        )

        total_loss_batch.backward()
        optimizer.step()

        # 限制 logit_scale 防止梯度爆炸
        with torch.no_grad():
            model.logit_scale_img.clamp_(math.log(1.0), math.log(3.0))

        total_loss += total_loss_batch.item()
        batch_cnt += 1

    return total_loss / batch_cnt, 0

# ==========================================
# 评估循环 (Evaluation Loop)
# ==========================================
def evaluate_model(sub, model, dataloader, device, img_features_all, epoch, k=200):
    model.eval()
    img_features_all = img_features_all.to(device).float()
    
    correct = 0
    top5_correct = 0
    total = 0
    all_labels = set(range(img_features_all.size(0)))

    save_path = './results/features'
    os.makedirs(save_path, exist_ok=True)

    all_eeg_features = []
    all_top5_indices = []
    
    with torch.no_grad():
        for eeg_data, labels, txt, img in dataloader:   
            eeg_data = eeg_data.to(device)
            labels = labels.to(device)
            txt = txt.to(device).float()
            img = img.to(device).float()
            
            batch_size = eeg_data.size(0) 
            subject_id = extract_id_from_string(sub)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)       
            
            # 仅取我们需要的主特征 (z_eeg1) 用于检索测试
            z_eeg1, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(eeg_data, subject_ids, img, txt)
            all_eeg_features.append(z_eeg1.cpu().numpy())
            
            logit_scale = model.logit_scale_img.exp().item()

            for idx, label in enumerate(labels):
                gt_label = label.item()
                possible_classes = list(all_labels - {gt_label})
                
                # 随机抽取 k-1 个干扰类，与真实类组成候选池
                selected_classes = random.sample(possible_classes, k - 1) + [gt_label]
                selected_img_features = img_features_all[selected_classes]
                
                # 计算余弦相似度分数
                logits_img = logit_scale * (z_eeg1[idx] @ selected_img_features.T)
                
                # Top-1 计算
                predicted_label = selected_classes[torch.argmax(logits_img).item()]
                if predicted_label == gt_label:
                    correct += 1
                    
                # Top-5 计算
                if k >= 5:
                    _, top5_indices = torch.topk(logits_img, 5, largest=True)
                    top5_classes = [selected_classes[i] for i in top5_indices.tolist()]
                    all_top5_indices.append(top5_classes)
                    if gt_label in top5_classes:                
                        top5_correct += 1                            
                total += 1

    # 仅在评估 200-way (全局) 时保存特征
    if k == 200:
        all_eeg_features = np.vstack(all_eeg_features)  
        np.save(os.path.join(save_path, f'eeg_features_{sub}_epoch{epoch}.npy'), all_eeg_features) 
        top5_df = pd.DataFrame(all_top5_indices, columns=[f'Top5_Idx_{i+1}' for i in range(5)])  
        top5_df.to_csv(os.path.join(save_path, f'top5_indices_{sub}_epoch{epoch}.csv'), index=False) 
    
    accuracy = correct / total
    top5_acc = (top5_correct / total) if k >= 5 else 0.0
    return 0.0, accuracy, top5_acc

# ==========================================
# 主控制流 (Main Loop)
# ==========================================
def main_train_loop(sub, current_time, model, train_loader, test_loader, optimizer, device, 
                    img_features_train_all, img_features_test_all, config, scheduler, logger=None):
    if logger:
        logger.watch(model, logger)
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs, v4_accs, v10_accs = [], [], []
    
    best_accuracy = 0.0
    best_model_path = None 
    results = []
    
    # 提取图像原型供对比
    img_bank_train = make_class_prototypes(img_features_train_all, samples_per_class=10).to(device)
    img_bank_test = make_class_prototypes(img_features_test_all, samples_per_class=10).to(device)

    for epoch in range(config.epochs):
        train_loss, train_accuracy = train_model(sub, model, train_loader, optimizer, device, config, epoch)
        if scheduler:
            scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 执行多尺度 K-way 检索评估
        _, test_accuracy, top5_acc = evaluate_model(sub, model, test_loader, device, img_bank_test, epoch, k=200)
        _, v2_acc, _ = evaluate_model(sub, model, test_loader, device, img_bank_test, epoch, k=2)
        _, v4_acc, _ = evaluate_model(sub, model, test_loader, device, img_bank_test, epoch, k=4)
        _, v10_acc, _ = evaluate_model(sub, model, test_loader, device, img_bank_test, epoch, k=10)
        _, v50_acc, v50_top5_acc = evaluate_model(sub, model, test_loader, device, img_bank_test, epoch, k=50)
        _, v100_acc, v100_top5_acc = evaluate_model(sub, model, test_loader, device, img_bank_test, epoch, k=100)
        
        test_losses.append(0.0)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)

        epoch_results = {
            "epoch": epoch + 1, "test_loss": 0.0, "test_accuracy": test_accuracy,
            "v2_acc": v2_acc, "v4_acc": v4_acc, "v10_acc": v10_acc,
            "top5_acc": top5_acc, "v50_acc": v50_acc, "v100_acc": v100_acc,
            "v50_top5_acc": v50_top5_acc, "v100_top5_acc": v100_top5_acc
        }
        results.append(epoch_results)

        # ==================== 最佳模型保存逻辑 ====================
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            
            mode_str = 'in' if config.insubject else 'across'
            model_save_dir = f"./models/contrast/{mode_str}/{config.encoder_type}/{sub}/{current_time}"
            os.makedirs(model_save_dir, exist_ok=True)             
            
            file_name = f"epoch{epoch+1}_{sub}_acc{test_accuracy:.4f}.pth"
            new_file_path = os.path.join(model_save_dir, file_name)
            
            torch.save(model.state_dict(), new_file_path)
            print(f"--> New Best Model saved to {new_file_path}!")
            
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = new_file_path

        if logger:
            logger.log(epoch_results)
            
        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Test (200-way) Acc: {test_accuracy:.4f}, Top-5 Acc: {top5_acc:.4f}")
        print(f"Sub-metrics -> v2: {v2_acc:.4f} | v4: {v4_acc:.4f} | v10: {v10_acc:.4f} | v50: {v50_acc:.4f} | v100: {v100_acc:.4f}\n")

    # 可视化训练曲线
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")
    axs[0, 1].plot(test_accuracies, label='200-way Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")
    axs[1, 0].plot(v2_accs, label='2-class Accuracy'); axs[1, 0].legend(); axs[1, 0].set_title("2-Class Accuracy Curve")
    axs[1, 1].plot(v4_accs, label='4-class Accuracy'); axs[1, 1].legend(); axs[1, 1].set_title("4-Class Accuracy Curve")
    axs[2, 0].plot(v10_accs, label='10-class Accuracy'); axs[2, 0].legend(); axs[2, 0].set_title("10-Class Accuracy Curve")
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.suptitle('STAMBRIDGE Evaluation Metrics', fontsize=16, y=1.02)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/metrics_{sub}_{current_time}.png')
    
    if logger:
        logger.finish()
    return results

def main():
    parser = argparse.ArgumentParser(description='EEG STAMBRIDGE Training Script')
    parser.add_argument('--data_path', type=str, default="/root/autodl-tmp/EEG2Vision/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save output results')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='NeuralMCRL', help='Encoder type')
    parser.add_argument('--subjects', nargs='+', default=['sub-08'], help='List of subject IDs')
    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    all_subjects = [f'sub-{i:02d}' for i in range(1, 11)]

    for sub in args.subjects:
        model = RouteModel(
            sequence_length=250,  
            num_subjects=10, 
            embedding_dim=1024, 
            proj_dim=1024 
        ).to(device)

        # Optimizer Setup (投影头与主干分离的学习率)
        base_params = [p for n, p in model.named_parameters() if not any(k in n for k in ['proj_eeg2', 'proj_eeg3'])]
        proj_params = [p for n, p in model.named_parameters() if any(k in n for k in ['proj_eeg2', 'proj_eeg3'])]

        optimizer = AdamW([
            {'params': base_params, 'lr': args.lr},
            {'params': proj_params, 'lr': 3e-4}
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Dataset Setup
        if args.insubject:
            train_dataset = EEGDataset(args.data_path, subjects=[sub], train=True)
            test_dataset = EEGDataset(args.data_path, subjects=[sub], train=False)
        else:
            train_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=all_subjects, train=True)
            test_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=all_subjects, train=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

        # Start Training
        results = main_train_loop(
            sub=sub,
            current_time=current_time,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            img_features_train_all=train_dataset.img_features,
            img_features_test_all=test_dataset.img_features,
            config=args,
            scheduler=scheduler,
            logger=wandb_logger(args) if args.logger else None
        )
        
        # Save CSV Results
        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)
        mode_str = 'in' if args.insubject else 'cross_exclude'
        results_file = os.path.join(results_dir, f"{args.encoder_type}_{mode_str}_{sub}.csv")

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')

if __name__ == '__main__':
    main()
