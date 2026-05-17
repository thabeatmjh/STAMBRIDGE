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

from STAMEncoder import STAMEncoder
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

def train_model(sub, eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, config, epoch):
    eeg_model.train()
    loss_fn = ClipLoss()
    text_features_all = text_features_all.to(device).float() # (n_cls, d)
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    save_features= True
    for batch_idx, (eeg_data, labels, text_features, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        batch_size = eeg_data.size(0)  
        subject_id = extract_id_from_string(sub)
        # eeg_data = eeg_data.permute(0, 2, 1)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
        # if not config.insubject:
        #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)     
        z_eeg,eeg_features = eeg_model(eeg_data, subject_ids)
        logit_scale = eeg_model.logit_scale
        l_img_main = loss_fn(eeg_features, img_features, logit_scale)
        f_mid_img2,loss_bridge,loss_distill = eeg_model.bridge_plugin(z_eeg, img_features,text_features, 0.4, logit_scale)
        lambda_0 = 0.99
        lambda_1 = 0.5
        lambda_2 = 0.2 * min(1.0, (epoch - 5) / 15.0) if epoch >= 5 else 0.0

        loss = (
            lambda_0 * l_img_main +
            lambda_1 * loss_bridge +
            lambda_2 * loss_distill
        )

        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            # 推荐范围：exp(logit_scale) 在 1 ~ 20 之间（可根据情况放宽）
            logit_scale.clamp_(math.log(1.0), math.log(3.0))
        # Compute the corresponding logits
        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy



def evaluate_model(sub, eeg_model, dataloader, device, text_features_all, img_features_all, k, config):
    eeg_model.eval()
    loss_fn = ClipLoss()   
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    top5_correct = 0
    top5_correct_count = 0
    # Get all unique classes
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text_features, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = eeg_data.size(0) 
            subject_id = extract_id_from_string(sub)
            # eeg_data = eeg_data.permute(0, 2, 1)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            # if not config.insubject:
            #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)          
            z_eeg,eeg_features = eeg_model(eeg_data, subject_ids)

        
            logit_scale = eeg_model.logit_scale
            l_img_main = loss_fn(eeg_features, img_features, logit_scale)
            f_mid_img2,loss_bridge,loss_distill = eeg_model.bridge_plugin(z_eeg, img_features,text_features, 0.4, logit_scale)
            lambda_0 = 0.99
            lambda_1 = 0.5
            lambda_2 = 0.2 * min(1.0, (epoch - 5) / 15.0) if epoch >= 5 else 0.0
    
            loss = (
                lambda_0 * l_img_main +
                lambda_1 * loss_bridge +
                lambda_2 * loss_distill
            )

            
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                # First, select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                selected_text_features = text_features_all[selected_classes]
                
                if k==200:
                    # Compute the corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get the predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        # print("predicted_label", predicted_label)
                        correct += 1
                    
                    # logits_single is the model's output, shape (n_batch, n_classes)
                    # label is the true label, shape (n_batch,)
                    # Get the indices of the top-5 predictions
                    # print("logits_single", logits_single)
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                   
                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k == 50 or k == 100:
                    # For k=50 or 100, select k classes for evaluation
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]

                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                   
                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k==2 or k==4 or k==10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    # Compute the corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                    # logits_single = (logits_text + logits_img) / 2.0
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get the predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
            del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc

def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device, text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model,logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    
    for epoch in range(config.epochs):
        # Train the model
        train_loss, train_accuracy = train_model(sub, eeg_model, train_dataloader, optimizer, device, text_features_train_all, img_features_train_all, config=config,epoch=epoch)
        if (epoch +1) % 5 == 0:                    
            # Get the current time and format it as a string (e.g., '2024-01-17_15-30-00')                  
            if config.insubject==True:       
                os.makedirs(f"./models/contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)             
                file_path = f"./models/contrast/{config.encoder_type}/{sub}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)            
            else:                
                os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}", exist_ok=True)             
                file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        # Evaluate the model
        test_loss, test_accuracy, top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all,k=200, config=config)
        _, v2_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 2, config=config)
        _, v4_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 4, config=config)
        _, v10_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 10, config=config)
        _, v50_acc, v50_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all,  k=50, config=config)
        _, v100_acc, v100_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all,  k=100, config=config)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        # "train_loss": train_loss,
        # "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc":top5_acc,
        "v50_acc": v50_acc,
        "v100_acc": v100_acc,
        "v50_top5_acc":v50_top5_acc,
        "v100_top5_acc": v100_top5_acc
        }

        results.append(epoch_results)
        # If the test accuracy of the current epoch is the best, save the model and related information
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # best_model_weights = model.state_dict().copy()
            
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc":v2_acc,
                "v4_acc":v4_acc,
                "v10_acc":v10_acc
            }
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "v2 Accuracy": v2_acc,
            "v4 Accuracy": v4_acc,
            "v10 Accuracy": v10_acc,
            "Epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
  
    # # Load the best model weights
    # model.load_state_dict(best_model_weights)

    # # # Save the best model
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')

    # Create 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # The following are the three new plots you added, assuming you've already calculated the corresponding accuracies
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information for annotation
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    # Add main title
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()
    return results

import datetime

def main():
    # Use argparse to parse the command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str, default="/root/autodl-tmp/EEG2Vision/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')    
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", help='WandB project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')    
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='STAMEncoder', help='Encoder type')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')    
    args = parser.parse_args()

    # Set device based on the argument
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')
      
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in args.subjects:
        eeg_model = globals()[args.encoder_type]()
        eeg_model.to(device)

        optimizer = AdamW(eeg_model.parameters(), lr=args.lr, weight_decay=1e-4)

        if args.insubject:
            train_dataset = EEGDataset(args.data_path, subjects=[sub], train=True)
            test_dataset = EEGDataset(args.data_path, subjects=[sub], train=False)
        else:
            train_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=True)
            test_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(sub, current_time, eeg_model, train_loader, test_loader, optimizer, device, 
                                  text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config=args, logger=args.logger)


        # Save results to a CSV file
        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)

        if args.insubject:
            results_file = f"{results_dir}/{args.encoder_type}_{sub}.csv"
        else:
            results_file = f"{results_dir}/{args.encoder_type}_cross_exclude_{sub}.csv"

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')

                
if __name__ == '__main__':
    main()
