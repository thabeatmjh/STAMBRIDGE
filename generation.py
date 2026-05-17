
from diffusion_prior import *
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from custom_pipeline import Generator4Embeds
import gc
train = False
classes = None
pictures= None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_data():
    data_list = []
    label_list = []
    texts = []
    images = []
    
    if train:
        text_directory = "/root/autodl-tmp/EEG2Vision/Image_set/training_images"  
    else:
        text_directory = "/root/autodl-tmp/EEG2Vision/Image_set/test_images"

    dirnames = [d for d in os.listdir(text_directory) if os.path.isdir(os.path.join(text_directory, d))]
    dirnames.sort()
    
    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    for dir in dirnames:

        try:
            idx = dir.index('_')
            description = dir[idx+1:]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue
            
        new_description = f"{description}"
        texts.append(new_description)

    if train:
        img_directory = "/root/autodl-tmp/EEG2Vision/Image_set/training_images" 
    else:
        img_directory ="/root/autodl-tmp/EEG2Vision/Image_set/test_images"
    
    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()

    if classes is not None and pictures is not None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            pic_idx = pictures[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                if pic_idx < len(all_images):
                    images.append(os.path.join(folder_path, all_images[pic_idx]))
    elif classes is not None and pictures is None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
    elif classes is None:
        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()  
            images.extend(os.path.join(folder_path, img) for img in all_images)
    else:

        print("Error")
    return texts, images
texts, images = load_data()
# eeg_model = RouteModel()
# eeg_model.load_state_dict(torch.load("models/contrast/ATMS/02-01_00-39/sub-08/40.pth"))
# eeg_model = eeg_model.to(device)
# sub = 'sub-08'
subjects = [ 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
for sub in subjects:
    print(f"当前正在处理: {sub}")
    emb_eeg = torch.load(f'/root/autodl-tmp/new/newmodel_eeg_features_{sub}_train.pt')
    emb_eeg = F.normalize(emb_eeg, dim=1)
    saved_features = torch.load('/root/autodl-tmp/new/ViT-H-14_features_train.pt')
    emb_img_train = saved_features['img_features']
    emb_img_train_4 = emb_img_train.view(1654,10,1,1024).repeat(1,1,4,1).view(-1,1024)
    emb_eeg_test = torch.load(f'/root/autodl-tmp/new/newmodel_eeg_features_{sub}_test.pt')
    print(emb_eeg.shape, emb_img_train_4.shape)
    print(torch.norm(emb_eeg, dim=1).mean(), torch.norm(emb_img_train_4, dim=1).mean())
    print(F.cosine_similarity(emb_eeg[:10], emb_img_train_4[:10], dim=1))
    eeg = F.normalize(emb_eeg, dim=1)
    img = F.normalize(emb_img_train_4, dim=1)
    
    # 正配对相似度
    pos = F.cosine_similarity(eeg, img, dim=1).mean().item()
    
    # 随机错配相似度
    perm = torch.randperm(img.size(0))
    neg = F.cosine_similarity(eeg, img[perm], dim=1).mean().item()
    
    print("pos:", pos)
    print("neg:", neg)
    dataset = EmbeddingDataset(
        c_embeddings=emb_eeg, h_embeddings=emb_img_train_4, 
        # h_embeds_uncond=h_embeds_imgnet
    )
    
    dl = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=64)
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    # number of parameters
    print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
    pipe = Pipe(diffusion_prior, device=device)
    
    # load pretrained model
    model_name = 'diffusion_prior' # 'diffusion_prior_vice_pre_imagenet' or 'diffusion_prior_vice_pre'
    pipe.train(dl, num_epochs=150, learning_rate=1e-3) # to 0.142 
    # pipe.diffusion_prior.load_state_dict(torch.load(f'./fintune_ckpts/{config['encoder_type']}/{sub}/{model_name}.pt', map_location=device))
    save_path = f'./fintune_ckpts/newmodel/{sub}/{model_name}.pt'
    
    directory = os.path.dirname(save_path)
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    torch.save(pipe.diffusion_prior.state_dict(), save_path)
    from PIL import Image
    import os
    
    # Assuming generator.generate returns a PIL Image
    generator = Generator4Embeds(num_inference_steps=4, device=device)
    
    directory = f"generated_imgs/{sub}"
    for k in range(200):
        eeg_embeds = emb_eeg_test[k:k+1]
        h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=5.0)
        for j in range(10):
            image = generator.generate(h.to(dtype=torch.float16))
            # Construct the save path for each image
            path = f'{directory}/{texts[k]}/{j}.png'
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Save the PIL Image
            image.save(path)
            print(f'Image saved to {path}')
    del emb_eeg, emb_eeg_test, emb_img_train, emb_img_train_4, dataset, dl
    del diffusion_prior, pipe, generator, h
    gc.collect()
    torch.cuda.empty_cache()
