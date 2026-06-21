import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  
from dataset import LettuceDetectionDataset, collate_fn
from models import get_teacher_model

def train_teacher():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Teacher on: {device}")
    
    DATA_DIR = "/content/dataset/train"
    ANNOTATION_FILE = "/content/dataset/train/_annotations.coco.json"
    
    with open(ANNOTATION_FILE, 'r') as f:
        coco_data = json.load(f)
        NUM_CLASSES = max([cat['id'] for cat in coco_data['categories']]) + 1
        print(f"Dynamically detected {NUM_CLASSES} total classes.")
    
    BATCH_SIZE = 8     
    
    # --- SESSION RECOVERY CONTROLS ---
    TOTAL_EPOCHS = 20
    RESUME_TRAINING = True  # Change to True if you need to resume after a crash
    START_EPOCH = 20      # Change to the next epoch number (e.g., if epoch 5 finished, set to 6)
    
    CHECKPOINT_DIR = "/content/drive/MyDrive/EXCESS/lettuce_project/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    dataset = LettuceDetectionDataset(root_dir=DATA_DIR, annotation_file=ANNOTATION_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=0, collate_fn=collate_fn)
    
    # Check if we are resuming from a previous crash
    if RESUME_TRAINING:
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, f"teacher_resnet101_epoch_{START_EPOCH-1}.pth")
        print(f"Resuming! Loading weights from: {latest_checkpoint}")
        teacher = get_teacher_model(NUM_CLASSES, pretrained_weights_path=latest_checkpoint).to(device)
    else:
        teacher = get_teacher_model(NUM_CLASSES, pretrained_weights_path=None).to(device)

    teacher.train()
    for param in teacher.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-4, weight_decay=1e-4)

    # Start the loop from the correct epoch
    for epoch in range(START_EPOCH - 1, TOTAL_EPOCHS):
        epoch_loss = 0.0
        
        loop = tqdm(dataloader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}]")
        
        for batch_idx, (images, targets) in enumerate(loop):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = teacher(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            cls_loss = loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
            box_loss = loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
            obj_loss = loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
            
            loop.set_postfix(Total=f"{losses.item():.3f}", Cls=f"{cls_loss:.3f}", Box=f"{box_loss:.3f}", Obj=f"{obj_loss:.3f}")
            
        print(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}] Final Average Loss: {epoch_loss/len(dataloader):.4f}\n")
        
        # --- PER-EPOCH SAVING ---
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"teacher_resnet101_epoch_{epoch+1}.pth")
        torch.save(teacher.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

if __name__ == "__main__":
    train_teacher()