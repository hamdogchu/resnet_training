import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

# ==========================================
# 1. NETWORK DEFINITIONS
# ==========================================

class TeacherResNet101(nn.Module):
    def __init__(self, num_classes=80, num_bbox_coords=4):
        super().__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # Extract layers up to Conv5 (layer4 in PyTorch)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # HTKD mentions a 'Detection Head' outputting Classifier Logits and Regressor Logits
        self.classifier = nn.Linear(2048, num_classes)
        self.regressor = nn.Linear(2048, num_bbox_coords)

    def forward(self, x):
        f_map = self.features(x)         # F_T: Feature map from Conv5
        pooled = self.pool(f_map).flatten(1)
        z_cls = self.classifier(pooled)  # z_T: Classification logits
        z_reg = self.regressor(pooled)   # Bounding box outputs
        return f_map, z_cls, z_reg

class StudentResNet18(nn.Module):
    def __init__(self, num_classes=80, num_bbox_coords=4):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Adaptation layer to match Teacher's 2048 dimensions for Feature KD Loss
        self.adaptation_layer = nn.Conv2d(512, 2048, kernel_size=1)
        
        self.classifier = nn.Linear(2048, num_classes)
        self.regressor = nn.Linear(2048, num_bbox_coords)

    def forward(self, x):
        f_map_raw = self.features(x)
        f_map_adapted = self.adaptation_layer(f_map_raw) # F_S: Adapted to 2048 channels
        
        pooled = self.pool(f_map_adapted).flatten(1)
        z_cls = self.classifier(pooled)
        z_reg = self.regressor(pooled)
        return f_map_adapted, z_cls, z_reg

# ==========================================
# 2. GAUSSIAN PROCESS UNCERTAINTY ESTIMATOR
# ==========================================

class MiniBatchGP(nn.Module):
    def __init__(self):
        super().__init__()
        # Trainable GP hyperparameters initialized
        self.log_l = nn.Parameter(torch.tensor(0.0))       # Length scale l
        self.log_sigma_f = nn.Parameter(torch.tensor(0.0)) # Signal variance
        self.log_sigma_n = nn.Parameter(torch.tensor(-2.0))# Noise variance (small initial)

    def compute_rbf_kernel(self, u):
        """
        Computes K[u(x_i), u(x_j)] 
        Equation: K = sigma_f^2 * exp(- ||u_i - u_j||^2 / 2l^2)
        """
        l = torch.exp(self.log_l)
        sigma_f_sq = torch.exp(self.log_sigma_f)**2
        
        # Calculate pairwise squared distances
        dist_sq = torch.cdist(u, u, p=2)**2
        K = sigma_f_sq * torch.exp(-dist_sq / (2 * l**2))
        return K

    def forward(self, F_T, z_T, targets):
        """
        F_T: Teacher features (B, 2048, H, W)
        z_T: Teacher logits (B, C)
        targets: Ground truth labels (used for optimizing GP via NLML)
        """
        B = z_T.size(0)
        
        # 1. Concatenate inputs: u_T(x_i) = concat(F_T, z_T)
        F_T_pooled = F_T.mean(dim=[2, 3]) # Global Average Pool to flatten
        u_T = torch.cat([F_T_pooled, z_T], dim=1) # Shape: (B, 2048 + C)
        
        # 2. Compute RBF Kernel & Covariance Matrix
        K = self.compute_rbf_kernel(u_T)
        sigma_n_sq = torch.exp(self.log_sigma_n)**2
        K_noisy = K + sigma_n_sq * torch.eye(B, device=u_T.device)
        
        # 3. Calculate Variance of Teacher (sigma^2_T)
        # Using diagonal of the covariance matrix for the current batch
        # sigma^2_T(x_*) = sigma_f^2 [k(x*, x*) - k_*X (K + sigma_n^2 I)^-1 k_*X]
        K_inv = torch.linalg.inv(K_noisy)
        
        # For in-batch estimation, the variance simplifies down to the self-covariance 
        # minus the explained variance. 
        sigma_f_sq = torch.exp(self.log_sigma_f)**2
        
        # Using diag elements for individual image uncertainties
        k_star_star = sigma_f_sq * torch.ones(B, device=u_T.device)
        
        # Explained variance per sample
        explained_var = torch.einsum('bi,ij,jb->b', K, K_inv, K)
        sigma_T_sq = k_star_star - explained_var

        # 4. Calculate NLML (Negative Log Marginal Likelihood) to train GP params
        # NLML = 0.5 * y^T (K + sigma_n^2 I)^-1 y + 0.5 * log|K + sigma_n^2 I| + const
        y = targets.float().view(-1, 1) # Ensure correct shape for matmul
        
        term1 = 0.5 * torch.matmul(torch.matmul(y.T, K_inv), y)
        term2 = 0.5 * torch.logdet(K_noisy)
        nlml = (term1 + term2).squeeze()

        # Ensure uncertainty is positive and detached (student shouldn't backprop into GP)
        sigma_T_sq = torch.clamp(sigma_T_sq.detach(), min=1e-6) 
        
        return sigma_T_sq, nlml

# ==========================================
# 3. THE TRAINING LOOP ALGORITHM
# ==========================================

def train_htkd(dataloader, num_epochs=50, device='cuda'):
    # A. Initialize Parameters
    teacher = TeacherResNet101().to(device)
    student = StudentResNet18().to(device)
    gp_module = MiniBatchGP().to(device)

    # Freeze teacher parameters (Step B implies Teacher is already trained)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Optimizers
    student_optimizer = optim.Adam(student.parameters(), lr=1e-4)
    gp_optimizer = optim.Adam(gp_module.parameters(), lr=1e-3)

    # Hyperparameters from PDF
    alpha = 1.0   # Weight for UAKD Feature Distillation Loss
    beta = 1.0    # Weight for Classification Loss
    gamma = 1.0   # Weight for Bounding Box Regression Loss
    epsilon = 1e-6 # Stability constant for UAKD denominator

    student.train()

    # D. Training Loop with Gaussian Processes (GPs)
    for epoch in range(num_epochs):
        for batch_idx, (images, labels, bboxes) in enumerate(dataloader):
            images, labels, bboxes = images.to(device), labels.to(device), bboxes.to(device)
            B = images.size(0)

            # 2.1 Forward Pass
            with torch.no_grad():
                F_T, z_T_cls, z_T_reg = teacher(images)
            
            F_S, z_S_cls, z_S_reg = student(images)

            # 2.2 Uncertainty Estimation using GPs
            # Pass ground truth labels to optimize the GP via NLML
            sigma_T_sq, nlml_loss = gp_module(F_T, z_T_cls, labels)

            # Update GP parameters by minimizing NLML
            gp_optimizer.zero_grad()
            nlml_loss.backward(retain_graph=True) # Retain graph for student update
            gp_optimizer.step()

            # 2.3 Loss Calculation
            # 1. L_UAKD: Uncertainty Aware Feature KD Loss
            # Expand sigma_T_sq to match Feature Map dimensions for broadcasting
            sigma_weight = 1.0 / (sigma_T_sq.view(B, 1, 1, 1) + epsilon)
            mse_features = (F_T - F_S) ** 2
            
            # Multiply MSE by the inverse of uncertainty, then average over batch
            L_UAKD = torch.mean(sigma_weight * mse_features)

            # 2. L_cls: Cross-Entropy Loss
            L_cls = F.cross_entropy(z_S_cls, labels)

            # 3. L_reg: Bounding Box Regression Loss (MSE)
            L_reg = F.mse_loss(z_S_reg, bboxes)

            # Combine total loss
            L_total = (alpha * L_UAKD) + (beta * L_cls) + (gamma * L_reg)

            # 2.4 Backpropagation for Student
            student_optimizer.zero_grad()
            L_total.backward()
            student_optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}")
                print(f"Total Loss: {L_total.item():.4f} | UAKD: {L_UAKD.item():.4f} | NLML: {nlml_loss.item():.4f}")

    return student