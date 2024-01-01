import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hinge Loss (Custom)
class HingeLoss(nn.Module):
    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

# Dice Loss (Custom)
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, epsilon=1e-6):
        intersection = inputs * targets
        union = inputs + targets
        dice_score = (2. * intersection + epsilon) / (union + epsilon)
        return 1 - dice_score

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss

def contrastive_loss(embeddings1, embeddings2, targets, margin=1.0):
    distances = F.pairwise_distance(embeddings1, embeddings2)
    loss = 0.5 * (targets * distances + (1 - targets) * F.relu(margin - distances))
    return loss

# Test cases
def test_losses():
    hinge_loss = HingeLoss()
    huber_loss = nn.SmoothL1Loss(reduction='none')
    dice_loss = DiceLoss()
    cosine_similarity_loss = nn.CosineEmbeddingLoss(reduction='none')

    predictions = torch.tensor([0.8, -1.5])
    targets = torch.tensor([1, -1])
    print("Hinge Loss:", hinge_loss(predictions, targets))

    predictions = torch.tensor([1.5, 0.5])
    targets = torch.tensor([1, 0])
    print("Huber Loss:", huber_loss(predictions, targets))

    inputs = torch.tensor([0.7, 0.3])
    targets = torch.tensor([1, 0])
    print("Dice Loss:", dice_loss(inputs, targets))

    inputs = torch.tensor([0.9, 0.1], dtype=torch.float32)
    targets = torch.tensor([1, 0], dtype=torch.float32)
    print("Focal Loss:", focal_loss(inputs, targets))

    embeddings1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
    embeddings2 = torch.tensor([[2, 3], [4, 5]], dtype=torch.float)
    targets = torch.tensor([1, 0], dtype=torch.float)
    print("Contrastive Loss:", contrastive_loss(embeddings1, embeddings2, targets))

    embeddings1 = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
    embeddings2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
    targets = torch.tensor([1, -1], dtype=torch.float)
    print("Cosine Similarity Loss:", cosine_similarity_loss(embeddings1, embeddings2, targets))

test_losses()
