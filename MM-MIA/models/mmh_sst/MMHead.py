import torch
import torch.nn as nn
from transformers import ViTModel

class MMHd(nn.Module):
    def __init__(self, num_classes=21, text_feature_dim=768):
        super(MMHd, self).__init__()

        # 1. PARTIE IMAGE
        # On charge le ViT png
        self.vit_backbone = ViTModel.from_pretrained('codewithdark/vit-chest-xray')
        img_feature_dim = self.vit_backbone.config.hidden_size # 768
        
        # 2. PARTIE TEXTE Hugo

        self.text_backbone = nn.Identity()    #TODO À remplacer par model_hugo.backbone
        
        # 3. LA TÊTE MULTIMODALE (Fusion)
        # On concatène Image (768) + Texte (768) = 1536
        combined_dim = img_feature_dim + text_feature_dim
        
        self.multimodal_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512), # Pour stabiliser l'entraînement sur GPU
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes) # Les 21 pathologies
        )

    def forward(self, pixel_values, text_input):
        # Extraction des features Image (ViT)
        # Aziz utilise le token [CLS] à l'indice 0
        img_outputs = self.vit_backbone(pixel_values=pixel_values)
        img_features = img_outputs.last_hidden_state[:, 0] 
        
        # Extraction des features Texte (Modèle d'Hugo)
        # On suppose que le modèle d'Hugo renvoie déjà un vecteur de features
        text_features = self.text_backbone(text_input)
        
        # FUSION : Concaténation des deux mondes
        combined = torch.cat((img_features, text_features), dim=1)
        
        # PRÉDICTION : Passage dans la tête multimodale
        logits = self.multimodal_head(combined)
        
        return logits

# --- Initialisation ---
# model = MultimodalBeyondPixels(num_classes=21)
# model = model.to(DEVICE)