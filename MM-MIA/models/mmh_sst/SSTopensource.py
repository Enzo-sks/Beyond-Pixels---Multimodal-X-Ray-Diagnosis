import torch
import torch.nn as nn
from transformers import ViTModel, BertModel

class SingleStreamMMHd(nn.Module):
    def __init__(self, num_classes=21, num_heads=8, num_layers=2):
        super(SingleStreamMMHd, self).__init__()

        # 1. PARTIE IMAGE (Modèle public sur Hugging Face)
        self.vit_backbone = ViTModel.from_pretrained('codewithdark/vit-chest-xray')
        embed_dim = self.vit_backbone.config.hidden_size # Récupère automatiquement 768
        
        # 2. PARTIE TEXTE (Modèle public BERT - ex: bert-base-uncased ou un BERT médical)
        # 'bert-base-uncased' possède nativement un hidden_size de 768
        self.text_backbone = BertModel.from_pretrained('bert-base-uncased') 
        
        # 3. LA TÊTE MULTIMODALE (Single-Stream Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=0.1,
            batch_first=True # Format obligatoire : [Batch, Sequence, Features]
        )
        self.multimodal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Token [CLS] multimodal global apprenable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Tête de classification finale (21 classes)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, pixel_values, input_ids, attention_mask):
        # 1. Extraction des caractéristiques de l'image
        img_outputs = self.vit_backbone(pixel_values=pixel_values)
        img_features = img_outputs.last_hidden_state # [Batch, 197, 768]
        
        # 2. Extraction des caractéristiques du texte (via le vrai BERT)
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state # [Batch, Sequence_Length, 768]
        
        # 3. Préparation du token [CLS] global
        batch_size = pixel_values.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # [Batch, 1, 768]

        # 4. FUSION SINGLE-STREAM (Concaténation sur la dimension de la séquence)
        # Forme finale : [Batch, 1 + 197 + Sequence_Length, 768]
        combined_sequence = torch.cat((cls_tokens, img_features, text_features), dim=1)
        
        # 5. GESTION DU MASQUE D'ATTENTION (Crucial pour ignorer le padding du texte)
        # En PyTorch, True = "Ignorer ce token", False = "Calculer l'attention"
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=combined_sequence.device)
        img_mask = torch.zeros((batch_size, img_features.size(1)), dtype=torch.bool, device=combined_sequence.device)
        txt_mask = (attention_mask == 0) # Devient True là où le token est du padding (0)
        
        combined_mask = torch.cat((cls_mask, img_mask, txt_mask), dim=1)
        
        # 6. PASSAGE DANS LE TRANSFORMER MULTIMODAL
        # L'image et le texte interagissent en ignorant les tokens de padding textuels
        transformer_outputs = self.multimodal_transformer(
            combined_sequence, 
            src_key_padding_mask=combined_mask
        )
        
        # 7. PRÉDICTION
        # On extrait la sortie du tout premier token (indice 0), notre [CLS] global fusionné
        cls_output = transformer_outputs[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits