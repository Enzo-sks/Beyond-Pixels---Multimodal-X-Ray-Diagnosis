import torch
import torch.nn as nn
from transformers import ViTModel
from text_classification.models.cxr_bert_classifier import CXRBertClassifier

class SingleStreamMMHd(nn.Module):
    def __init__(self, num_classes=21, embed_dim=768, num_heads=8, num_layers=2):
        super(SingleStreamMMHd, self).__init__()

        # 1. PARTIE IMAGE
        self.vit_backbone = ViTModel.from_pretrained('codewithdark/vit-chest-xray')
        # On garde embed_dim constant (768)
        
        # 2. PARTIE TEXTE 
        # On instancie le modèle BERT défini dans text_classification
        # Nous n'utiliserons que le backbone BERT (`.bert`) pour récupérer `last_hidden_state`
        self.text_backbone = CXRBertClassifier(n_labels=num_classes)
        
        # 3. LA TÊTE MULTIMODALE (Single-Stream Transformer)
        # Un encodeur Transformer qui va permettre l'attention croisée
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=0.1,
            batch_first=True # Important pour matcher le format [Batch, Sequence, Features]
        )
        self.multimodal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Token [CLS] multimodal optionnel (ou on utilise celui de l'image)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Tête de classification finale
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, pixel_values, text_input):
        # 1. Extraction des features Image 
        img_outputs = self.vit_backbone(pixel_values=pixel_values)
        # Au lieu de prendre juste l'indice 0, on prend TOUS les tokens de l'image (ex: [Batch, 197, 768])
        img_features = img_outputs.last_hidden_state 
        
        # 2. Extraction des features Texte
        # Le modèle dans `cxr_bert_classifier` expose `.bert` (AutoModel).
        # On construit le mask de padding comme dans CXRBertClassifier.forward
        mask = (text_input != self.text_backbone.pad_id).long()
        text_outputs = self.text_backbone.bert(input_ids=text_input, attention_mask=mask)
        # Récupère la séquence de tokens: [Batch, Seq_len, Hidden]
        text_features = text_outputs.last_hidden_state

        # Si le texte n'a pas de dimension de séquence (ex: [Batch, 768]), on en ajoute une artificiellement :
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1) # Devient [Batch, 1, 768]

        # Prepend un token [CLS] multimodal global
        batch_size = pixel_values.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # [Batch, 1, 768]

        # 3. FUSION SINGLE-STREAM (Concaténation de la séquence)
        # On fusionne sur la dimension 1 (la longueur de la séquence)
        # Forme finale : [Batch, 1 (CLS) + Longueur_Img + Longueur_Texte, 768]
        combined_sequence = torch.cat((cls_tokens, img_features, text_features), dim=1)
        
        # 4. PASSAGE DANS LE TRANSFORMER (Self-Attention Multimodale)
        # Ici, l'image regarde le texte et le texte regarde l'image
        transformer_outputs = self.multimodal_transformer(combined_sequence)
        
        # 5. PRÉDICTION
        # On récupère la sortie du token [CLS] multimodal (indice 0)
        cls_output = transformer_outputs[:, 0, :]
        
        logits = self.classifier(cls_output)
        
        return logits