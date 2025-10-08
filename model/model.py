import random
import torch
import torch.nn as nn
from .clip import clip
import torch.nn.functional as F
from .BLIP.models.blip_retrieval import blip_retrieval


import time

class ZSCIR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.model_name = cfg.model_name
        if self.model_name == 'blip':
            self.pretrained_model = blip_retrieval(pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth') # usa Base  #'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth'
            #self.pretrained_model = blip_retrieval(pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth', vit = 'large')  # Large
            self.feature_dim = 256
        elif self.model_name == 'clip-Vit-B/32':
            self.pretrained_model, self.preprocess = clip.load("ViT-B/32",
                                                               device=cfg.device, jit=False)
            self.feature_dim = self.pretrained_model.visual.output_dim
        elif self.model_name == 'clip-Vit-L/14':
            self.pretrained_model, self.preprocess = clip.load("ViT-L/14",
                                                               device=cfg.device, jit=False)
            self.feature_dim = self.pretrained_model.visual.output_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8, dropout=cfg.dropout,
                                                   batch_first=True, norm_first=True, activation="gelu")

        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.logit_scale = 100
        self.dropout = nn.Dropout(cfg.dropout)
        self.combiner_layerw = nn.Linear(self.feature_dim, (self.feature_dim + self.feature_dim) * 4)
        self.combiner_layer = nn.Linear(self.feature_dim + self.feature_dim,  (self.feature_dim + self.feature_dim) * 4)
        self.weighted_layer = nn.Linear(self.feature_dim, 3)

        self.alpha_gate= nn.Parameter(torch.randn(1,self.feature_dim))
        self.beta_gate = nn.Parameter(torch.randn(1, self.feature_dim))

        self.output_layer = nn.Linear((self.feature_dim + self.feature_dim) * 4, self.feature_dim)
        self.sep_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))

        self.sam = 0
        self.rqn = 0
        self.transformer=0
        self.text_encoder_Time =0

    def forward(self, texts, reference_images, target_images):

        target_images= target_images.to(self.device)

        img_text_rep = self.combine_features(reference_images, texts)
        target_features, _ = self.pretrained_model.encode_image(target_images)
        target_features = F.normalize(target_features, dim=-1)
        logits = self.logit_scale * (img_text_rep @ target_features.T)
        return logits

    def combine_features(self, reference_images_texts, texts):
        ref_img_text_features_list = []
        ref_img_text_attention_mask_list = []
        ref_img_pool_text_features_list = []
        text_encoder_time = time.perf_counter()
        for img_text_batch in reference_images_texts: # qui dentro vengono generate le 15 caption
            #print("[DEBUG] img_text_batch:", type(texts))
            random_reference_texts = random.sample(img_text_batch, min(15, len(img_text_batch))) # Sample 15 captions # Non c'è bisogno di usare random.sample se le didascalie sono 15 per ogni immagine. Questo è il caso in cui hai più di 15 didascalie per immagine.
            # Se ho 15 caption non genera errore. Genera errore se uso ad esempio 20 quando ho 15 caption
            #print("[DEBUG] Type of texts:", type(texts))
            #print("[DEBUG] Example of texts:", texts if isinstance(texts, str) else texts[:1])

            # Gestione diversa per BLIP e CLIP
            if self.model_name.startswith('blip'):
                tokenized_ref_img_texts = self.pretrained_model.tokenizer(
                    random_reference_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=45,
                    return_tensors='pt'
                ).to(self.device) # # se come modello usi BLIP usa questo

                # Encode the text and extract features - BLIP restituisce una tupla
                reference_image_text_features, reference_total_image_text_features = self.pretrained_model.encode_text(tokenized_ref_img_texts)
                ref_img_text_attention_mask_list.append(tokenized_ref_img_texts['attention_mask'])

            elif self.model_name.startswith('clip'):
                # Per CLIP, usa clip.tokenize invece del tokenizer
                tokenized_ref_img_texts=clip.tokenize(random_reference_texts, truncate=True).to(self.device) # se come modello uso CLIP usa questo
                
                # Encode the text with CLIP - restituisce una tupla (features_pooled, features_total)
                reference_image_text_features, reference_total_image_text_features = self.pretrained_model.encode_text(tokenized_ref_img_texts) 
                
                # Crea una maschera di attenzione fittizia per CLIP
                attention_mask = (tokenized_ref_img_texts != 0).float()
                ref_img_text_attention_mask_list.append(attention_mask)

        #     # Append features and attention masks - assicurati di aggiungere i tensori, non tuple
            # DEBUG per vedere se ho tensori o tuple
            #print(f"Type of reference_total_image_text_features: {type(reference_total_image_text_features)}")
            #print(f"Type of reference_image_text_features: {type(reference_image_text_features)}")
            ref_img_text_features_list.append(reference_total_image_text_features)
            ref_img_pool_text_features_list.append(reference_image_text_features)

        # # # Stack features and masks along the first dimension
        ref_img_text_features_list = torch.stack(ref_img_text_features_list) # Shape: (num_captions, batch_size, embedding_dim)
        ref_img_pool_text_features_list =torch.stack(ref_img_pool_text_features_list)
        ref_img_text_attention_mask = torch.stack(ref_img_text_attention_mask_list)  # Shape: (num_captions, seq_length)

        # Aggregate features by averaging across captions
        reference_total_image_text_features = torch.mean(ref_img_text_features_list, dim=1)  # Shape: (batch_size, embedding_dim)
        reference_image_pool_features= torch.mean(ref_img_pool_text_features_list, dim=1)
        
        # Aggregate attention masks with torch.any along num_captions dimension
        ref_img_text_attention_mask = torch.any(ref_img_text_attention_mask, dim=1)# Shape: (seq_length,)

        ref_img_text_attention_mask = ref_img_text_attention_mask.float()
        batch_size = reference_total_image_text_features.size(0)
        reference_total_image_features = reference_total_image_text_features.float()

        # Tokenizzazione del testo principale (rel caption)
        if self.model_name.startswith('blip'):
            tokenized_texts = self.pretrained_model.tokenizer(texts, padding='max_length', truncation=True,
                                                              max_length=35,
                                                              return_tensors='pt').to(self.device)
            mask = (tokenized_texts.attention_mask == 0)
        elif self.model_name.startswith('clip'):
            tokenized_texts = clip.tokenize(texts, truncate=True).to(self.device)
            mask = (tokenized_texts == 0)

        # Encoding del testo principale
        text_features, total_text_features = self.pretrained_model.encode_text(tokenized_texts)

        num_patches = reference_total_image_features.size(1)
        sep_token = self.sep_token.repeat(batch_size, 1, 1)
        # sep_token_mask = torch.zeros(batch_size, 1).to(self.device)

        combine_features = torch.cat((total_text_features, sep_token, reference_total_image_features), dim=1)
        image_mask = torch.zeros(batch_size, num_patches + 1).to(self.device) #check here

        mask = torch.cat((mask, image_mask), dim=1)
        self.text_encoder_Time += (time.perf_counter() - text_encoder_time)
        transformers_time = time.perf_counter()
        img_text_rep = self.fusion(combine_features, src_key_padding_mask=mask)
        self.transformer += (time.perf_counter() - transformers_time)

        if self.model_name.startswith('blip'):
            multimodal_img_rep = img_text_rep[:, 36, :] # 36 percè 35 sono i token di BLIP e 1 è il separator token
            multimodal_text_rep = img_text_rep[:, 0, :]
        elif self.model_name.startswith('clip'):
            multimodal_img_rep = img_text_rep[:, 78, :] # here we are extracting seperator token which is considered as cls of image # 77 token di CLIP e 1 separator token
            multimodal_text_rep = img_text_rep[torch.arange(batch_size), tokenized_texts.argmax(dim=-1), :]
        # our model
        concate = torch.cat((multimodal_img_rep, multimodal_text_rep), dim=-1)
        f_U = self.output_layer(self.dropout(F.relu(self.combiner_layer(concate)))) #U_T
        sam_time = time.perf_counter()
        alpha = self.alpha_gate.repeat(batch_size, 1)
        beta = self.beta_gate.repeat(batch_size, 1) # bs, 1, dim

        if self.model_name.startswith('blip'):
            text_length = tokenized_texts.attention_mask.sum(dim=1, keepdim=True).float()  # caption_length
        else:
            text_length = (tokenized_texts != 0).sum(dim=1, keepdim=True).float()

        text_contribution_scale = beta * text_length   # Scale for text_features
        multimodal_text_rep_weighted= alpha * multimodal_text_rep
        multimodal_text_rep_weightedb = text_contribution_scale * multimodal_text_rep

        text_weighted_features = multimodal_text_rep_weighted + multimodal_text_rep_weightedb
        multimodal_img_rep_weighted= alpha * multimodal_img_rep
        self.sam += (time.perf_counter() - sam_time)

        rqn_time = time.perf_counter()
        weighted_f_U= multimodal_img_rep_weighted  + text_weighted_features
        Wf_U = self.output_layer(self.dropout(F.relu(self.combiner_layerw(weighted_f_U)))) #w_q

        weighted = self.weighted_layer(Wf_U) #w_q
        query_rep = weighted[:, 0:1] * text_features + weighted[:, 1:2] * f_U + weighted[:,2:3] * reference_image_pool_features
        #our model
        query_rep = F.normalize(query_rep, dim=-1)

        self.rqn += (time.perf_counter() - rqn_time)
        return query_rep
    
