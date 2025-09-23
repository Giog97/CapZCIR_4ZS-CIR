from model.BLIP.models.blip_retrieval import blip_retrieval

# load model
#pretrained_model = blip_retrieval(pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth') # usa Base  #'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth'
pretrained_model = blip_retrieval(pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth')  # Large

# check the patch embedding
print(pretrained_model.visual_encoder.patch_embed.proj.kernel_size)
#model_base_retrieval_coco.pth--> (16, 16) --> la dimensione del patch embedding è 16×16 → quindi il backbone è ViT-B/16.
#model_large_retrieval_coco.pth--> (16, 16) --> la dimensione del patch embedding è 16×16 → quindi il backbone è ViT-L/16.