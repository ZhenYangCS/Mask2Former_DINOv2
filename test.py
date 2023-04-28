import torch
model_path = "/home/yangzhen/code/Mask2Former_DINOv2/output/Dinov2_Base/model_0099999.pth"
origin_path = '/home/yangzhen/checkpoints/segmentation/new_dinov2_vitb14_pretrain.pth'
name = 'backbone.ViT.pos_embed'
model_dict = torch.load(model_path)
origin_dict = torch.load(origin_path)
model_dict = model_dict['model']
weight_model = model_dict[name].cpu()
weight_origin = origin_dict[name].cpu()
print(weight_model - weight_origin)


# model_dict = torch.load(model_path)
# model_names = list(model_dict['model'].keys())
# base_lrs = model_dict['trainer']['hooks']['LRScheduler']['base_lrs']
# print(len(model_names), len(base_lrs))

# for i in range(len(base_lrs)):
#     print(model_names[i], base_lrs[i])

# CUDA_VISIBLE_DEVICES=3 python test.py 






# backbone.ViT.pos_embed 1e-05
