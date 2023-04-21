import torch
path = '/home/yangzhen/checkpoints/segmentation/dinov2_vitb14_pretrain.pth'
new_model_path = '/home/yangzhen/checkpoints/segmentation/new_dinov2_vitb14_pretrain.pth'
model = torch.load(path)


def interpolate_pos_embed(checkpoint_model, new_size=16):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = 196
        num_extra_tokens = 1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
    if 'patch_embed.proj.weight' in checkpoint_model:
        patch_embed = checkpoint_model['patch_embed.proj.weight']
        C_o, C_in, H, W = patch_embed.shape
        patch_embed = torch.nn.functional.interpolate(patch_embed.float(), size=(16, 16), mode='bicubic', align_corners=False)
        checkpoint_model['patch_embed.proj.weight'] = patch_embed





print(model['pos_embed'].shape, model['patch_embed.proj.weight'].shape)
interpolate_pos_embed(model)
print(model['pos_embed'].shape, model['patch_embed.proj.weight'].shape)


def change_name(lists):
    lists.pop(0)
    lists.insert(0, '0')
    lists.insert(0, 'blocks')
    lists.insert(0, 'ViT')
    lists.insert(0, 'backbone')
    name = '.'.join(lists)
    return name

new_model = {}

for key in model:
    key_list = key.split('.')
    if key in ['norm.weight', 'norm.bias', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'cls_token', 'mask_token', 'pos_embed']:
        new_key = 'backbone.ViT.' + key
        # print(new_key)
    elif key_list[0] == 'blocks':
        new_key = change_name(key_list)
    new_model[new_key] = model[key]


torch.save(new_model, new_model_path)


new_pos_embed = new_model['backbone.ViT.pos_embed']
new_patch_embed_proj_weight = new_model['backbone.ViT.patch_embed.proj.weight']

pos_embed = model['pos_embed']
patch_embed_proj_weight = model['patch_embed.proj.weight']
