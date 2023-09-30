import torch
from collections import OrderedDict

# my_model = torch.load('/work/vig/yimingx/deper_logs_test/release/parq_23-06-09-22-03-49-epoch0000-val_loss0.000-train_loss0.000.ckpt')
official_model = torch.load('/work/vig/yimingx/deper_log_compare/model_perceiverIO_23-02-17-10-51-32/checkpoints/epoch0-step15259.ckpt')

official_model = official_model['state_dict']
new_weight_dict = OrderedDict()
new_model = {}
for key, value in official_model.items():
    if 'input_preprocessors.resnet_fpn' in key:
        ind = len('input_preprocessors.resnet_fpn')
        key = 'backbone2d' + key[ind:]
    if 'input_tokenizers.rgb_snippet.to_tokens.project' in key:
        continue
    if 'input_tokenizers.rgb_snippet' in key:
        ind = len('input_tokenizers.rgb_snippet.token_position_encoder')
        key = 'add_ray_pe' + key[ind:]
    if 'ray_offset_scale' in key:
        continue
    if 'query_tokenizers.deper.transformer' in key:
        ind = len('query_tokenizers.deper.transformer')
        key = 'box3d_decoder.parq_module' + key[ind:]
    if 'query_tokenizers.deper' in key:
        ind = len('query_tokenizers.deper')
        key = 'box3d_decoder' + key[ind:]
    new_weight_dict[key] = value

new_model['state_dict'] = new_weight_dict
torch.save(new_model, '/work/vig/yimingx/deper_log_compare/parq_release.ckpt')