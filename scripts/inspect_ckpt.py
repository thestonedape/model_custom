import torch
import pprint

ckpt_path = 'results/enhanced_checkpoints/best_model.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
print('Saved epoch:', ckpt.get('epoch'))
print('Val results:')
pprint.pprint(ckpt.get('val_results'))
