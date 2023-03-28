from ocr.tools.config import Cfg
from ocr.model.train import Trainer
import torch

config = Cfg.load_config_from_name('vgg_transformer')

dataset_params = {
    'name': 'hw',
    'data_root': r'C:\Users\admin\Desktop\ocr_custom_dataset\custom_dataset',
    'train_annotation': 'train_annotation.txt',
    'valid_annotation': 'val_annotation.txt'
}

params = {
    'print_every': 200,
    'valid_every': 15*200,
    'iters': 20,
    'checkpoint': './checkpoint/transformerocr_checkpoint.pth',
    'export': './weights/transformerocr.pth',
    'metrics': 10000
}

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'

if __name__ == '__main__':
    train = Trainer(config=config, pretrained=True)

