import sys
sys.path.append('./DB')
sys.path.append('./Recognition/')

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageColor
from detection import Detection
from DB.concern.config import Configurable, Config
from Recognition.ocr.tools.predictor import Predictor
from Recognition.ocr.tools.config import Cfg
from postprocess import PostProcess

def main():
    parser = argparse.ArgumentParser(description='Text Recognition inference')
    parser.add_argument('--exp', type=str, default='./DB/experiments/seg_detector/ic15_resnet50_deform_thre.yaml')
    parser.add_argument('--resume', default='./DB/weights/td500_resnet50')
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--unclip_ratio', type=float, default=1.5)
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--image_short_side', type=int, default=1152,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true', default=True)
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--visualize_box', action='store_true', default=False)

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    
    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    demo = Detection(experiment, experiment_args, cmd=args)
    contours = demo.inference(args['image_path'])

    image = Image.open(args['image_path'])
    image = np.array(image)
    post = PostProcess()
    imgs = post(image, contours, unclip_ratio=args['unclip_ratio'])
    texts = recog(imgs)
    if args['visualize_box']:
        cv2.drawContours(image, post.contours, -1, (0, 255, 0), 3)
    image = Image.fromarray(image)
    createImage(image, texts)

def recog(images):
    config = Cfg.load_config_from_name('vgg_transformer')
    config['predictor']['import'] = './Recognition/weights/vgg_transformerocr_500k.pth'
    config['predictor']['beamsearch'] = True
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    detector = Predictor(config)
    images.reverse()
    predict = []
    for image in images:
        image = Image.fromarray(image)
        predict.append(detector.predict(image))
    return predict 

def add_text(image, texts):
    new_img = Image.fromarray(np.full_like(image, 255))
    _, h = new_img.size
    font = ImageFont.truetype('./datagenerator/trdg/fonts/vi/Roboto-Black.ttf', h // 50)
    text = '\n'.join(texts)
    draw = ImageDraw.Draw(new_img)
    draw.text(xy=(20, 20), text=text, font=font, fill='black')
    return np.array(new_img)
     
def createImage(image, texts):
    image = np.array(image)
    text_img = add_text(image, texts)
    image = np.concatenate([image, text_img], axis=1)
    image = Image.fromarray(image)
    image.save('./test.jpg')
        
main()