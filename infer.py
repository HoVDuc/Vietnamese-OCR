import sys
sys.path.append('./DB')
sys.path.append('./Recognition/')

import os
import math
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
    parser.add_argument('--image', type=str)
    parser.add_argument('--unclip_ratio', type=float, default=1.5)
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--image_short_side', type=int, default=1152,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--recog', action='store_true', default=False)
    parser.add_argument('--recognition_path', type=str, default='./Recognition/weights/vgg_transformerocr_1M_500k.pth')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true', default=True)
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--visualize_box', action='store_true', default=False)

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    
    image = Image.open(args['image'])
    image = np.array(image)
    
    if args['recog']:
        texts = recog([image], args['recognition_path'])
        print('\n'.join(texts))
    else: 
        conf = Config()
        experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
        experiment_args.update(cmd=args)
        experiment = Configurable.construct_class_from_config(experiment_args)

        demo = Detection(experiment, experiment_args, cmd=args)
        contours = demo.inference(args['image'])

        post = PostProcess()
        imgs = post(image, contours, unclip_ratio=args['unclip_ratio'])
        texts = recog(imgs, args['recognition_path'])
        
        open('result.txt', 'w+').writelines([text + '\n' for text in texts])
        if args['visualize_box']:
            cv2.drawContours(image, post.contours, -1, (0, 255, 0), 3)
        image = Image.fromarray(image)
        # createImage(image, texts)
        im_show = draw_ocr(image, post.contours, texts, font_path='./doc/fonts/CormorantGaramond-Light.ttf')
        im_show = Image.fromarray(im_show)
        im_show.save('result.jpg')
        print('ok')

def recog(images, weight_path):
    config = Cfg.load_config_from_name('vgg_transformer')
    config['predictor']['import'] = weight_path
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

def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/fonts/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)

def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/fonts/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image

def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img

def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)

main()