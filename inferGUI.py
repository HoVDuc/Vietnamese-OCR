import sys
sys.path.append('./DB')
sys.path.append('./Recognition/')

import numpy as np
import cv2
from PIL import Image
from detection import Detection
from DB.concern.config import Configurable, Config
from Recognition.ocr.tools.predictor import Predictor
from Recognition.ocr.tools.config import Cfg
from postprocess import PostProcess
import gradio as gr

def main(image, unclip_ratio, box_thresh, recognition, polygon, visualize, visualize_box):
    args = {
        "exp": "./DB/experiments/seg_detector/ic15_resnet50_deform_thre.yaml",
        "resume": "./DB/weights/td500_resnet50",
        "recognition_path": "./Recognition/weights/vgg_transformerocr_1M_500k.pth",
        "image_short_side": 1152,
        "image": image,
        "unclip_ratio": unclip_ratio, 
        "box_thresh": box_thresh,
        "recog": recognition,
        "polygon": polygon,
        "visualize": visualize,
        "visualize_box": visualize_box
    }
    
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
        
        if args['visualize_box']:
            cv2.drawContours(image, post.contours, -1, (0, 255, 0), 3)
        
        return image, "\n".join(texts)
        # image = Image.fromarray(image)
        # # createImage(image, texts)
        # im_show = draw_ocr(image, post.contours, texts, font_path='./doc/fonts/CormorantGaramond-Light.ttf')
        # im_show = Image.fromarray(im_show)
        # im_show.save('result.jpg')

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

def GUI():
    demo = gr.Interface(
        fn=main,
        inputs=[gr.Image(), gr.Slider(0, 2, 1.5), gr.Slider(0, 1, 0.6), "checkbox", "checkbox", "checkbox", "checkbox"],
        outputs=["image", "text"]
    )

    demo.launch()

if __name__ == "__main__":
    GUI()