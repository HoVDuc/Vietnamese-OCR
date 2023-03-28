import torch
import numpy as np
import math
from PIL import Image
from torch.nn.functional import log_softmax, softmax


def resize(w, h, expected_height, image_min_width, image_max_width):
    # Tính new width dựa vào expected height
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    # Làm tròn số và nhân với round_to
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    """_summary_

    Args:
        image (Image): ảnh đầu cần predict
        image_height (int): Chiều cao của ảnh
        image_min_width (int): Chiều rộng tối thiểu của ảnh
        image_max_width (_type_): Chiều rộng tối đa của ảnh

    Returns:
        np.array: Ảnh được xử lý
    """
    img = image.convert('RGB')  # Đưa về màu RGB

    w, h = img.size
    # Resize lại chiều rộng ảnh
    new_w, image_height = resize(
        w, h, image_height, image_min_width, image_max_width)

    # Resize lại ảnh dựa vào chiều rộng mới và image cố định, với kernel ANTIAIAS
    img = img.resize((new_w, image_height), Image.ANTIALIAS)

    # Đưa về chiều của pytorch (,c, h, w)
    img = np.asarray(img).transpose(2, 0, 1)
    img = img/255  # scale lại ảnh
    return img
