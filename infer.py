import os
import argparse
import sys
sys.path.append('./DB')
from detection import Detection
from DB.concern.config import Configurable, Config
sys.path.append('./Recognition/')
from Recognition.ocr.tools.predictor import Predictor
from Recognition.ocr.tools.config import Cfg