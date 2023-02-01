import argparse
import time
import os
from os import walk
from PIL import Image
import csv
import classify
import tflite_runtime.interpreter as tflite
import platform
import numpy as np
#import pandas as pd
import time
import RPi.GPIO as GPIO
from picamera import PiCamera
from time import sleep
from itertools import zip_longest

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file)
def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).
  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}
def objet_classification(i):
###initialization
  top_k=1  
  threshold=0.0 
  count=5
  model_file="Amobilenet_v2_1.0_224_quant.tflite"
  labels = load_labels("waste_labels.txt")
  interpreter = make_interpreter(model_file)
  interpreter.allocate_tensors()
  size = classify.input_size(interpreter)
#########################################
  image = Image.open('./image/image'+str(i)+'.jpeg').convert('RGB').resize(size)
  classify.set_input(interpreter, image)
  interpreter.invoke()
  classes = classify.get_output(interpreter,top_k, threshold)
    
  return (labels.get(klass.id, klass.id))

       

