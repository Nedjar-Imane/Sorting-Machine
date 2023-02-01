import RPi.GPIO as GPIO
from time import sleep
from picamera import PiCamera
from window_motor import *
from capteur_porte import *
from classification import *
from sorting_boart_motor import *
from CapteurIR import *
from camera import *
from capt_object import *
from box_pens_motor import *
from show_LCD import *
import sys
import time
import os
    
def main():
    
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    show_LCD()
    camera=PiCamera()
    i=0
    c=0
    start_time=time.time()
    while True:
       detect=capteur_state()
       if detect==0:
           window_motor()
           camera_cap(camera,i)
           objet_class=objet_classification(i)
           if objet_class=='plastic':
             Trash('left')
             if c==0:
                   start_time=time.time()
             c=c+1
             end_time=time.time()-start_time
             if (c==3) & (end_time<=60):
                 reward()
                 c=0
             elif end_time>60:
                 c=0
                
           else:
             Trash('right')
           
       sleep(0.3)
     
            
    
main()
   