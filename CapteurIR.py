# Imports
import time
import RPi.GPIO as GPIO

def capteur_state() :
 captIR = 24
 GPIO.setmode(GPIO.BCM)
 GPIO.setwarnings(False)
 GPIO.setup(captIR, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)
 etat = GPIO.input(captIR)
 return etat      
    
