import RPi.GPIO as GPIO
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

def setAngle(angle,pwm):
    duty = angle / 18 + 1.5
    GPIO.output(17, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(17, False)
    pwm.ChangeDutyCycle(duty)

def window_motor () :
     GPIO.setup(17, GPIO.OUT)
     pwm=GPIO.PWM(17, 50)
     pwm.start(0)
     #The door is open"
     setAngle(20,pwm)
     sleep(4)
     #The door closed  
     setAngle(100,pwm)
     sleep(1)
      