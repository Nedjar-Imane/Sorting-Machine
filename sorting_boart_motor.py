import RPi.GPIO as GPIO
from time import sleep

def setAngle(angle,pwm):
    duty = angle / 18 + 2
    GPIO.output(18, True)
    pwm.ChangeDutyCycle(duty)
    GPIO.output(18, False)
    pwm.ChangeDutyCycle(duty)

     
def lef(pwm):
     pwm.start(92)
     #The door is open 
     setAngle(165,pwm)
     sleep(1)
     #The door is closed 
     setAngle(90,pwm)
     sleep(1)
def right(pwm):
     pwm.start(92)
     #The door is open 
     setAngle(23,pwm)
     sleep(2)
     #The door is closed 
     setAngle(90,pwm)
     sleep(1)

def Trash(pos):
#Use pin 12 for PWM signal (gpio18)
 pwm_gpio = 18
 frequence = 50
 GPIO.setup(pwm_gpio, GPIO.OUT)
 pwm = GPIO.PWM(pwm_gpio, frequence)
 if pos=='right':
  right(pwm)
 else:
  lef(pwm)



