import RPi.GPIO as GPIO
from time import sleep
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

def setAngle(angle,pwm):
    duty = angle / 18 + 1.5
    GPIO.output(26, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(26, False)
    pwm.ChangeDutyCycle(duty)

def reward() :
     GPIO.setup(26, GPIO.OUT)
     pwm=GPIO.PWM(26, 50)
     pwm.start(0)
     # The door is open
     setAngle(100,pwm)
     sleep(2)
     # The door is closed
     setAngle(21,pwm)
     sleep(1)

