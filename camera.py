from picamera import  PiCamera
from time import sleep


def camera_cap(camera,i):
  camera.framerate = 15
  camera.start_preview()
  sleep(1)
    