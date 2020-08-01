import picamera
import time

with picamera.PiCamera() as camera:
    camera.resolution = (416,416)
    camera.start_preview()
    time.sleep(30)
    camera.capture("test.jpg")