import RPi.GPIO as GPIO
import time

PIN_A = 18
PIN_B = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_A, GPIO.OUT)
GPIO.setup(PIN_B, GPIO.OUT)

while True:
    GPIO.output(PIN_A, GPIO.HIGH)
    GPIO.output(PIN_B, GPIO.LOW)
    time.sleep(0.002)

    GPIO.output(PIN_A, GPIO.LOW)
    GPIO.output(PIN_B, GPIO.HIGH)
    time.sleep(0.002)
