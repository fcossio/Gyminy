""" Remote control """

brick_MAC = '00:16:53:06:37:7B' #configure this with the bluetooth MAC address. Can be found in Settings > NXT Version > ID

import time
import pygame
from nxt.sensor import *
from nxt.motor import *
import nxt.bluesock


def bumper_car(b, m_left, m_right, b_left, b_right, ultrasonic, compass):
    car = {}
    #returns a dict with the parts of the bumper car
    car['left_wheel'] = Motor(b, m_left)
    car['right_wheel'] = Motor(b, m_right)
    car['left_bumper'] = Touch(b, b_left)
    car['right_bumper'] = Touch(b, b_right)
    car['ultrasonic'] = Ultrasonic(b, ultrasonic)
    car['compass']= mindsensors.Compassv2(b,compass)
    return car

#connect to brick
print 'connecting to ', brick_MAC, '...'
b = nxt.bluesock.BlueSock(brick_MAC).connect()
print (b)
b.play_tone_and_wait(500, 500)
b.play_tone_and_wait(550, 500)
b.play_tone_and_wait(600, 500)

#define wheels
car=bumper_car(b, PORT_C, PORT_B, PORT_4, PORT_3, PORT_1, PORT_2)

#Start controlling with keyboard
pygame.init()
screen = pygame.display.set_mode((640, 480))
print(pygame.key.get_focused())
r_speed = 0
l_speed = 0
while True: 
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        if r_speed < 30 and r_speed > 0 : r_speed = 30
        if l_speed < 30 and l_speed > 0 : l_speed = 30
        r_speed = min(r_speed + 15, 100)
        l_speed = min(l_speed + 15, 100)

        if keys[pygame.K_RIGHT]:
            r_speed = l_speed - 20
        if keys[pygame.K_LEFT]:
            l_speed = r_speed - 20

    elif keys[pygame.K_DOWN]:

        r_speed = max(r_speed - 15, -100)
        l_speed = max(l_speed - 15, -100)

        if keys[pygame.K_RIGHT]:
            r_speed = l_speed + 20
        if keys[pygame.K_LEFT]:
            l_speed = r_speed + 20

    elif keys[pygame.K_RIGHT]:
        r_speed = max(r_speed - 10, -80)
        l_speed = min(l_speed + 10, 80 )

    elif keys[pygame.K_LEFT]:
        r_speed = min(r_speed + 10, 80 )
        l_speed = max(l_speed - 10, -80)

    else:
        if r_speed > 0:
            r_speed = max(r_speed - 5, 0)
        if r_speed < 0:
            r_speed = min(r_speed + 5, 0)
        if l_speed > 0:
            l_speed = max(l_speed - 5, 0)
        if l_speed < 0:
            l_speed = min(l_speed + 5, 0)    
    
    pygame.event.pump()
    
    car['left_wheel'].run(l_speed)
    car['right_wheel'].run(r_speed)
    #distance = car['ultrasonic'].get_sample()
    #direction = car['compass'].get_sample()
    #bumpers = [car['left_bumper'].get_sample(), car['right_bumper'].get_sample()]
    #print 'bumpers ', bumpers , 'ultrasonic ', distance, 'direction', direction

    time.sleep(0.05)
    #except:
    #print('exiting')
    #break  # if user pressed a key other than the given key the loop will break
