""" Remote control """

brick_MAC = '00:16:53:06:37:7B' #configure this with the bluetooth MAC address. Can be found in Settings > NXT Version > ID

import time, thread
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
    car['compass']= Ultrasonic(b,compass)
    return car

def turnmotor(m, power):
    power = int(power*4/5)
    a = m.get_tacho().tacho_count
    m.run(int(power/2))
    time.sleep(0.1)
    m.run(power)
    time.sleep(0.1)
    m.idle()
    b = m.get_tacho().tacho_count
    print(b)
    return a

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
        move = 'forward'

        if keys[pygame.K_RIGHT]:
            move = 'forward-right'
        if keys[pygame.K_LEFT]:
            move = 'forward-left'

    elif keys[pygame.K_DOWN]:

        move = 'backward'

        if keys[pygame.K_RIGHT]:
            move = 'backward-right'
        if keys[pygame.K_LEFT]:
            move = 'backward-left'

    elif keys[pygame.K_RIGHT]:
        move = 'right'

    elif keys[pygame.K_LEFT]:
        move = 'left'
    else:
        move = 'none'    
    
    pygame.event.pump()

    
    
    if move == 'forward':
        print(thread.start_new_thread(turnmotor, (car['right_wheel'], 100)))
        print(thread.start_new_thread(turnmotor, (car['left_wheel'], 100)))
    if move == 'backward':
        thread.start_new_thread(turnmotor, (car['right_wheel'], -100))
        thread.start_new_thread(turnmotor, (car['left_wheel'], -100))
    if move == 'right':
        thread.start_new_thread(turnmotor, (car['right_wheel'], -80))
        thread.start_new_thread(turnmotor, (car['left_wheel'], 80))
    if move == 'left':
        thread.start_new_thread(turnmotor, (car['right_wheel'], 80))
        thread.start_new_thread(turnmotor, (car['left_wheel'], -80))
    if move == 'forward-right':
        thread.start_new_thread(turnmotor, (car['right_wheel'], 80))
        thread.start_new_thread(turnmotor, (car['left_wheel'], 100))
    if move == 'forward-left':
        thread.start_new_thread(turnmotor, (car['right_wheel'], 100))
        thread.start_new_thread(turnmotor, (car['left_wheel'], 80))
    if move == 'backward-right':
        thread.start_new_thread(turnmotor, (car['right_wheel'], -80))
        thread.start_new_thread(turnmotor, (car['left_wheel'], -100))
    if move == 'backward-left':
        thread.start_new_thread(turnmotor, (car['right_wheel'], -100))
        thread.start_new_thread(turnmotor, (car['left_wheel'], -80))
    distance = car['ultrasonic'].get_sample()
    direction = car['compass'].get_sample()
    bumpers = [car['left_bumper'].get_sample(), car['right_bumper'].get_sample()]
    encoders = [car['left_wheel'].get_tacho().tacho_count,car['right_wheel'].get_tacho().tacho_count]
    print 'bumpers ', bumpers , 'ultrasonic ', distance, 'direction', direction, 'encoders', encoders

    #time.sleep(0.05)
    #except:
    #print('exiting')
    #break  # if user pressed a key other than the given key the loop will break
