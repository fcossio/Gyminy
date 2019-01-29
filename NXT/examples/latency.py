#!/usr/bin/env python

import time
import nxt.locator
from nxt.sensor import *
import nxt.bluesock

b = nxt.bluesock.BlueSock('00:16:53:06:37:7B').connect()

#Touch sensor latency test
toucha = Touch(b, PORT_4)
touchb = Touch(b, PORT_3)
start = time.time()
for i in range(100):
    toucha.get_sample()
    touchb.get_sample()
stop = time.time()
print 'touch latency: %s ms' % (1000 * (stop - start) / 100.0)

#Ultrasonic sensor latency test
ultrasonic = Ultrasonic(b, PORT_1)
start = time.time()
for i in range(100):
    ultrasonic.get_sample()
stop = time.time()
print 'ultrasonic latency: %s ms' % (1000 * (stop - start) / 100.0)

compass = mindsensors.Compassv2(b,PORT_2)
start = time.time()
for i in range(100):
    compass.get_sample()
stop = time.time()
print 'compass latency: %s ms' % (1000 * (stop - start) / 100.0)

start = time.time()
for i in range(100):
    toucha.get_sample()
    touchb.get_sample()
    ultrasonic.get_sample()
    compass.get_sample()
stop = time.time()
print 'full latency: %s ms' % (1000 * (stop - start) / 100.0)