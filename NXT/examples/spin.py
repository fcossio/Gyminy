#!/usr/bin/env python

import nxt.locator
from nxt.motor import *
import nxt.bluesock


def spin_around(b):
    m_left = Motor(b, PORT_B)
    m_left.turn(50, 360)
    m_right = Motor(b, PORT_C)
    m_right.turn(-50, 360)

b = nxt.bluesock.BlueSock('00:16:53:06:37:7B').connect()
spin_around(b)

