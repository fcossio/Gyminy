#!/usr/bin/env python

#During this test you need to run any program on the brick
#which doesn't use the messaging system. Most programs fit
#this requirement.

import nxt.locator

import nxt.bluesock
b = nxt.bluesock.BlueSock('00:16:53:06:37:7B').connect()

for box in range(10):
    b.message_write(box, 'message test %d' % box)
for box in range(10):
    local_box, message = b.message_read(box, box, True)
    print local_box, message
print 'Test succeeded!'
