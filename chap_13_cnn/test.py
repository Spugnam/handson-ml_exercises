#!/usr/local/bin//python3

import time


t1 = time.time()
time.sleep(2)
t2 = time.time()
elapsed = t2-t1


print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))
print(elapsed)
