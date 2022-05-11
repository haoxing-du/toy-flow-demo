import sys
import time

args = sys.argv

N = 1000
for i in range(N):
    print("Steps: %i/%i" % (i + 1, N))
    time.sleep(1)