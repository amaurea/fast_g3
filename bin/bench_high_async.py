import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
args = parser.parse_args()
from pixell import bunch, bench
import numpy as np, time, os
import fast_g3

names = ["getsize", "start", "scan", "alloc", "extract", "finish", "free"]
def ftime(times, names):
	parts = []
	ttot  = 0
	for name in names:
		t     = times[name][0]
		ttot += t
		parts.append("%s %6.4f" % (name, t))
	parts.append("tot %6.4f" % ttot)
	return " ".join(parts)

t1 = time.time()
with fast_g3.open_multi(args.ifiles, reuse="full") as reader:
	for data in reader.read():
		print(ftime(reader.times, names))
t2 = time.time()
print("-------------------------")
print(ftime(reader.cum_times, names))
print("Actual total %8.5f" % (t2-t1))
