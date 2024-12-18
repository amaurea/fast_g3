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
def split(list_, val):
	res = [[]]
	for item in list_:
		if item == val:
			if len(res[-1]) > 0: res.append([])
		else: res[-1].append(item)
	return res

# Split file list by ,
fname_lists = split(args.ifiles, ",")

t1 = time.time()
with fast_g3.G3MultiMultiFile(reuse="full") as mmfile:
	for fname_list in fname_lists: mmfile.queue(fname_list)
	for reader in mmfile:
		for data in reader.read():
			print(ftime(reader.times, names))
		print("-------------------------")
t2 = time.time()
print("-------------------------")
print(ftime(mmfile.cum_times, names))
print("Actual total %8.5f" % (t2-t1))
