import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
args = parser.parse_args()
from pixell import bunch, bench
import numpy as np, time, os
import fast_g3
from fast_g3.g3file import Timer, tadd

# This program uses the high-level single-file
# interface to read a list of files. This is
# inefficient for two reasons:
# 1. the cpu idles while reading, and
#    the disk idles while extracting
# 2. Large arrays are allocated and deallocated
#    for each file. This is surprisngly slow!
#    This time loss is incorrectly attributed to
#    the extract step due to this being when
#    python chooses to do this.
#
# Issue #2 is handled in bench_high_reuse.py.
#
# Both #2 and #1 are handled in bench_high_async.py,
# which uses the high-level multi-file interface.
# That's the way I recommend reading multiple files.

names = ["read","alloc","scan","extract"]
cum_timer = Timer(names)

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
for fi, ifile in enumerate(args.ifiles):
	with fast_g3.open(ifile) as reader:
		data = reader.read()
		tadd(cum_timer.data, reader.times)
		print(ftime(reader.times, names))

t2 = time.time()
print("-------------------------")
print(ftime(cum_timer.data, names))
print("Actual total %8.5f" % (t2-t1))
