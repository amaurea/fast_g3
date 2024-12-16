import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
args = parser.parse_args()
from pixell import bunch, bench
import numpy as np, time, os
import fast_g3
from fast_g3.g3file import Timer, tadd, BufAlloc, AsyncReader

# This program reads a list of files one at a time
# using the single-file high-level interface, but
# improves on bench_high.py by reusing buffers to avoid
# allocation/deallocation overhead. It is still much
# slower than bench_high_async.py on systems where
# read and extract take about the same time.

names = ["read","alloc","scan","extract","free"]
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

# Set up reusable allocators. This is cumbersome,
# and is handled transparently in open_multi, but
# we do it here to see its effect in isolation from
# async
read_alloc    = BufAlloc()
extract_alloc = BufAlloc()

t1 = time.time()
for fi, ifile in enumerate(args.ifiles):
	timer = Timer(names)
	# This has async in the name, but we're not using it asynchronously
	# here, we're just using it as a way to read a file with control of
	# how the memory is allocated
	with timer("read"):
		with AsyncReader(ifile, allocator=read_alloc) as raw_reader: pass
	with fast_g3.open(ifile, buffer=raw_reader.buf) as reader:
		data = reader.read(allocator=extract_alloc)
		tadd(timer.data, reader.times)
		tadd(cum_timer.data, timer.data)
		with timer("free"):
			read_alloc.reset()
			extract_alloc.reset()
	print(ftime(timer.data, names))
t2 = time.time()
print("-------------------------")
print(ftime(cum_timer.data, names))
print("Actual total %8.5f" % (t2-t1))
