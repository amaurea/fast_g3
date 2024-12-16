import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
args = parser.parse_args()
from pixell import bunch, bench
import numpy as np, time
import fast_g3

def ceil(a,b): return (a+b-1)//b
def round_up(a,b): return ceil(a,b)*b

# Bump allocator for numpy. Hands out memory from an internal buffer.
# If it runs out of memory, replaces buffer with a larger one. Previous
# buffer will still be valid until reset is called. When used in a
# steady-state loop no memory allocation will happen after the first
# iteration
class NpAllocator:
	def __init__(self, name="[unnamed]", size=512, growth_factor=1.5):
		self.pos    = 0
		self.align  = 8
		self.size   = int(size)
		self.buffer = np.empty(ceil(self.size,8),np.float64).view(np.uint8)
		self.old    = []
		self.name   = name
		self.growth_factor = growth_factor
	def bytes(self, nbyte):
		if self.pos + nbyte > self.size:
			# Not enough space. Make a new buffer
			self.old.append(self.buffer)
			self.size   = int(nbyte*self.growth_factor)
			self.buffer = np.empty(ceil(self.size,8),np.float64).view(np.uint8)
			self.pos    = 0
		res = self.buffer[self.pos:self.pos+nbyte]
		self.pos = round_up(self.pos+nbyte,self.align)
		return res
	def empty(self, shape, dtype=np.float32):
		# Get memory to place this array in
		buf = self.bytes(np.prod(shape)*np.dtype(dtype).itemsize)
		return np.frombuffer(buf, dtype).reshape(shape)
	def full(self, shape, val, dtype=np.float32):
		arr    = self.empty(shape, dtype=dtype)
		arr[:] = val
		return arr
	def zeros(self, shape, dtype=np.float32):
		return self.full(shape, 0, dtype=dtype)
	def array(self, arr):
		"""Like empty(), but based on the data of an existing array, which
		can be either a numpy or cupy array."""
		res = self.empty(arr.shape, arr.dtype)
		res[:] = arr
		return res
	def reset(self):
		"""Reset memory. All allocated arrays become invalid, but the
		primary buffer will not be freed, so it's ready to hand out
		memory again"""
		self.old = []
		self.pos = 0
	def __repr__(self): return "NpAllocator(name='%s', size=%d, pos=%d)" % (self.name, self.size, self.pos)

# Set up our allocators
bufs = [bunch.Bunch(buf=None, alloc=NpAllocator("buf%d"%i)) for i in range(2)]
field_alloc = NpAllocator("field")

def make_fields(meta, alloc):
	with bench.mark("make_fields"):
		fields = {}
		for name, finfo in meta["fields"].items():
			fields[name] = {"oarr":alloc.empty(finfo["shape"], finfo["dtype"]), "rows":None}
		return fields

# Low-level async benchmark
nfile = len(args.ifiles)
for fi in range(nfile+1):
	with bench.mark("all"):
		# Start read of file
		with bench.mark("rstart"):
			if fi < nfile:
				size        = os.path.getsize(args.ifiles[fi])
				bufs[1].buf = bufs[1].alloc.bytes(size)
				task        = fast_g3.start_async_read(args.ifiles[fi], bufs[1].buf)
		# Scan file
		with bench.mark("scan"):
			if fi > 0: meta = fast_g3.scan(bufs[0].buf)
		# Extract prev file
		with bench.mark("empty"):
			if fi > 0: fields = make_fields(meta, field_alloc)
		with bench.mark("extract"):
			if fi > 0: fast_g3.extract(bufs[0].buf, meta, fields)
		# 40% of runtime is being spent here?!
		with bench.mark("free"):
			if fi > 0:
				bufs[0].alloc.reset()
				field_alloc.reset()
		# Finish read of file
		with bench.mark("rend"):
			if fi < nfile:
				fast_g3.end_async_read(task)
		# Swap buffers
		bufs = bufs[::-1]
	print("rstart %6.3f scan %6.3f empty %6.3f extract %6.3f free %6.3f rend %6.3f tot %6.3f" % (bench.t.rstart, bench.t.scan, bench.t.empty, bench.t.extract, bench.t_tot.free, bench.t.rend, bench.t.all))
print("--------------------------------------------------------------")
print("rstart %6.3f scan %6.3f empty %6.3f extract %6.3f free %6.3f rend %6.3f tot %6.3f" % (bench.t_tot.rstart, bench.t_tot.scan, bench.t_tot.empty, bench.t_tot.extract, bench.t_tot.free, bench.t_tot.rend, bench.t_tot.all))
