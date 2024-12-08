from .fast_g3_core import scan, extract, start_async_read, end_async_read
from contextlib import contextmanager
import numpy as np, os

class G3File:
	"""G3File: High-level interface for fast_g3.

	Example usage:

	with G3File("file.g3") as f:
		# Print the fields in the file
		print(f.fields)
		# Read a single field
		times = f.read_field("signal/times")
		# Read a single field, but only some rows
		some  = f.read_field("signal/data", rows=[0,10,20,30])
		# Queue up multiple fields to be read at once
		f.queue("signal/data")
		f.queue("primary/data", rows=[1,4,5])
		f.queue("anzil/az_enc")
		fields = f.read()
		# Or just read everything at once. Also uses read(),
		# just without building a queue first
		all_fields = f.read()

	open_g3 is an alias for G3File.

	Built on the low-level interface scan() and extract()."""
	def __init__(self, fname, buffer=None):
		"""Initialize the G3File by reading the whole file into
		memory and performing a quick scan through the data to
		determine which fields are persent. This can be memory-
		heavy, but is needed for performance reasons."""
		self.fname  = fname
		if buffer is None:
			with open(self.fname, "rb") as ifile:
				self._buffer = ifile.read()
		else: self._buffer = buffer
		self._meta  = scan(self._buffer)
		self.fields = {key:Field(**val,owner=self,field_name=key) for key,val in self._meta["fields"].items()}
		self._queue = {}
	@property
	def nsamp(self): return self._meta["nsamp"]
	def read_field(self, field_name, rows=None, oarr=None, allocator=None):
		"""Read a single field from the file. If you want more than
		just one field, consider using read() instead.

		Arguments:
		* field_name: Name of the field to read, as seen in .fields
		* rows: Array-like of which rows to read for a 2d field.
		  E.g. [5,0,10] to read the 5th,0th and 10th rows in that order.
		  Default: None, which reads all rows.
		* oarr: Write the output to this array. It must be contiguous
		  along the last axis, and match the dtype and shape of the fied
		  after row selection.

		Returns a numpy array with the values from the field."""
		request = {field_name:self._prepare_request(field_name, rows=rows, oarr=oarr, allocator=allocator)}
		extract(self._buffer, self._meta, request)
		return request[field_name]["oarr"]
	def queue(self, field_name, rows=None, oarr=None, allocator=None):
		"""Queue up a field to be read. All queued fields will
		be read in parallel and returned by a subsequent read() call.

		Arguments:
		* field_name: Name of the field to read, as seen in .fields
		* rows: Array-like of which rows to read for a 2d field.
		  E.g. [5,0,10] to read the 5th,0th and 10th rows in that order.
		  Default: None, which reads all rows.
		* oarr: Write the output to this array. It must be contiguous
		  along the last axis, and match the dtype and shape of the fied
		  after row selection."""
		self._queue[field_name] = self._prepare_request(field_name, rows=rows, oarr=oarr, allocator=allocator)
	def read(self, allocator=None):
		"""Read all the queued-up fields in parallel, or every field if
		queue() was not used.

		Returns a dictionary of {field_name:field_data}.
		Empties the queue."""
		# If we don't have a queue, then read everything
		if len(self._queue) == 0:
			for name in self.fields:
				self.queue(name, allocator=allocator)
		# Do the actual extraction
		extract(self._buffer, self._meta, self._queue)
		# Format output
		result = {name:request["oarr"] for name,request in self._queue.items()}
		self._queue = {}
		return result
	def __repr__(self):
		fieldnames = sorted(self.fields.keys())
		nchar      = max([len(fn) for fn in fieldnames])+2
		msg = "G3File('%s', nsamp=%d, fields={\n" % (self.fname, self.nsamp)
		for fieldname, fdesc in self.fields.items():
			msg += " %-*s: %s,\n" % (nchar, "'"+fieldname+"'", str(fdesc))
		msg += "}"
		return msg
	def __enter__(self): return self
	def __exit__(self, type, value, traceback):
		# Clean up memory. The buffer contains the whole raw file
		# read into memory
		self._buffer = None
	def _prepare_request(self, field_name, rows=None, oarr=None, allocator=None):
		info  = self.fields[field_name]
		# Allocate output array if necessary
		shape = rowshape(info.shape,rows)
		if allocator is None: allocator = np
		if oarr is None: oarr = allocator.empty(shape, info.dtype)
		# Check that everything makes sense
		if oarr.shape != shape or oarr.dtype != info.dtype or oarr.strides[-1] != oarr.itemsize:
			raise ValueError("Field %s output array must have shape %s dtype %s and be contiguous along the last axis" % (name, str(shape), str(info.dtype)))
		return {"oarr":oarr, "rows":rows}

class Field:
	def __init__(self, shape, dtype, names, segments, owner=None, field_name=None):
		self.shape, self.dtype, self.names, self.segments = shape, dtype, names, segments
		# These allow us to call read directly on the field structure, as
		# a convenience
		self.owner = owner
		self.field_name = field_name
	def read(self, rows=None, oarr=None):
		return self.owner.read_field(self.field_name, rows=rows, oarr=oarr)
	def __repr__(self):
		return "Field(shape=%s,dtype=%s,names:%s)" % (str(self.shape), str(self.dtype), format_names(self.names,35))

@contextmanager
def async_read(fname, allocator=None):
	size = os.path.getsize(fname)
	if allocator: buf = allocator.bytes(size)
	else:         buf = bytearray(size)
	task = start_async_read(fname, buf)
	try:
		yield buf
	finally:
		end_async_read(task)

class G3MultiFile:
	def __init__(self, fnames, reuse="full"):
		self.fnames = fnames
		# Set up our allocators
		if reuse not in ["full","partial","none"]:
			raise ValuError("Unknown buffer reuse strategy '%s'" % str(reuse))
		batype = BufAlloc if reuse in ["full","partial"] else DummyAlloc
		fatype = BufAlloc if reuse in ["full"] else DummyAlloc
		self.ballocs = [batype("buf%d" % i) for i in range(2)]
		self.falloc  = fatype("fields")
		# First file doesn't benefit from async, but easiest to do it
		# this way anyway in light of the allocator
		with async_read(fnames[0], allocator=self.ballocs[0]) as buf: pass
		self.g3file = G3File(fnames[0], buf)
		# This meta is only representative of the first file
		self._meta  = self.g3file._meta
		self.fields = {key:MultiField(val["shape"],val["dtype"],val["names"]) for key,val in self._meta["fields"].items()}
		self._queue = {}
	@property
	def nfile(self): return len(self.fnames)
	def queue(self, field_name, rows=None):
		self._queue[field_name] = {"rows":rows}
	def read(self):
		for fi in range(self.nfile-1):
			# Start the next read while we work
			with async_read(self.fnames[fi+1], allocator=self.ballocs[1]) as buf:
				for field_name, val in self._queue.items():
					self.g3file.queue(field_name, **val, allocator=self.falloc)
				yield self.g3file.read()
			# Use newly read bytes to set up new file
			self.g3file = G3File(self.fnames[fi+1], buf)
			# Invalidate memory we're done with.
			self.falloc.reset()
			self.ballocs[0].reset()
			# Swap file buffer allocators, so we overwrite the oldest
			# buffer next time
			self.ballocs = self.ballocs[::-1]
		# Last file doesn't have a next one to start
		for field_name, val in self._queue.items():
			self.g3file.queue(field_name, **val, allocator=self.falloc[1])
		yield self.g3file.read()
		# Reset the last buffers
		self.falloc.reset()
		self.ballocs[0].reset()
		self._queue = {}
	# We do not support read_field here, as it would
	# be too inefficient
	def __repr__(self):
		fieldnames = sorted(self.fields.keys())
		nchar      = max([len(fn) for fn in fieldnames])+2
		msg = "G3MultiFile([\n"
		for fname in self.fnames:
			msg += " %s,\n" % fname
		msg += "], fields={\n"
		for fieldname, fdesc in self.fields.items():
			msg += " %-*s: %s,\n" % (nchar, "'"+fieldname+"'", str(fdesc))
		msg += "}"
		return msg

class MultiField:
	def __init__(self, shape, dtype, names):
		self.shape, self.dtype, self.names = shape, dtype, names
	def __repr__(self):
		if len(self.shape) == 1: shape_str = "(?,)"
		else: shape_str = "(%d,?)" % self.shape[-2]
		return "Field(shape=%s,dtype=%s,names:%s)" % (shape_str, str(self.dtype), format_names(self.names,35))

def rowshape(shape, dets):
	"""Return the correct shape of a field output array
	after row selection. Handles both 1d and 2d fields."""
	if dets is None or len(shape)==1: return shape
	else: return (len(dets),shape[1])

def format_names(names, maxlen):
	msg = ",".join(names)
	if len(msg) > maxlen-2: msg = msg[:maxlen-5]+"..."
	return "["+msg+"]"

# Alias to make it more like python's open()
open_g3    = G3File

############################
# Manual memory management #
############################

# Unecpectedly simply freeing up buffers between each
# file read in the multi-file reader took 40% of the time!
# This was very hard to track down as it happened at the end
# of scopes. The stuff below implements semi-manual memory
# management in the form of growable, reusable buffers.

# Yuck, manual memory management. But it's needed to avoid
# losing most of the async read speedup
def ceil(a,b): return (a+b-1)//b
def round_up(a,b): return ceil(a,b)*b

# Bump allocator for numpy. Hands out memory from an internal buffer.
# If it runs out of memory, replaces buffer with a larger one. Previous
# buffer will still be valid until reset is called. When used in a
# steady-state loop no memory allocation will happen after the first
# iteration
class BufAlloc:
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
	def __repr__(self): return "BufAlloc(name='%s', size=%d, pos=%d)" % (self.name, self.size, self.pos)

class DummyAlloc:
	def __init__(self, name="[unnamed]", size=512, growth_factor=1.5):
		self.name   = name
	def bytes(self, nbyte): return bytearray(nbyte)
	def empty(self, shape, dtype=np.float32): return np.empty(shape, dtype)
	def full(self, shape, val, dtype=np.float32): return np.full(shape, val, dtype)
	def zeros(self, shape, dtype=np.float32): return np.zeros(shape, dtype)
	def array(self, arr): return np.array(arr)
	def reset(self): pass
	def __repr__(self): return "DummyAlloc(name='%s')" % (self.name)
