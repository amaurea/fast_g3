from .fast_g3_core import scan, extract, start_async_read, end_async_read
from contextlib import contextmanager
import numpy as np, os, time

class G3File:
	"""G3File: High-level interface for fast_g3.

	Example usage:

	with G3File("file.g3") as f:
		# Print the main fields in the file
		print(f.fields)
		# Print the misc fields in the file. These are not
		# merged across frames, so they are presented per-frame
		print(f.frames)
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
	def __init__(self, fname, allocator=None, async_init=False):
		"""Initialize the G3File by reading the whole file into
		memory and performing a quick scan through the data to
		determine which fields are persent. This can be memory-
		heavy, but is needed for performance reasons."""
		self.fname   = fname
		self._timer  = Timer(["getsize","start","alloc","finish","scan","extract"])
		self._reader = AsyncReader(fname, allocator=allocator)
		self._queue  = {}
		self._reader.start()
		self.finished= False
		if not async_init: self.finish()
	def finish(self):
		if self.finished: return
		self._reader.finish()
		with self._timer("scan"):
			self._meta  = scan(self._reader.buf)
		self.fields = {key:Field(**val,owner=self,field_name=key) for key,val in self._meta["fields"].items()}
		tadd(self.times, self._reader.times)
		self.finished = True
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
		with self._timer("extract"):
			extract(self._reader.buf, self._meta, request)
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

		Returns a dictionary of {field_name:field_data}. The misc fields
		are also included in this dictionary under the key "frames".
		Empties the queue."""
		# If we don't have a queue, then read everything
		if len(self._queue) == 0:
			for name in self.fields:
				self.queue(name, allocator=allocator)
		# Do the actual extraction
		with self._timer("extract"):
			extract(self._reader.buf, self._meta, self._queue)
		# Format output
		result = {name:request["oarr"] for name,request in self._queue.items()}
		# Add frames to result. This would cause a collision if one of our
		# main fields has a name "frames", but this is unlikely
		result["frames"] = self._meta["frames"]
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
		self._reader.buf = None
	@property
	def frames(self): return self._meta["frames"]
	@property
	def times(self): return self._timer.data
	def _prepare_request(self, field_name, rows=None, oarr=None, allocator=None):
		info  = self.fields[field_name]
		# Allocate output array if necessary
		shape = rowshape(info.shape,rows)
		if allocator is None: allocator = np
		with self._timer("alloc"):
			if oarr is None: oarr = allocator.empty(shape, info.dtype)
		# rows is small, so just convert it if it has the wrong type
		if rows is not None: rows = np.asarray(rows, dtype=np.int32)
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

# This would be simpler using contextmanager, but I
# wanted to also be able to provide timing information
class AsyncReader:
	def __init__(self, fname, allocator=None):
		self.fname     = fname
		self.allocator = allocator
		self._timer    = Timer(["getsize","alloc","start","finish"])
		with self._timer("getsize"):
			self.size    = os.path.getsize(self.fname)
		with self._timer("alloc"):
			if allocator: self.buf = allocator.bytes(self.size)
			else:         self.buf = bytearray(self.size)
	def start(self):
		with self._timer("start"):
			self.task = start_async_read(self.fname, self.buf)
		return self
	def finish(self):
		with self._timer("finish"):
			end_async_read(self.task)
		return self
	def __enter__(self):
		return self.start()
	def __exit__(self, type, value, traceback):
		# Do not clean up buffer here, since the
		# whole point is to access the data after the
		# read is done
		self.finish()
	@property
	def times(self): return self._timer.data

async_read = AsyncReader

class G3MultiFile:
	"""G3File: High-level interface interleaving the reading
	and extraction of multiple g3 files. Uses asynchronous
	file reads (aio_read) behind the scenes.

	Example usage:

	with G3MultiFile(["file1.g3","file2.g3",...,"fileN.g3"]) as f:
		# Print the fields. Extracted from the first file. Number
		# of samples shown as "?" since the total length across the
		# files is unknown
		print(f)
		# Select some fields for reading. Same syntax as G3File
		f.queue("signal/time")
		f.queue("signal/data", rows=[0,10,20,30])
		f.queue("ancil/az_enc")
		# Process the data for all the files
		for filedata in f.read():
			do_something_with_filedata()

	Due to slow garbage collection with the numpy arrays involved,
	G3MultiFile reuses buffers internally. There are three modes,
	controlled with the "reuse" argument in __init__:
	* "full" (default): Reuse both file buffers and output array buffers.
	  This is fastest, but means that the data yielded from read() is only
	  valid for that iteration - by the time of the next yield it will be
	  invalid. You must therefore copy the data if you want to keep it
	  around.
	* "partial": Reuse only the file buffers. Use this if the behavior in
	  "full" is problematic for you. Medium speed.
	* "none": Don't reuse any buffers. Slowest, with no advantages compared
	  to "partial".
	Alternatively a G3MultiAlloc can be passed in the alloc argument,
	in which case reuse will be ignores.

	If async_init is passed, then __init__ will be asynchronous, and class
	initialization will not finish until .finish() is called. This does
	not affect .read(), which will use asyncronous reads internally
	in either case. The purpose of this argument is to be able to start
	initialization of the next G3MultFile when the previous one when the
	previous one is still extracting its last file.

	Timing information can be accessed through .times and .cum_times, both
	of which are dictionaries {name:[t,n]}, where name is one of
	["getsize","alloc","start","finish","scan","extract","free"]
	representing internal operations. For .times t is the time spent in
	each of these since the last yield and n is the number of invocations.
	For .cum_times it's the same, but since the beginning of the read.
	Note that for reuse != "full" there may be hidden time losses outside
	of the reading code itself, e.g. when you free the data yielded from
	read (which can happen as part of your for loop).

	open_multi is an alias for G3MultiFile.

	Built on the low-level interface scan(), extract(), start_async_read()
	and end_async_read()."""
	def __init__(self, fnames, reuse="full", alloc=None, async_init=False,
			_next_read_callback=None):
		self.fnames = fnames
		# These provide timing data. The timing information clutters up
		# this class quite a bit, but it was useful when debugging
		# performance
		self._timer = Timer(["getsize","alloc","start","finish","scan","extract","free"])
		self._cum_timer = Timer(self._timer.names)
		# Set up our allocators
		self.alloc = alloc or G3MultiAlloc(reuse=reuse)
		# First file doesn't benefit from async, but easiest to do it
		# this way anyway in light of the allocator
		self.g3file = G3File(fnames[0], allocator=self.alloc.ballocs[1], async_init=True)
		# Allow interleaved read chaining
		self._next_read_callback = _next_read_callback or (lambda: None)
		self.finished = False
		if not async_init: self.finish()
	def finish(self):
		if self.finished: return
		self.g3file.finish()
		self.alloc.swap()
		tadd(self.times, self.g3file.times)
		# This meta is only representative of the first file
		self._meta  = self.g3file._meta
		self.fields = {key:MultiField(val["shape"],val["dtype"],val["names"]) for key,val in self._meta["fields"].items()}
		self._queue = {}
		self.finished = True
	@property
	def nfile(self): return len(self.fnames)
	def queue(self, field_name, rows=None):
		if rows is not None: rows = np.asarray(rows, dtype=np.int32)
		self._queue[field_name] = {"rows":rows}
	def read(self):
		for fi in range(self.nfile-1):
			# Start the next read while we work
			next_g3file = G3File(self.fnames[fi+1], allocator=self.alloc.ballocs[1], async_init=True)
			# Process current file
			for field_name, val in self._queue.items():
				self.g3file.queue(field_name, **val, allocator=self.falloc)
			fields = self.g3file.read(allocator=self.alloc.falloc)
			tadd(self.times, self.g3file.times, ["alloc","extract"])
			tadd(self.times, next_g3file.times, ["getsize","alloc","start"])
			tadd(self.cum_times, self.times)
			yield fields
			self._timer.reset()
			with self._timer("free"): del fields
			# Done with it. Move to next file
			self.g3file = next_g3file
			self.g3file.finish()
			tadd(self.times, self.g3file.times, ["finish","scan"])
			# Invalidate memory we're done with.
			with self._timer("free"):
				self.alloc.falloc.reset()
				self.alloc.ballocs[0].reset()
			# Swap file buffer allocators, so we overwrite the oldest
			# buffer next time
			self.alloc.swap()
		# Last file doesn't have a next one to start, but we may have
		# callback to allow us to chain with external async code.
		# When this is called, it is safe to overwrite alloc.ballocs[1]
		self._next_read_callback()
		for field_name, val in self._queue.items():
			self.g3file.queue(field_name, **val, allocator=self.falloc)
		fields = self.g3file.read(allocator=self.alloc.falloc)
		tadd(self.times, self.g3file.times, ["alloc","extract"])
		tadd(self.cum_times, self.times)
		yield fields
		self._timer.reset()
		# Reset the last buffers
		with self._timer("free"):
			del fields
			self.alloc.falloc.reset()
			self.alloc.ballocs[0].reset()
			self._queue = {}
		tadd(self.cum_times, self.times)
	def __enter__(self): return self
	def __exit__(self, type, value, traceback):
		# Clean up memory. The buffer contains the whole raw file
		# read into memory
		self.g3file = None
		self.alloc  = None
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
	@property
	def times(self): return self._timer.data
	@property
	def cum_times(self): return self._cum_timer.data

# Consider replacing the long list of lists interface to
# G3MultiMuliFile with a minimal constructor, and then
# a function (queue?) to register each list of files.
# This would generalize well for sample range support,
# and would also fit nicely with how G3MultiFile works

class G3MultiMultiFile:
	"""Class for chaining reads of multiple G3MultiFiles together.
	This is useful to allow buffer reuse between them, and to allow
	interleaving the extraction of the last file in one with the
	reading of the first file in the next.

	Example usage:

	with G3MultiMultiFile() as mmfile:
		# queue up the file lists to iterate over
		mmfile.queue([wafer2file1,wafer2file2,...])
		mmfile.queue([wafer2file1,wafer2file2,...])
		...
		for f in mmfile:
			# f is a single G3MultiFile that we can work with normally. For example
			dets = f.fields["signal/data"].names
			rows = find(dets, list_of_detectors_we_want)
			f.queue("signal/times")
			f.queue("signal/data", rows=rows)
			f.queue("ancil/az_enc")
			# Read data for each file in this multifile
			for filedata in f.read():
				do_something_with_filedata()
	"""
	def __init__(self, reuse="full", alloc=None):
		self.alloc       = alloc or G3MultiAlloc(reuse=reuse)
		self.fname_lists = []
		self._timer = Timer(["getsize","alloc","start","finish","scan","extract","free"])
		self._cum_timer = Timer(self._timer.names)
	def queue(self, fnames):
		"""Queue up a list of file names. When iterating over a G3MultiMultiFile,
		one G3MultiFile will be yielded for each list of fnames queued up this way."""
		self.fname_lists.append(fnames)
	@property
	def nlist(self): return len(self.fname_lists)
	def __enter__(self): return self
	def __exit__(self, type, value, traceback):
		self.mfile      = None
		self.next_mfile = None
		self.alloc      = None
	def __iter__(self):
		if len(self.fname_lists) == 0: return
		# Handle the first file
		self.mfile = G3MultiFile(self.fname_lists[0], alloc=self.alloc)
		self.next_mfile = None
		for fi in range(self.nlist):
			# If we still have more files left after the current, prepare fot it
			if fi < self.nlist-1:
				next_fnames = self.fname_lists[fi+1]
				def start_next():
					# This will be called when the current mfile has just finished
					# its last read from disk. It will start writing to
					# self.alloc.ballocs[1], which has just become available
					self.next_mfile = G3MultiFile(next_fnames, alloc=self.alloc, async_init=True)
				self.mfile._next_read_callback = start_next
			# Yield the mfile to the caller. We will regain control
			# when reader is finished with all files in this mfile.
			# At this point self.alloc.balloc[0] and self.alloc.falloc
			# will be invalidated, in time for mfile.finish() below to
			# use them.
			yield self.mfile
			self._timer.reset()
			tadd(self.times, self.mfile.cum_times)
			tadd(self.cum_times, self.times)
			# Wait for next file to be ready
			if self.next_mfile:
				self.mfile      = self.next_mfile
				self.mfile.finish()
				self.next_mfile = None
		self.queue = []
	def __repr__(self):
		return "G3MultiMultiFile(%s)" % str(self.fname_lists)
	@property
	def times(self): return self._timer.data
	@property
	def cum_times(self): return self._cum_timer.data

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
open_multi = G3MultiFile

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

class FixedAlloc(BufAlloc):
	"""Allocator that uses an existing buffer"""
	def __init__(self, buffer, name="[unnamed]"):
		self.pos    = 0
		self.buffer = buffer.view(np.uint8)
		self.size   = buffer.size
		self.old    = []
		self.name   = name
	def bytes(self, nbyte):
		res = self.buffer[self.pos:self.pos+nbyte]
		self.pos = round_up(self.pos+nbyte,self.align)
		return res
	def __repr__(self): return "FixedAlloc(name='%s', size=%d, pos=%d)" % (self.name, self.size, self.pos)

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

class G3MultiAlloc:
	"""Class representing the three allocators needed to perform
	inteleaved reads and extracts in G3MultiFile"""
	def __init__(self, reuse="full"):
		if reuse not in ["full","partial","none"]:
			raise ValuError("Unknown buffer reuse strategy '%s'" % str(reuse))
		self.reuse = reuse
		batype = BufAlloc if reuse in ["full","partial"] else DummyAlloc
		fatype = BufAlloc if reuse in ["full"] else DummyAlloc
		self.ballocs = [batype("buf%d" % i) for i in range(2)]
		self.falloc  = fatype("fields")
	def swap(self):
		self.ballocs = self.ballocs[::-1]

class Timer:
	def __init__(self, names):
		self.names = names
		self.reset()
	@contextmanager
	def __call__(self, name):
		d  = self.data[name]
		t1 = time.time()
		try:
			yield
		finally:
			t2 = time.time()
			d[0] += t2-t1
			d[1] += 1
	def reset(self):
		self.data = {name:[0,0] for name in self.names}

def tadd(a, b, names=None, map=None):
	if names is None: names = b.keys()
	for name in names:
		# This is clumsy, and only needed because operating
		# directly on the entries is cumbersome
		if map and name in map: oname = map[name]
		else: oname = name
		da = a[oname]
		db = b[name]
		for i in range(2):
			da[i] += db[i]
