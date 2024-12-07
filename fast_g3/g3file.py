from .fast_g3_core import scan, extract
from contextlib import contextmanager
import numpy as np

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
	def __init__(self, fname):
		"""Initialize the G3File by reading the whole file into
		memory and performing a quick scan through the data to
		determine which fields are persent. This can be memory-
		heavy, but is needed for performance reasons."""
		self.fname  = fname
		with open(self.fname, "rb") as ifile:
			self._buffer = ifile.read()
		self._meta  = scan(self._buffer)
		self.fields = {key:Field(**val,owner=self,field_name=key) for key,val in self._meta["fields"].items()}
		self._queue = {}
	@property
	def nsamp(self): return self._meta["nsamp"]
	def read_field(self, field_name, rows=None, oarr=None):
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
		request = {field_name:self._prepare_request(field_name, rows=rows, oarr=oarr)}
		extract(self._buffer, self._meta, request)
		return request[field_name]["oarr"]
	def queue(self, field_name, rows=None, oarr=None):
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
		self._queue[field_name] = self._prepare_request(field_name, rows=rows, oarr=oarr)
	def read(self):
		"""Read all the queued-up fields in parallel, or every field if
		queue() was not used.

		Returns a dictionary of {field_name:field_data}.
		Empties the queue."""
		# If we don't have a queue, then read everything
		if len(self._queue) == 0:
			for name in self.fields:
				self.queue(name)
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
	def _prepare_request(self, field_name, rows=None, oarr=None):
		info  = self.fields[field_name]
		# Allocate output array if necessary
		shape = rowshape(info.shape,rows)
		if oarr is None: oarr = np.zeros(shape, info.dtype)
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
open_g3 = G3File
