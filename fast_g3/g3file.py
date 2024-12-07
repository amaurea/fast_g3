from .fast_g3_core import scan, extract
from contextlib import contextmanager
import numpy as np

@contextmanager
def open_g3(fname):
	try:
		f = G3File(fname)
		yield f
	finally:
		pass

class G3File:
	def __init__(self, fname):
		self.fname  = fname
		with open(self.fname, "rb") as ifile:
			self._buffer = ifile.read()
		self._meta  = scan(self._buffer)
		self.fields = {key:Field(**val) for key,val in self._meta["fields"].items()}
		self._queue = {}
	@property
	def nsamp(self): return self._meta["nsamp"]
	def queue(self, field, rows=None, oarr=None):
		info  = self.fields[field]
		# Allocate output array if necessary
		shape = rowshape(info.shape,rows)
		if oarr is None: oarr = np.zeros(shape, info.dtype)
		# Check that everything makes sense
		if oarr.shape != shape or oarr.dtype != info.dtype or oarr.strides[-1] != oarr.itemsize:
			raise ValueError("Field %s output array must have shape %s dtype %s and be contiguous along the last axis" % (name, str(shape), str(info.dtype)))
		# Queue it up
		self._queue[field] = {"oarr":oarr, "rows":rows}
	def read(self):
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

class Field:
	def __init__(self, shape, dtype, names, segments):
		self.shape, self.dtype, self.names, self.segments = shape, dtype, names, segments
	def __repr__(self):
		return "Field(shape=%s, dtype=%s, %d segments)" % (str(self.shape), str(self.dtype), len(self.segments))

def rowshape(shape, dets):
  if dets is None or len(shape)==1: return shape
  else: return (len(dets),shape[1])
