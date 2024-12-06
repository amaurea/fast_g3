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
	@property
	def ndet(self): return self._meta["ndet"]
	@property
	def nsamp(self): return self._meta["nsamp"]
	@property
	def detnames(self): return self._meta["detnames"]
	def read(self, fields=None, detinds=None):
		# If fields is not passed, initialize it to a
		# list of all our field names
		if fields is None:
			fields = self.fields.keys()
		# If we have a list of field names, replace it
		# with a dict fieldname:output_array
		if not hasattr(fields, "keys"):
			ofields = {}
			for fieldname in fields:
				info = self.fields[fieldname]
				ofields[fieldname] = np.zeros(detshape(info.shape, detinds), info.dtype)
			fields = ofields
		else:
			# Check that we have the right shape and data type
			for fieldname, arr in fields.items():
				info  = self.fields[fieldname]
				shape = detshape(info.shape,detinds)
				if arr.shape != shape or arr.dtype != info.dtype or arr.strides[-1] != arr.itemsize:
					raise ValueError("Field %s must have shape %s dtype %s and be contiguous along the last axis" % (fieldname, str(shape), str(info.dtype)))
		# Do the actual extraction
		extract(self._buffer, self._meta, fields, dets=detinds)
		return fields
	def __repr__(self):
		fieldnames = sorted(self.fields.keys())
		nchar      = max([len(fn) for fn in fieldnames])+2
		msg = "G3File('%s', ndet=%d, nsamp=%d, fields={\n" % (self.fname, self.ndet, self.nsamp)
		for fieldname, fdesc in self.fields.items():
			msg += " %-*s: %s,\n" % (nchar, "'"+fieldname+"'", str(fdesc))
		msg += "}"
		return msg

class Field:
	def __init__(self, shape, dtype, segments):
		self.shape, self.dtype, self.segments = shape, dtype, segments
	def __repr__(self):
		return "Field(shape=%s, dtype=%s, %d segments)" % (str(self.shape), str(self.dtype), len(self.segments))

def detshape(shape, dets):
  if dets is None or len(shape)==1: return shape
  else: return (len(dets),shape[1])
