import argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
args = parser.parse_args()
from contextlib import contextmanager
from pixell import bunch
import numpy as np, time
import fast_g3

def rowshape(shape, rows):
	if rows is None or len(shape)==1: return shape
	else: return (len(rows),shape[1])

class Bench:
	def __init__(self):
		self.total = bunch.Bunch()
		self.last  = bunch.Bunch()
		self.n     = bunch.Bunch()
	@contextmanager
	def mark(self, name):
		if name not in self.total:
			self.total[name] = 0
			self.n[name] = 0
		t1 = time.time()
		try:
			yield
		finally:
			t2 = time.time()
			self.n[name] += 1
			self.last[name] = t2-t1
			self.total[name] += t2-t1
bench = Bench()

for fi, ifile in enumerate(args.ifiles):
	with bench.mark("all"):
		with bench.mark("read"):
			data = open(ifile,"rb").read()
		with bench.mark("scan"):
			meta = fast_g3.scan(data)
		with bench.mark("zeros"):
			rows = None
			# Read everything
			fields = {}
			for name, finfo in meta["fields"].items():
				fields[name] = {"oarr":np.zeros(rowshape(finfo["shape"],rows), finfo["dtype"]), "rows":rows}
		with bench.mark("extract"):
			fast_g3.extract(data, meta, fields)
	print("read %6.3f scan %6.3f zeros %6.3f extract %6.3f tot %6.3f" % (bench.last.read, bench.last.scan, bench.last.zeros, bench.last.extract, bench.last.all))
	del data
print("--------------------------------------------------------------")
print("read %6.3f scan %6.3f zeros %6.3f extract %6.3f tot %6.3f" % (bench.total.read, bench.total.scan, bench.total.zeros, bench.total.extract, bench.total.all))
