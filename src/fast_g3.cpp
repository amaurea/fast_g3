#include <stdint.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

// Low-level python interface consists of two functions
// * scan(buffer), which returns an info dictionary
//   {ndet:int, nsamp:int, detnames:[], fields:{
//      shape:[], dtype:, segments:[]}}
// * read(buffer, info, fields={name:{oarr, dets=None}}), which
//   reads the given detectors into the provided output arrays
//   for the specified fields. The interface works like this
//   to all the reads to happen as much in parallel as possible.

using std::string;
struct Segment  { int64_t det; uint64_t samp0, nsamp, buf0, nbyte; uint8_t algo; };
typedef std::vector<Segment> Segments;
struct Field    { int npy_type, ndim; Segments segments; };
typedef std::unordered_map<string,Field> Fields;
typedef std::vector<string> Names;
struct ScanInfo { uint64_t ndet, nsamp, samp0, curnsamp; Names detnames; Fields fields; };

size_t dread(const char * src, size_t off, size_t size, void * dest) { memcpy(dest, src+off, size); return off+size; }
string sread(const char * src, size_t * off, size_t size) {
	string s(src+*off, src+*off+size);
	*off += size;
	return s;
}

// Used to ensure python resources are freed.
// Set ptr to NULL if you change your mind
struct PyHandle {
	// This is a bit heavy-handed. We only need it for the Python object types
	template <typename T> PyHandle(T * ptr = NULL):ptr((PyObject*)ptr) {}
	~PyHandle() { Py_XDECREF(ptr); }
	operator PyObject*() { return ptr; }
	PyHandle & inc() { Py_XINCREF(ptr); return *this; }
	PyHandle & dec() { Py_XDECREF(ptr); return *this; }
	PyObject * ptr;
};

struct Buffer {
	Buffer(const char * data, Py_ssize_t size):data(data),size(size),pos(0) {}
	Buffer(const Buffer & buf):data(buf.data),size(buf.size),pos(buf.pos) {}
	void read_raw(size_t len, void * dest) { _check(len); memcpy(dest, data+pos, len); pos += len; }
	template <typename T> T read() { T val; read_raw(sizeof(val), &val); return val; }
	string read_string(size_t len) { _check(len); string s(data+pos,data+pos+len); pos += len; return s; }
	void _check(size_t len) { if(pos+len > size) throw std::out_of_range("read exceeds buffer size"); }
	const char * data;
	Py_ssize_t size, pos;
};

// Why isn't there a numpy function for this?
int numpy_type_size(int numpy_type) {
	switch(numpy_type) {
		case 0: case 1: case  2: case 26: return 1;
		case 3: case 4:          return 2;
		case 5: case 6: case 11: return 4;
		case 7: case 8: case 12: case 14: return 8;
		case 15: return 16;
	}
	return 0;
}

// unordered_map only got contains() in C++20, so use this to be safe}
bool contains(const Fields & fields, const string & name) { return fields.find(name) != fields.end(); }
Field & get_field(Fields & fields, const string & name, int npy_type, int ndim) {
	if(!contains(fields, name)) {
		Field new_field = { npy_type, ndim };
		fields[name] = new_field;
	}
	return fields[name];
}

void process_timesamplemap(Buffer & buf, uint32_t nfield, const string & prefix, ScanInfo & sinfo) {
	for(int fi = 0; fi < nfield; fi++) {
		uint64_t field_name_len = buf.read<uint64_t>();
		string field_name       = prefix + buf.read_string(field_name_len);
		uint64_t val_len        = buf.read<uint64_t>();
		// Handle value
		Buffer subbuf(buf);
		subbuf.pos += 5;
		uint64_t type_name_len  = subbuf.read<uint64_t>();
		string type_name        = subbuf.read_string(type_name_len);

		int npy_type = -1;
		if(type_name == "G3VectorDouble") npy_type = NPY_DOUBLE;
		if(npy_type >= 0) {
			subbuf.pos += 12;
			Segment seg;
			seg.det   = -1;
			seg.samp0 = sinfo.samp0;
			seg.nsamp = subbuf.read<uint64_t>()/numpy_type_size(npy_type);
			seg.buf0  = subbuf.pos;
			seg.algo  = 0;
			// Insert into sinfo
			get_field(sinfo.fields, field_name, npy_type, 1).segments.push_back(seg);
		}
		buf.pos += val_len;
	}
}

void process_supertimestream(Buffer & buf, const string & name, ScanInfo & sinfo) {
	// Skip unknown stuff
	buf.pos += 12;
	uint8_t  flac_level       = buf.read<uint8_t>();
	uint8_t  bsz_workfactor   = buf.read<uint8_t>();
	uint8_t  times_algo       = buf.read<uint8_t>();
	uint32_t nsamp            = buf.read<uint32_t>();
	uint32_t times_nbyte_comp = buf.read<uint32_t>();
	// TODO: Handle times here
	buf.pos += times_nbyte_comp;
	buf.pos += 4; // skip names header
	uint64_t ndet             = buf.read<uint64_t>();
	// Loop through the detectors
	bool need_dets = sinfo.detnames.empty();
	for(int di = 0; di < ndet; di++) {
		uint64_t dlen = buf.read<uint64_t>();
		string dname  = buf.read_string(dlen);
		if(need_dets) sinfo.detnames.push_back(dname);
	}
	if(need_dets) sinfo.ndet  = sinfo.detnames.size();
	// Prepare for the data
	int64_t numpy_type        = buf.read<int64_t>();
	int ntype_size = numpy_type_size(numpy_type);
	uint64_t ndim             = buf.read<uint64_t>();
	uint64_t shape[32];
	buf.read_raw(sizeof(shape), shape);
	uint64_t signal_nbyte_raw = buf.read<uint64_t>();
	uint8_t data_algo         = buf.read<uint8_t>();
	uint64_t nquanta          = buf.read<uint64_t>();
	// Skip the quanta for now
	buf.pos += nquanta*8;
	uint64_t noffset          = buf.read<uint64_t>();
	std::vector<uint32_t> offsets(noffset);
	buf.read_raw(noffset*4, &offsets[0]);
	uint32_t signal_nbyte_comp= buf.read<uint32_t>();

	sinfo.curnsamp = nsamp;
	Field & field = get_field(sinfo.fields, name, numpy_type, 2);

	// Read the detector data
	Buffer dbuf(buf);

	char used_algo = 0;
	uint32_t none_size = 0, flac_size = 0, bz_size = 0, const_size = 0, tot_size = 0;
	size_t det_start, det_end;
	for(int di = 0; di < ndet; di++) {
		uint8_t used_algo = dbuf.read<uint8_t>();
		det_start = dbuf.pos;
		if(used_algo == 0) {
			none_size = nsamp*ntype_size;
			dbuf.pos += none_size;
		} else {
			if(used_algo & 1) {
				flac_size = dbuf.read<uint32_t>();
				dbuf.pos += flac_size;
			}
			if(used_algo & 2) {
				bz_size   = dbuf.read<uint32_t>();
				dbuf.pos += bz_size;
			}
			if(used_algo & 4) {
				const_size = ntype_size;
				dbuf.pos  += const_size;
			}
		}
		det_end   = dbuf.pos;
		tot_size = none_size + flac_size + bz_size + const_size;
		// Ok, done parsing this detector
		Segment seg = { di, sinfo.nsamp, nsamp, det_start, det_end-det_start, used_algo };
		field.segments.push_back(seg);
	}
	buf.pos += signal_nbyte_comp;
}

void process_fields(Buffer & buf, uint32_t nfield, ScanInfo & sinfo) {
	for(int fi = 0; fi < nfield; fi++) {
		uint64_t field_name_len = buf.read<uint64_t>();
		string field_name       = buf.read_string(field_name_len);
		uint64_t val_len        = buf.read<uint64_t>();
		// Handle the value
		Buffer subbuf(buf);
		subbuf.pos += 5;
		uint64_t type_name_len  = subbuf.read<uint64_t>();
		string type_name        = subbuf.read_string(type_name_len);
		if(type_name == "G3TimesampleMap") {
			subbuf.pos += 16;
			uint32_t subfields = subbuf.read<uint32_t>();
			process_timesamplemap(subbuf, subfields, field_name + "/", sinfo);
		} else if(type_name == "G3SuperTimestream") {
			process_supertimestream(subbuf, field_name, sinfo);
		}
		buf.pos += val_len;
	}
}

ScanInfo scan(Buffer & buf) {
	ScanInfo sinfo;
	// Loop over frames in file
	for(int framei = 0; buf.pos < buf.size; framei++) {
		char     foo1    = buf.read<char>();
		uint32_t version = buf.read<uint32_t>();
		uint32_t nfield  = buf.read<uint32_t>();
		uint32_t type    = buf.read<uint32_t>();
		process_fields(buf, nfield, sinfo);
		sinfo.samp0  = sinfo.nsamp;
		sinfo.nsamp += sinfo.curnsamp;
		uint32_t crc     = buf.read<uint32_t>();
	}
	return sinfo;
}

extern "C" {

static PyObject * scan_py(PyObject * self, PyObject * args) {
	const char * bdata;
	Py_ssize_t bsize;
	if(!PyArg_ParseTuple(args, "y#", &bdata, &bsize)) return NULL;
	// Scan the file
	Buffer buf(bdata, bsize);
	ScanInfo sinfo = scan(buf);
	// Convert to python types
	PyObject * meta = PyDict_New(); PyHandle hinf(meta);
	PyHandle ndet  = PyLong_FromUnsignedLong(sinfo.ndet); if(!ndet) return NULL;
	PyDict_SetItemString(meta, "ndet", ndet);
	PyHandle nsamp = PyLong_FromUnsignedLong(sinfo.nsamp);if(!nsamp)return NULL;
	PyDict_SetItemString(meta, "nsamp", nsamp);
	// Set up the detector names
	PyHandle detnames = PyList_New(sinfo.ndet);
	for(int di = 0; di < sinfo.ndet; di++) {
		// No PyHandle needed since SET_ITEM steals the reference
		PyObject * detname = PyUnicode_FromString(sinfo.detnames[di].c_str()); if(!detname) return NULL;
		PyList_SET_ITEM(detnames, di, detname);
	}
	PyDict_SetItemString(meta, "detnames", detnames);
	// Set up the fields
	PyHandle fields = PyDict_New();
	for(const auto & [name, field] : sinfo.fields) {
		PyHandle dtype = PyArray_DescrFromType(field.npy_type);
		PyHandle shape = PyTuple_New(field.ndim);
		for(int dim = 0; dim < field.ndim; dim++) {
			uint64_t val = dim == 0 ? sinfo.nsamp : dim == 1 ? sinfo.ndet : 1;
			PyObject * pyval = PyLong_FromUnsignedLong(val); if(!pyval) return NULL;
			PyTuple_SET_ITEM(shape, field.ndim-1-dim, pyval); // steals
		}
		// [nseg,6] int64_t numpy array for the segments
		npy_intp segshape[2] = { (npy_intp)field.segments.size(), 6 };
		PyHandle segments = PyArray_ZEROS(2, segshape, NPY_INT64, 0);
		for(int si = 0; si < field.segments.size(); si++) {
			const Segment & seg = field.segments[si];
			int64_t * row = (int64_t*)PyArray_GETPTR2((PyArrayObject*)segments.ptr, si, 0);
			row[0] = seg.det;
			row[1] = seg.samp0;
			row[2] = seg.nsamp;
			row[3] = seg.buf0;
			row[4] = seg.nbyte;
			row[5] = seg.algo;
		}
		// Make field dict to add these to
		PyHandle pyfield = PyDict_New();
		PyDict_SetItemString(pyfield, "dtype", dtype);
		PyDict_SetItemString(pyfield, "shape", shape);
		PyDict_SetItemString(pyfield, "segments", segments);
		// And add this field to the fields list
		PyDict_SetItemString(fields, name.c_str(), pyfield);
	}
	PyDict_SetItemString(meta, "fields", fields);
	// Looks like everything is done. Return our result.
	return meta;
}

PyDoc_STRVAR(scan__doc,
	"scan(buffer)\n--\n"
	"\n"
	"Scan the bytes from a .g3 file. Returning the metadata needed to\n"
	"read from it using read(bytes buffer, meta)\n"
);

PyDoc_STRVAR(read__doc,
	"read(buffer, meta, fields=None, dets=None)\n--\n"
	"\n"
	"Read the given fields from the .g3 bytes for the subset of\n"
	"detector indices specified, using the metadata provided by\n"
	"scan(buffer). Returns dictionary of numpy arrays.\n"
);

static PyMethodDef methods[] = {
	{"scan",        scan_py,        METH_VARARGS, scan__doc},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
	PyModuleDef_HEAD_INIT,
	"fast_g3", NULL, -1, methods,
};

PyMODINIT_FUNC
PyInit_fast_g3(void)
{
	PyObject *module = PyModule_Create(&module_def);
	if (module == NULL) return NULL;
	import_array();
	return module;
}

}
