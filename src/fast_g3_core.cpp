#include <stdint.h>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <FLAC/stream_decoder.h>
#include <bzlib.h>

#if NPY_ABI_VERSION < 0x02000000
  #define PyDataType_ELSIZE(descr) ((descr)->elsize)
#endif

// Low-level python interface consists of two functions
// * scan(buffer), which returns an info dictionary
//   {ndet:int, nsamp:int, detnames:[], fields:{
//      shape:[], dtype:, segments:[]}}
// * read(buffer, info, fields={name:{oarr, dets=None}}), which
//   reads the given detectors into the provided output arrays
//   for the specified fields. The interface works like this
//   to all the reads to happen as much in parallel as possible.

using std::string;
struct Segment  { int64_t det, samp0, nsamp, buf0, nbyte; double quantum; uint8_t algo; };
typedef std::vector<Segment> Segments;
struct Field    { int npy_type, ndim; Segments segments; };
typedef std::unordered_map<string,Field> Fields;
typedef std::vector<string> Names;
struct ScanInfo {
	ScanInfo():ndet(0),nsamp(0),curnsamp(0) {}
	int64_t ndet, nsamp, curnsamp; Names detnames; Fields fields;
};
struct Work { const void *ibuf; void *obuf; int64_t inbyte, onbyte; double quantum; int npy_type, itemsize; uint8_t algo; };
typedef std::vector<Work> WorkList;

// Used to ensure python resources are freed.
// Set ptr to NULL if you change your mind
struct PyHandle {
	// This is a bit heavy-handed. We only need it for the Python object types
	PyHandle():ptr(NULL) {}
	template <typename T> PyHandle(T * ptr):ptr((PyObject*)ptr) {}
	~PyHandle() { Py_XDECREF(ptr); }
	operator PyObject*() { return ptr; }
	operator bool()      { return ptr != NULL; }
	PyHandle & inc() { Py_XINCREF(ptr); return *this; }
	PyHandle & dec() { Py_XDECREF(ptr); return *this; }
	PyObject * ptr;
};

struct Buffer {
	Buffer(const char * data, ssize_t size):data(data),size(size),pos(0) {}
	Buffer(const void * data, ssize_t size):data((const char*) data),size(size),pos(0) {}
	Buffer(const Buffer & buf):data(buf.data),size(buf.size),pos(buf.pos) {}
	void read_raw(ssize_t len, void * dest) { _check(len); memcpy(dest, data+pos, len); pos += len; }
	template <typename T> T read() { T val; read_raw(sizeof(val), &val); return val; }
	string read_string(ssize_t len) { _check(len); string s(data+pos,data+pos+len); pos += len; return s; }
	void _check(ssize_t len) { if(pos+len > size) throw std::out_of_range("read exceeds buffer size"); }
	const char * data;
	ssize_t size, pos;
};

struct FlacHelper {
	const void * idata;
	void * odata;
	int64_t ipos, isize, opos, osize;
};

FLAC__StreamDecoderReadStatus FlacReadBuf(const FLAC__StreamDecoder *decoder, FLAC__byte buffer[], size_t *bytes, void *client_data) {
	FlacHelper * info = (FlacHelper*)client_data;
	if((uint64_t)info->isize < *bytes+info->ipos) *bytes = info->isize-info->ipos;
	if(*bytes==0) return FLAC__STREAM_DECODER_READ_STATUS_END_OF_STREAM;
	memcpy(buffer, (char*)info->idata+info->ipos, *bytes);
	info->ipos += *bytes;
	return FLAC__STREAM_DECODER_READ_STATUS_CONTINUE;
}
FLAC__StreamDecoderSeekStatus FlacSeekBuf(const FLAC__StreamDecoder *decoder, FLAC__uint64 absolute_byte_offset, void *client_data) {
	FlacHelper * info = (FlacHelper*)client_data;
	info->ipos = absolute_byte_offset;
	return FLAC__STREAM_DECODER_SEEK_STATUS_OK;
}
FLAC__StreamDecoderTellStatus FlacTellBuf(const FLAC__StreamDecoder *decoder, FLAC__uint64 *absolute_byte_offset, void *client_data) {
	FlacHelper * info = (FlacHelper*)client_data;
	*absolute_byte_offset = info->ipos;
	return FLAC__STREAM_DECODER_TELL_STATUS_OK;
}
FLAC__StreamDecoderLengthStatus FlacLenBuf(const FLAC__StreamDecoder *decoder, FLAC__uint64 *stream_length, void *client_data) {
	FlacHelper * info = (FlacHelper*)client_data;
	*stream_length = info->isize;
	return FLAC__STREAM_DECODER_LENGTH_STATUS_OK;
}
FLAC__bool FlacEofBuf(const FLAC__StreamDecoder *decoder, void *client_data) {
	FlacHelper * info = (FlacHelper*)client_data;
	return info->ipos >= info->isize;
}
FLAC__StreamDecoderWriteStatus FlacWriteBuf(const FLAC__StreamDecoder *decoder, const FLAC__Frame *frame, const FLAC__int32 *const buffer[], void *client_data) {
	FlacHelper * info = (FlacHelper*)client_data;
	ssize_t n = frame->header.blocksize*4;
	if(info->opos+n > info->osize) return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
	memcpy((char*)info->odata+info->opos, buffer[0], n);
	info->opos += n;
	return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}
void FlacErrBuf(const FLAC__StreamDecoder *decoder, FLAC__StreamDecoderErrorStatus status, void *client_data) {
	fprintf(stderr, "FLAC error: %d\n", status);
}

struct FlacDecoder {
	FlacDecoder()  { decoder = FLAC__stream_decoder_new(); }
	~FlacDecoder() { FLAC__stream_decoder_delete(decoder); }
	int decode(const void * idata, void * odata, int64_t isize, int64_t osize) {
		FlacHelper info = { idata, odata, 0, isize, 0, osize };
		int err = 0;
		if((err=FLAC__stream_decoder_init_stream(decoder, FlacReadBuf, FlacSeekBuf,
				FlacTellBuf, FlacLenBuf, FlacEofBuf, FlacWriteBuf, NULL, FlacErrBuf,
				(void*)&info))!=FLAC__STREAM_DECODER_INIT_STATUS_OK) return 1;
		if(!(err=FLAC__stream_decoder_process_until_end_of_stream(decoder))) return 2;
		if(!(err=FLAC__stream_decoder_finish(decoder))) return 3;
		return 0;
	}
	FLAC__StreamDecoder *decoder;
};

// Why isn't there a numpy function for this?
int numpy_type_size(int npy_type) {
	switch(npy_type) {
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

void process_timesamplemap(Buffer & buf, int32_t nfield, const string & prefix, ScanInfo & sinfo) {
	for(int32_t fi = 0; fi < nfield; fi++) {
		int64_t field_name_len = buf.read<int64_t>();
		string field_name       = prefix + buf.read_string(field_name_len);
		int64_t val_len        = buf.read<int64_t>();
		// Handle value
		Buffer subbuf(buf);
		subbuf.pos += 5;
		int64_t type_name_len  = subbuf.read<int64_t>();
		string type_name        = subbuf.read_string(type_name_len);

		int npy_type = -1;
		if(type_name == "G3VectorDouble") npy_type = NPY_DOUBLE;
		if(npy_type >= 0) {
			subbuf.pos += 12;
			Segment seg;
			seg.det   = -1;
			seg.samp0 = sinfo.nsamp;
			seg.nsamp = subbuf.read<int64_t>();
			seg.nbyte = seg.nsamp*numpy_type_size(npy_type);
			seg.buf0  = subbuf.pos;
			seg.quantum=0;
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
	uint8_t flac_level       = buf.read<uint8_t>();   (void)flac_level;
	uint8_t bsz_workfactor   = buf.read<uint8_t>();   (void)bsz_workfactor;
	uint8_t times_algo       = buf.read<uint8_t>();   (void)times_algo;
	int32_t nsamp            = buf.read<int32_t>();
	int32_t times_nbyte_comp = buf.read<int32_t>();
	// TODO: Handle times here
	buf.pos += times_nbyte_comp;
	buf.pos += 4; // skip names header
	int64_t ndet             = buf.read<int64_t>();
	// Loop through the detectors
	bool need_dets = sinfo.detnames.empty();
	for(int32_t di = 0; di < ndet; di++) {
		int64_t dlen = buf.read<int64_t>();
		string dname  = buf.read_string(dlen);
		if(need_dets) sinfo.detnames.push_back(dname);
	}
	if(need_dets) sinfo.ndet  = sinfo.detnames.size();
	// Prepare for the data
	int64_t npy_type          = buf.read<int64_t>();
	int itemsize = numpy_type_size(npy_type);
	int64_t ndim             = buf.read<int64_t>();
	int64_t shape[32];
	buf.read_raw(sizeof(shape), shape);
	int64_t signal_nbyte_raw = buf.read<int64_t>();  (void)signal_nbyte_raw;
	uint8_t data_algo        = buf.read<uint8_t>();  (void)data_algo;
	int64_t nquanta          = buf.read<int64_t>();
	// We need the quanta for decoding
	std::vector<double> quanta(nquanta);
	if(nquanta > 0)
		buf.read_raw(8*nquanta, &quanta[0]);
	int64_t noffset          = buf.read<int64_t>();
	std::vector<int32_t> offsets(noffset);
	buf.read_raw(noffset*4, &offsets[0]);
	int32_t signal_nbyte_comp= buf.read<int32_t>();

	sinfo.curnsamp = nsamp;
	Field & field = get_field(sinfo.fields, name, npy_type, 2);
	(void)ndim;

	// Read the detector data
	Buffer dbuf(buf);

	int32_t none_size = 0, flac_size = 0, bz_size = 0, const_size = 0, tot_size = 0;
	ssize_t det_start, det_end;
	for(int32_t di = 0; di < ndet; di++) {
		uint8_t used_algo = dbuf.read<uint8_t>();
		det_start = dbuf.pos;
		if(used_algo == 0) {
			none_size = nsamp*itemsize;
			dbuf.pos += none_size;
		} else {
			if(used_algo & 1) {
				flac_size = dbuf.read<int32_t>();
				dbuf.pos += flac_size;
			}
			if(used_algo & 2) {
				bz_size   = dbuf.read<int32_t>();
				dbuf.pos += bz_size;
			}
			if(used_algo & 4) {
				const_size = itemsize;
				dbuf.pos  += const_size;
			}
		}
		det_end   = dbuf.pos;
		tot_size = none_size + flac_size + bz_size + const_size; (void)tot_size;
		// Ok, done parsing this detector
		double quantum = nquanta > 0 ? quanta[di] : 1.0;
		Segment seg = { di, sinfo.nsamp, nsamp, det_start, det_end-det_start, quantum, used_algo };
		field.segments.push_back(seg);
	}
	buf.pos += signal_nbyte_comp;
}

void process_fields(Buffer & buf, int32_t nfield, ScanInfo & sinfo) {
	for(int fi = 0; fi < nfield; fi++) {
		int64_t field_name_len = buf.read<int64_t>();
		string field_name       = buf.read_string(field_name_len);
		int64_t val_len        = buf.read<int64_t>();
		// Handle the value
		Buffer subbuf(buf);
		subbuf.pos += 5;
		int64_t type_name_len  = subbuf.read<int64_t>();
		string type_name        = subbuf.read_string(type_name_len);
		if(type_name == "G3TimesampleMap") {
			subbuf.pos += 16;
			int32_t subfields = subbuf.read<int32_t>();
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
		uint8_t foo1    = buf.read<uint8_t>(); (void)foo1;
		int32_t version = buf.read<int32_t>(); (void)version;
		int32_t nfield  = buf.read<int32_t>(); (void)nfield;
		int32_t type    = buf.read<int32_t>(); (void)type;
		process_fields(buf, nfield, sinfo);
		sinfo.nsamp += sinfo.curnsamp;
		int32_t crc     = buf.read<int32_t>(); (void)crc;
	}
	return sinfo;
}

// FIXME: Need quanta for decompression. Store as a double in work and segments.
// Also add npy_type to work. Work also needs to know how big the output size will be.
// float32 → (flac int32, corrections int32) → (quanta32) → float32
// float64 → (flat int32, corrections int64) → (quanta64) → float64
// Since the corrections are stored as int64, we can't just go the whole way
// to quanta with an int32 buffer. So we have to special case everything.
//
// The decompression implementation in so3g looks pretty reasonable. Maybe there's
// not that much to win here. We'll see.

template <typename otype, typename itype> void copy_arr(void * odata, const void * idata, int64_t nvalue) {
	otype *optr = (otype*) odata;
	const itype *iptr = (itype*)idata;
	for(int64_t i = 0; i < nvalue; i++)
		optr[i] = (otype)iptr[i];
}

template <typename dtype> void add_arr(void * odata, const void * idata, int64_t nvalue) {
	dtype *optr = (dtype*) odata;
	const dtype *iptr = (dtype*)idata;
	for(int64_t i = 0; i < nvalue; i++)
		optr[i] += iptr[i];
}

template <typename dtype> void add_arr_val(void * odata, const void * idata, int64_t nvalue) {
	dtype *optr = (dtype*) odata;
	dtype val = *(dtype*)idata;
	for(int64_t i = 0; i < nvalue; i++)
		optr[i] += val;
}

template <typename itype, typename otype> void mul_arr_val(void * odata, const void * idata, otype val, int64_t nvalue) {
	// Does this casting run afoul of aliasing rules?
	otype *optr = (otype*) odata;
	itype *iptr = (itype*) idata;
	for(int64_t i = 0; i < nvalue; i++)
		optr[i] = val * iptr[i];
}

void decode_flac(const Work & work, Buffer & ibuf) {
	int64_t isize = ibuf.read<uint32_t>();
	// Decode into a temporary output buffer
	int nsamp = work.onbyte/work.itemsize;
	std::vector<int32_t> tmp(nsamp);
	FlacDecoder flac;
	if(flac.decode(ibuf.data+ibuf.pos, &tmp[0], isize, nsamp*4))
		throw std::runtime_error("Error decoding FLAC data");
	// Copy to target buffer why expanding to target precision
	if     (work.itemsize == 4) memcpy(work.obuf, &tmp[0], work.onbyte);
	else if(work.itemsize == 8) copy_arr<int64_t,int32_t>(work.obuf, &tmp[0], nsamp);
	else throw std::runtime_error("Only 32-bit and 64-bit supported");
	ibuf.pos += isize;
}

void add_offs_bz2(const Work & work, Buffer & ibuf) {
	int nsamp = work.onbyte/work.itemsize;
	unsigned int isize = (unsigned int) ibuf.read<uint32_t>();
	unsigned int osize = (unsigned int) work.onbyte;
	// Decode into a temporary buffer of the right size
	// Can save memory by setting the "small" (second to last)
	// argument to 1, but this halves the decompression speed
	// according to the documentation
	std::vector<char> tmp(work.onbyte);
	int err = BZ2_bzBuffToBuffDecompress(&tmp[0], &osize, (char*)ibuf.data+ibuf.pos, isize, 0, 0);
	if(err != BZ_OK) throw std::runtime_error("Error decoding BZ2 data");
	// Add to the actual output buffer
	if     (work.itemsize == 4) add_arr<int32_t>(work.obuf, &tmp[0], nsamp);
	else if(work.itemsize == 8) add_arr<int64_t>(work.obuf, &tmp[0], nsamp);
	else throw std::runtime_error("Only 32-bit and 64-bit supported");
	ibuf.pos += isize;
}

void add_offs_const(const Work & work, Buffer & ibuf) {
	int nsamp = work.onbyte/work.itemsize;
	// Read the value as raw bytes. This is big enough to hold both options
	if(work.itemsize > 8) throw std::runtime_error("Only 32-bit and 64-bit supported");
	char tmp[8];
	ibuf.read_raw(work.itemsize, tmp);
	// Then add it to our output buffer
	if     (work.itemsize == 4) add_arr_val<int32_t>(work.obuf, &tmp[0], nsamp);
	else if(work.itemsize == 8) add_arr_val<int64_t>(work.obuf, &tmp[0], nsamp);
	else throw std::runtime_error("Only 32-bit and 64-bit supported");
}

void add_quanta(const Work & work) {
	int nsamp = work.onbyte/work.itemsize;
	if     (work.npy_type == NPY_FLOAT32) mul_arr_val<int32_t,float >(work.obuf, work.obuf, work.quantum, nsamp);
	else if(work.npy_type == NPY_FLOAT64) mul_arr_val<int64_t,double>(work.obuf, work.obuf, work.quantum, nsamp);
}

// Process a single work item
void read_work(const Work & work) {
	// No compression. Simple memcpy
	if(work.algo == 0) memcpy(work.obuf, work.ibuf, work.onbyte);
	else {
		// First decode the data
		Buffer ibuf(work.ibuf, work.inbyte);
		if(work.algo & 1) decode_flac   (work, ibuf);
		if(work.algo & 2) add_offs_bz2  (work, ibuf);
		if(work.algo & 4) add_offs_const(work, ibuf);
		// Then handle the quanta
		add_quanta(work);
	}
}

void read_worklist(const WorkList & worklist) {
	int nwork = worklist.size();
	_Pragma("omp parallel for schedule(dynamic)")
	for(int wi = 0; wi < nwork; wi++)
		read_work(worklist[wi]);
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
	PyObject * meta = PyDict_New(); PyHandle delete_meta(meta);
	PyHandle ndet  = PyLong_FromUnsignedLong(sinfo.ndet); if(!ndet) return NULL;
	PyDict_SetItemString(meta, "ndet", ndet);
	PyHandle nsamp = PyLong_FromUnsignedLong(sinfo.nsamp);if(!nsamp)return NULL;
	PyDict_SetItemString(meta, "nsamp", nsamp);
	// Set up the detector names
	PyHandle detnames = PyList_New(sinfo.ndet);
	for(int32_t di = 0; di < sinfo.ndet; di++) {
		// No PyHandle needed since SET_ITEM steals the reference
		PyObject * detname = PyUnicode_FromString(sinfo.detnames[di].c_str()); if(!detname) return NULL;
		PyList_SET_ITEM(detnames.ptr, di, detname);
	}
	PyDict_SetItemString(meta, "detnames", detnames.ptr);
	// Set up the fields
	PyHandle fields = PyDict_New();
	for(const auto & [name, field] : sinfo.fields) {
		PyHandle dtype = PyArray_DescrFromType(field.npy_type);
		PyHandle shape = PyTuple_New(field.ndim);
		for(int32_t dim = 0; dim < field.ndim; dim++) {
			int64_t val = dim == 0 ? sinfo.nsamp : dim == 1 ? sinfo.ndet : 1;
			PyObject * pyval = PyLong_FromUnsignedLong(val); if(!pyval) return NULL;
			PyTuple_SET_ITEM(shape.ptr, field.ndim-1-dim, pyval); // steals
		}
		// [nseg,6] int64_t numpy array for the segments
		npy_intp segshape[2] = { (npy_intp)field.segments.size(), 7 };
		PyHandle segments = PyArray_ZEROS(2, segshape, NPY_INT64, 0);
		for(unsigned int si = 0; si < field.segments.size(); si++) {
			const Segment & seg = field.segments[si];
			int64_t * row = (int64_t*)PyArray_GETPTR2((PyArrayObject*)segments.ptr, si, 0);
			row[0] = seg.det;
			row[1] = seg.samp0;
			row[2] = seg.nsamp;
			row[3] = seg.buf0;
			row[4] = seg.nbyte;
			row[5] = *(int64_t*)&seg.quantum; // hack
			row[6] = seg.algo;
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
	// Looks like everything is done. Return our result after cancelling the meta cleanup
	delete_meta.ptr = NULL;
	return meta;
}


PyObject * read_py(PyObject * self, PyObject * args, PyObject * kwargs) {
	const char * bdata = NULL;
	Py_ssize_t bsize;
	PyObject *meta = NULL, *fields=NULL, *dets_in=NULL;
	PyHandle hdets;
	static const char * kwlist[] = {"buffer", "meta", "fields", "dets", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "y#OO|O", (char**)kwlist, &bdata, &bsize, &meta, &fields, &dets_in)) return NULL;
	// Extract field dict from meta. GetItem gives borrowed ref, so no PyHandle
	PyObject * meta_fields = PyDict_GetItemString(meta, "fields"); if(!fields)   return NULL;
	PyObject * obj_ndet    = PyDict_GetItemString(meta, "ndet");   if(!obj_ndet) return NULL;
	PyObject * obj_nsamp   = PyDict_GetItemString(meta, "nsamp");  if(!obj_nsamp)return NULL;
	int64_t ndet  = PyLong_AsLong(obj_ndet); if(PyErr_Occurred()) return NULL;
	//int64_t nsamp = PyLong_AsLong(obj_nsamp);if(PyErr_Occurred()) return NULL;
	// Detector list
	if(!dets_in || dets_in == Py_None) hdets.ptr = PyArray_Arange(0, ndet, 1, NPY_INT);
	else hdets.ptr = PyArray_ContiguousFromAny(dets_in, NPY_INT, 1, 1);
	if(!hdets) return NULL;
	// Make a mapping from the full detector list to the position in dets.
	// This will let us ensure the detectors are returned in the requested order
	std::unordered_map<int,int> detmap;
	npy_intp nodets = PyArray_DIM((PyArrayObject*)hdets.ptr,0);
	for(npy_intp di = 0; di < nodets; di++) {
		int det = *(int*)PyArray_GETPTR1((PyArrayObject*)hdets.ptr, di);
		detmap[det] = di;
	}
	// fields = {name:arr, name:arr,...}
	// Build the work array, which specifies all the information we need to decode
	// and copy each segment
	WorkList worklist;
	PyHandle iterator = PyObject_GetIter(fields); if(!iterator) return NULL;
	PyObject * name;
	while((name = PyIter_Next(iterator))) {
		PyHandle del_name(name);
		// Look up this in metadata
		PyObject * finfo  = PyDict_GetItem(meta_fields, name); if(!finfo) return NULL;
		PyObject * ishape = PyDict_GetItemString(finfo, "shape");if(!ishape)return NULL;
		long ndim = PyTuple_Size(ishape);
		PyArray_Descr * dtype = (PyArray_Descr*)PyDict_GetItemString(finfo, "dtype"); if(!dtype) return NULL;
		int npy_type      = dtype->type_num;
		int itemsize      = PyDataType_ELSIZE(dtype);
		// Get the output array. Python side must ensure this is
		// contiguous and has the right shape and dtype
		PyArrayObject * arr = (PyArrayObject*)PyDict_GetItem(fields, name); if(!arr)   return NULL;
		// Loop through the segments
		PyArrayObject * segments = (PyArrayObject*)PyDict_GetItemString(finfo, "segments"); if(!segments) return NULL;
		if(PyArray_NDIM(segments) != 2 || PyArray_TYPE(segments) != NPY_INT64 || PyArray_DIM(segments,1) != 7) return NULL;
		npy_intp nseg = PyArray_DIM(segments,0);
		for(npy_intp seg = 0; seg < nseg; seg++) {
			int64_t det   = *(int64_t*)PyArray_GETPTR2(segments, seg, 0);
			int64_t ind   = 0;
			if(det >= 0) {
				auto it = detmap.find(det);
				// Skip if this is an unwanted detector
				if(it == detmap.end()) continue;
				ind = it->second;
			}
			int64_t samp0 = *(int64_t*)PyArray_GETPTR2(segments, seg, 1);
			int64_t nsamp = *(int64_t*)PyArray_GETPTR2(segments, seg, 2);
			int64_t buf0  = *(int64_t*)PyArray_GETPTR2(segments, seg, 3);
			int64_t nbyte = *(int64_t*)PyArray_GETPTR2(segments, seg, 4);
			double quantum= *(double *)PyArray_GETPTR2(segments, seg, 5);
			uint8_t algo  = (uint8_t)*(int64_t*)PyArray_GETPTR2(segments, seg, 6);
			void * dest;
			if     (ndim == 1) dest = (void*)PyArray_GETPTR1(arr, samp0);
			else if(ndim == 2) dest = (void*)PyArray_GETPTR2(arr, ind, samp0);
			else return NULL;
			// This should be all we need to know
			Work work = { bdata+buf0, dest, nbyte, nsamp*itemsize, quantum, npy_type, itemsize, algo };
			worklist.push_back(work);
		}
	}
	// Phew! The work list is done. Do the actual work
	read_worklist(worklist);
	Py_RETURN_NONE;
}


PyDoc_STRVAR(scan__doc,
	"scan(buffer)\n--\n"
	"\n"
	"Scan the bytes from a .g3 file. Returning the metadata needed to\n"
	"read from it using read(bytes buffer, meta)\n"
);

PyDoc_STRVAR(read__doc,
	"read(buffer, meta, fields, dets=None)\n--\n"
	"\n"
	"Read the given fields from the .g3 bytes for the subset of\n"
	"detector indices specified, using the metadata provided by\n"
	"scan(buffer). Returns dictionary of numpy arrays.\n"
);

static PyMethodDef methods[] = {
	{"scan", scan_py, METH_VARARGS, scan__doc},
	{"read", (PyCFunction)read_py, METH_VARARGS|METH_KEYWORDS, read__doc},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
	PyModuleDef_HEAD_INIT,
	"fast_g3", NULL, -1, methods,
};

PyMODINIT_FUNC
PyInit_fast_g3_core(void)
{
	PyObject *module = PyModule_Create(&module_def);
	if (module == NULL) return NULL;
	import_array();
	return module;
}

}
