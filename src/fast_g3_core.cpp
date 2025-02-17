#include <stdint.h>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h> // memset
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <FLAC/stream_decoder.h>
#include <bzlib.h>
#include <aio.h>
#include <memory>

#if NPY_ABI_VERSION < 0x02000000
  #define PyDataType_ELSIZE(descr) ((descr)->elsize)
#endif

// Need to make some changes with how the detectors are handled.
// I thought any 2-axis array would have detectors as the first
// axis, but that's wrong. Only supertimestreams have two axes,
// and each has their own names. This information should therefore
// not be attached to ScanInfo, but instead to the field, which
// will now have a names array. We will still assume this is
// attached to the first axis.

// Low-level python interface consists of two functions
// * scan(buffer), which returns an info dictionary
//   {ndet:int, nsamp:int, detnames:[], fields:{
//      shape:[], dtype:, segments:[]}}
// * read(buffer, info, fields={name:{oarr, dets=None}}), which
//   reads the given detectors into the provided output arrays
//   for the specified fields. The interface works like this
//   to all the reads to happen as much in parallel as possible.

// We also need the scalars from the individual frames. We don't know
// what these do or how they would combine in general, so we can just
// return them as a list of frames, each of which has a type and list
// of key,value pairs, excluding the fields we already handled.
// It probably makes the most sense to include this as part of meta.
//
// Add a member Frame[] frames to ScanInfo,
// where Frame = { int32_t type; Entry[] entries; } and
// Entry = { string name; int64_t buf0, nbyte; enum etype; };
// This would then be translated to Python objects later.

// How to handle reading only some sub-range of the samples?
// [------------------- file1 --------------][ ----------------- file2 ---------------][...]
// [-- frame1 --][-- frame2 --][-- frame3 --][-- frame1 --][-- frame2 --][-- frame3 --][...]
//                      [--------- samprange --------]
// In terms of global samples (at the python level):
//  * For each file, check if we need anything
//    If samprange[0] >= file_end or samprange[1] < file_start, then skip the file
//  * Define file-relative samprange: fsamprange = samprange - file_start
//  * For each frame in file, check if we need anything
//    If fsamprange[0] >= frame_end = segment.samp0+segment.nsamp
//    or fsamprange[1] < frame_start = segment.samp0, then skip the segment
//  * Work needs to be modified in two ways:
//    1. obuf was &arr[samp0], but should be &arr[samp0-fsamprange[0]]
//    2. Work-processing code needs to know the segment-relative range:
//       ssamprange = fsamprange - frame_start
//    Work-processing will then skip ssamprange[0] samples, and then
//    copy out ssamprange[1]-ssamprange[0] samples

using std::string;
using std::vector;
enum class FType { None, G3Int, G3Double, G3String, G3Time, G3VectorInt8, G3VectorInt16, G3VectorInt32, G3VectorInt64, G3VectorDouble, G3VectorBool, G3TimesampleMap, G3SuperTimestream };
struct Segment     { int64_t row, samp0, nsamp, buf0, nbyte; double quantum; uint8_t algo; };
struct SimpleField { int64_t buf0, nbyte; string name; FType type; };
typedef vector<Segment> Segments;
typedef vector<string> Names;
struct Field    { int npy_type, ndim; Names names; Segments segments; };
typedef vector<SimpleField> SimpleFields;
struct Frame    { int32_t type; SimpleFields simple_fields; };
typedef vector<Frame> Frames;
typedef std::unordered_map<string,Field> Fields;
struct ScanInfo {
	ScanInfo():nsamp(0),curnsamp(0) {}
	int64_t nsamp, curnsamp; Fields fields; Frames frames;
};
struct Work { const void *ibuf; void *obuf; int64_t inbyte, onbyte, oskip; double quantum; int npy_type, itemsize; uint8_t algo; };
typedef vector<Work> WorkList;
// Used for async reads
struct AioTask {
	AioTask(FILE * file=NULL, aiocb * ao=NULL):file(file),ao(ao) {}
	~AioTask() { if(file) fclose(file); if(ao) delete ao; }
	FILE * file; aiocb * ao;
};

// printf that returns a new string
string fmt(const char * format, ...) {
	// First figure out how big the string needs to be
	va_list ap;
	size_t size = 0;
	va_start(ap, format);
	int n = vsnprintf(NULL, size, format, ap);
	va_end(ap);
	// Make a big enough string
	string msg(n, '\0');
	// And fill it
	va_start(ap, format);
	vsnprintf(msg.data(), n, format, ap);
	va_end(ap);
	return msg;
}

// Used to ensure python resources are freed.
// Set ptr to NULL if you change your mind
// This class should have just used reference counting
// itself, with a copy constructor. Oh well.
struct PyHandle {
	// This is a bit heavy-handed. We only need it for the Python object types
	PyHandle():ptr(NULL) {}
	template <typename T> PyHandle(T * ptr):ptr((PyObject*)ptr) {}
	~PyHandle() { Py_XDECREF(ptr); }
	operator PyObject*() { return ptr; }
	operator bool()      { return ptr != NULL; }
	PyHandle & inc() { Py_XINCREF(ptr); return *this; }
	PyHandle & dec() { Py_XDECREF(ptr); return *this; }
	PyObject * defuse() { PyObject * ret = ptr; ptr = NULL; return ret; }
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
	const void * idata;   // input buffer
	void * odata;         // output buffer
	int64_t ipos, isize;  // input offset and size
	int64_t opos, osize;  // output offset and size. Can be smaller than full stream
	int64_t oskip;        // number of output bytes to skip at the beginning
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
	// Sample range we want
	int64_t ostart = info->oskip;
	int64_t oend   = info->oskip+info->osize;
	// Subset of this we have in this chunk
	int64_t cstart = std::max(ostart, info->opos);
	int64_t cend   = std::min(oend,   info->opos+n);
	// Bytes we will handle for this chunk
	int64_t ncopy  = cend-cstart;
	if(ncopy > 0) memcpy((char*)info->odata+(cstart-info->oskip), (const char*)buffer[0]+(cstart-info->opos), ncopy);
	info->opos += n;
	// Stop if we're past the end. This isn't necessarily
	// an error, so caller should check info->opos
	if(info->opos >= oend) return FLAC__STREAM_DECODER_WRITE_STATUS_ABORT;
	else return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}
void FlacErrBuf(const FLAC__StreamDecoder *decoder, FLAC__StreamDecoderErrorStatus status, void *client_data) {
	fprintf(stderr, "FLAC error: %d\n", status);
}

struct FlacDecoder {
	FlacDecoder()  { decoder = FLAC__stream_decoder_new(); }
	~FlacDecoder() { FLAC__stream_decoder_delete(decoder); }
	int decode(const void * idata, void * odata, int64_t isize, int64_t osize, int64_t ostart = 0) {
		FlacHelper info = { idata, odata, 0, isize, 0, osize, ostart };
		int err = 0;
		if((err=FLAC__stream_decoder_init_stream(decoder, FlacReadBuf, FlacSeekBuf,
				FlacTellBuf, FlacLenBuf, FlacEofBuf, FlacWriteBuf, NULL, FlacErrBuf,
				(void*)&info))!=FLAC__STREAM_DECODER_INIT_STATUS_OK) return 1;
		if(!(err=FLAC__stream_decoder_process_until_end_of_stream(decoder))) {
			// We can end up here if we aborted earlier. Not an error if we got enough data
			if(info.opos < info.osize) return 2;
		}
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
bool contains(PyObject * obj, const char * name) {
	PyHandle pyname = PyUnicode_FromString(name);
	return PyDict_Contains(obj, pyname);
}
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
			seg.row   = -1;
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
	// Times are int64 ctimes in units of 10 ns
	// The -4,+4 is because the decoder considers the starting length tag to be
	// part of the input buffer
	Segment tseg = { -1, sinfo.nsamp, nsamp, buf.pos-4, times_nbyte_comp+4, 0.0, times_algo };
	get_field(sinfo.fields, name+"/times", NPY_INT64, 1).segments.push_back(tseg);
	buf.pos += times_nbyte_comp;
	buf.pos += 4; // skip names header
	int64_t nname            = buf.read<int64_t>();
	// Loop through the row names
	Names names(nname);
	for(int32_t ni = 0; ni < nname; ni++) {
		int64_t nlen = buf.read<int64_t>();
		names[ni] = buf.read_string(nlen);
	}
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
	vector<double> quanta(nquanta);
	if(nquanta > 0)
		buf.read_raw(8*nquanta, &quanta[0]);
	int64_t noffset          = buf.read<int64_t>();
	vector<int32_t> offsets(noffset);
	buf.read_raw(noffset*4, &offsets[0]);
	int32_t signal_nbyte_comp= buf.read<int32_t>();

	sinfo.curnsamp = nsamp;
	Field & field = get_field(sinfo.fields, name + "/data", npy_type, 2);
	if(field.names.empty()) field.names.swap(names);
	(void)ndim;

	// Read the row data
	Buffer dbuf(buf);

	int32_t none_size = 0, flac_size = 0, bz_size = 0, const_size = 0, tot_size = 0;
	ssize_t row_start, row_end;
	for(int32_t ni = 0; ni < nname; ni++) {
		uint8_t used_algo = dbuf.read<uint8_t>();
		row_start = dbuf.pos;
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
		row_end   = dbuf.pos;
		tot_size = none_size + flac_size + bz_size + const_size; (void)tot_size;
		// Ok, done parsing this row
		double quantum = nquanta > 0 ? quanta[ni] : 1.0;
		Segment seg = { ni, sinfo.nsamp, nsamp, row_start, row_end-row_start, quantum, used_algo };
		field.segments.push_back(seg);
	}
	buf.pos += signal_nbyte_comp;
}

void process_fields(Buffer & buf, int32_t nfield, ScanInfo & sinfo, Frame & frame) {
	for(int fi = 0; fi < nfield; fi++) {
		int64_t field_name_len = buf.read<int64_t>();
		string field_name       = buf.read_string(field_name_len);
		int64_t val_len        = buf.read<int64_t>();
		// Handle the value
		Buffer subbuf(buf);
		subbuf.pos += 5;
		int64_t type_name_len  = subbuf.read<int64_t>();
		string type_name        = subbuf.read_string(type_name_len);
		FType ftype = FType::None;
		int64_t nbyte;
		// First the cases we handle specially. These have samples as
		// their last axis, and will be concatenated into big arrays.
		if(type_name == "G3TimesampleMap") {
			subbuf.pos += 16;
			int32_t subfields = subbuf.read<int32_t>();
			process_timesamplemap(subbuf, subfields, field_name + "/", sinfo);
		} else if(type_name == "G3SuperTimestream") {
			process_supertimestream(subbuf, field_name, sinfo);
		} else {
			// Remaning field types here
			subbuf.pos += 12;
			if     (type_name == "G3Int")   { ftype = FType::G3Int;    nbyte = 8; }
			else if(type_name == "G3Time")  { ftype = FType::G3Time;   nbyte = 8; }
			else if(type_name == "G3Double"){ ftype = FType::G3Double; nbyte = 8; }
			else if(type_name == "G3String"){ ftype = FType::G3String; nbyte = subbuf.read<int64_t>(); }
			else if(type_name == "G3VectorDouble") { ftype = FType::G3VectorDouble; nbyte = subbuf.read<int64_t>()*8; }
			else if(type_name == "G3VectorInt") {
				int32_t itemsize = subbuf.read<int32_t>()/8;
				nbyte = subbuf.read<int64_t>()*itemsize;
				if     (itemsize == 1) ftype = FType::G3VectorInt8;
				else if(itemsize == 2) ftype = FType::G3VectorInt16;
				else if(itemsize == 4) ftype = FType::G3VectorInt32;
				else if(itemsize == 8) ftype = FType::G3VectorInt64;
				else {
					// Invalid field. Ignore for now
				}
			}
			else if(type_name == "G3VectorBool") { ftype = FType::G3VectorBool; nbyte = subbuf.read<int64_t>()*1; }
			else {
				// Unknown type. Just ignore for now
			}
		}
		buf.pos += val_len;
		// Append non-sample fields to frame
		if(ftype != FType::None) {
			SimpleField simple_field = { subbuf.pos, nbyte, field_name, ftype };
			frame.simple_fields.push_back(simple_field);
		}
	}
}

ScanInfo scan(Buffer & buf, int32_t until_frame_type = -1) {
	ScanInfo sinfo;
	// Loop over frames in file
	for(int framei = 0; buf.pos < buf.size; framei++) {
		Frame frame; // holds all frame data we don't extract into Fields
		uint8_t foo1    = buf.read<uint8_t>(); (void)foo1;
		int32_t version = buf.read<int32_t>(); (void)version;
		int32_t nfield  = buf.read<int32_t>(); (void)nfield;
		frame.type      = buf.read<int32_t>();
		if(frame.type == until_frame_type) break;
		process_fields(buf, nfield, sinfo, frame);
		sinfo.frames.push_back(frame);
		sinfo.nsamp += sinfo.curnsamp;
		int32_t crc     = buf.read<int32_t>(); (void)crc;
	}
	return sinfo;
}

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
	int nskip = work.oskip /work.itemsize;
	vector<int32_t> tmp(nsamp);
	FlacDecoder flac;
	if(flac.decode(ibuf.data+ibuf.pos, &tmp[0], isize, nsamp*4, nskip*4))
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
	unsigned int otot  = (unsigned int) (work.onbyte+work.oskip);
	// Decode into a temporary buffer of the right size
	// Can save memory by setting the "small" (second to last)
	// argument to 1, but this halves the decompression speed
	// according to the documentation
	vector<char> tmp(otot);
	int err = BZ2_bzBuffToBuffDecompress(&tmp[0], &otot, (char*)ibuf.data+ibuf.pos, isize, 0, 0);
	if(err != BZ_OK && err != BZ_OUTBUFF_FULL) throw std::runtime_error("Error decoding BZ2 data");
	// Add to the actual output buffer
	if     (work.itemsize == 4) add_arr<int32_t>(work.obuf, &tmp[work.oskip], nsamp);
	else if(work.itemsize == 8) add_arr<int64_t>(work.obuf, &tmp[work.oskip], nsamp);
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

void zero_buffer(const Work & work) {
	memset(work.obuf, 0, work.onbyte);
}

// Process a single work item
void read_work(const Work & work) {
	// No compression. Simple memcpy
	if(work.algo == 0) memcpy(work.obuf, (char*)work.ibuf+work.oskip, work.onbyte);
	else {
		// First decode the data
		// bz2 and const usually add to the result from flac,
		// but in a few cases we start directly from zero instead
		Buffer ibuf(work.ibuf, work.inbyte);
		if(work.algo & 1) decode_flac   (work, ibuf);
		else              zero_buffer   (work);
		if(work.algo & 2) add_offs_bz2  (work, ibuf);
		if(work.algo & 4) add_offs_const(work, ibuf);
		// Then handle the quanta
		add_quanta(work);
	}
}

void read_worklist(const WorkList & worklist) {
	int nwork = worklist.size();
	int nerr  = 0;
	_Pragma("omp parallel for schedule(dynamic) reduction(+:nerr)")
	for(int wi = 0; wi < nwork; wi++) {
		try { read_work(worklist[wi]); }
		catch (const std::exception & e) { nerr ++; }
	}
	if(nerr > 0) throw std::runtime_error("extract error in read_worklist");
}

// Asynchronous read stuff
std::shared_ptr<AioTask> start_async_read(const char * fname, char * obuf, ssize_t obytes) {
	std::shared_ptr<AioTask> task = std::make_shared<AioTask>();
	// Open file for reading
	task->file = fopen(fname, "rb");
	if(!task->file) throw std::runtime_error(fmt("Error opening file '%s'", fname));
	// Set up task
	aiocb * ao = new aiocb();
	ao->aio_fildes = fileno(task->file);
	ao->aio_offset = 0;
	ao->aio_buf    = obuf;
	ao->aio_nbytes = obytes;
	task->ao = ao;
	// Start reading
	int status = aio_read(task->ao);
	if(status)
		throw std::runtime_error(fmt("Error %d in aio_read for '%s'", status, fname));
	return task;
}

int end_async_read(AioTask & task) {
	int status = aio_suspend(&task.ao, 1, NULL);
	// Regardless of status we should close the resources
	fclose(task.file);
	delete task.ao;
	task.file = NULL;
	task.ao   = NULL;
	return status;
}

const char * get_frame_type_name(int32_t code) {
	switch(code) {
		case 'T': return "timepoint";
		case 'H': return "housekeeping";
		case 'O': return "observation";
		case 'S': return "scan";
		case 'M': return "map";
		case 'I': return "instrumentationstatus";
		case 'W': return "wiring";
		case 'C': return "calibration";
		case 'G': return "gcpslow";
		case 'P': return "pipelineinfo";
		case 'E': return "ephemeris";
		case 'L': return "lightcurve";
		case 'R': return "statistics";
		case 'Z': return "endprocessing";
		case 'N': return "none";
		default: return "unknown";
	}
}

PyObject * array_copy_from_bytes(npy_intp nitem, int typenum, const void * ptr, int64_t nbyte) {
	return PyArray_New(&PyArray_Type, 1, &nitem, typenum, NULL, (void*)ptr, 0, NPY_ARRAY_ENSURECOPY, NULL);
	//PyObject * arr = PyArray_EMPTY(1, &nitem, typenum, 0); if(!arr) return NULL;
	//memcpy(PyArray_GETPTR1((PyArrayObject*)arr,0), ptr, nbyte);
	//return arr;
}

// Expand sinfo.frames into python objects
PyObject * expand_simple(const Frames & frames, Buffer buf) {
	PyHandle pframes = PyList_New(frames.size()); if(!pframes) return NULL;
	for(size_t fi = 0; fi < frames.size(); fi++) {
		const Frame & frame = frames[fi];
		// Make dict to hold entries in this frame. Stolen when added
		PyObject * pframe = PyDict_New(); if(!pframe) return NULL;
		PyList_SET_ITEM(pframes.ptr, fi, pframe);
		PyHandle pcode = PyLong_FromLong(frame.type); if(!pcode) return NULL;
		PyHandle pname = PyUnicode_FromString(get_frame_type_name(frame.type)); if(!pname) return NULL;
		PyDict_SetItemString(pframe, "code", pcode); if(PyErr_Occurred()) return NULL;
		PyDict_SetItemString(pframe, "type", pname); if(PyErr_Occurred()) return NULL;
		// Set up the field info, which will be a subdict
		PyHandle pfields = PyDict_New(); if(!pfields) return NULL;
		PyDict_SetItemString(pframe, "fields", pfields); if(PyErr_Occurred()) return NULL;
		for (const SimpleField & field : frame.simple_fields) {
			PyHandle fname = PyUnicode_FromString(field.name.c_str()); if(!fname) return NULL;
			// Ok, time to handle the different types!
			const char * vptr = &buf.data[field.buf0];
			PyHandle value;
			switch(field.type) {
				case FType::G3Int:
				case FType::G3Time:
					value.ptr = PyLong_FromLong(*(int64_t*)vptr); break;
				case FType::G3Double:
					value.ptr = PyFloat_FromDouble(*(double*)vptr); break;
				case FType::G3String:
					value.ptr = PyUnicode_FromStringAndSize(vptr, field.nbyte); break;
				case FType::G3VectorInt8:
					value.ptr = array_copy_from_bytes(field.nbyte, NPY_INT8, vptr, field.nbyte);
					break;
				case FType::G3VectorInt16:
					value.ptr = array_copy_from_bytes(field.nbyte/2, NPY_INT16, vptr, field.nbyte);
					break;
				case FType::G3VectorInt32:
					value.ptr = array_copy_from_bytes(field.nbyte/4, NPY_INT32, vptr, field.nbyte);
					break;
				case FType::G3VectorInt64:
					value.ptr = array_copy_from_bytes(field.nbyte/8, NPY_INT64, vptr, field.nbyte);
					break;
				case FType::G3VectorDouble:
					value.ptr = array_copy_from_bytes(field.nbyte/8, NPY_FLOAT64, vptr, field.nbyte);
					break;
				case FType::G3VectorBool:
					value.ptr = array_copy_from_bytes(field.nbyte, NPY_BOOL, vptr, field.nbyte);
					break;
				default:
					// Skip unknown
					continue;
			}
			if(!value) return NULL;
			PyDict_SetItem(pfields, fname, value); if(PyErr_Occurred()) return NULL;
		}
	}
	return pframes.defuse();
}

extern "C" {

static PyObject * scan_py(PyObject * self, PyObject * args, PyObject * kwargs) {
	Py_buffer pybuf;
	int32_t until_frame_type = -1;
	static const char * kwlist[] = {"buffer", "until_frame_type", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "y*|i", (char**)kwlist, &pybuf, &until_frame_type)) return NULL;
	// Scan the file
	Buffer buf(pybuf.buf, pybuf.len);
	ScanInfo sinfo;
	try { sinfo = scan(buf, until_frame_type); }
	catch (const std::exception & e) {
		PyErr_SetString(PyExc_IOError, e.what());
		return NULL;
	}
	// Convert to python types
	PyHandle meta = PyDict_New();
	PyHandle nsamp = PyLong_FromUnsignedLong(sinfo.nsamp);if(!nsamp)return NULL;
	PyDict_SetItemString(meta, "nsamp", nsamp);
	// Set up the fields
	PyHandle fields = PyDict_New();
	for(const auto & [name, field] : sinfo.fields) {
		PyHandle dtype = PyArray_DescrFromType(field.npy_type);
		// Set up the shape
		int nname = field.names.size();
		PyHandle shape = PyTuple_New(field.ndim);
		for(int32_t dim = 0; dim < field.ndim; dim++) {
			int64_t val = dim == 0 ? sinfo.nsamp : dim == 1 ? nname : 1;
			PyObject * pyval = PyLong_FromUnsignedLong(val); if(!pyval) return NULL;
			PyTuple_SET_ITEM(shape.ptr, field.ndim-1-dim, pyval); // steals
		}
		// Set up the name list
		PyHandle names = PyTuple_New(nname);
		for(int ni = 0; ni < nname; ni++) {
			PyObject * pyname = PyUnicode_FromString(field.names[ni].c_str()); if(!pyname) return NULL;
			PyTuple_SET_ITEM(names.ptr, ni, pyname); // steals
		}
		// [nseg,6] int64_t numpy array for the segments
		npy_intp segshape[2] = { (npy_intp)field.segments.size(), 7 };
		PyHandle segments = PyArray_ZEROS(2, segshape, NPY_INT64, 0);
		for(unsigned int si = 0; si < field.segments.size(); si++) {
			const Segment & seg = field.segments[si];
			int64_t * row = (int64_t*)PyArray_GETPTR2((PyArrayObject*)segments.ptr, si, 0);
			row[0] = seg.row;
			row[1] = seg.samp0;
			row[2] = seg.nsamp;
			row[3] = seg.buf0;
			row[4] = seg.nbyte;
			// hack: store the float64 quantum in the int field
			memcpy(&row[5], &seg.quantum, 8);
			row[6] = seg.algo;
		}
		// Make field dict to add these to
		PyHandle pyfield = PyDict_New();
		PyDict_SetItemString(pyfield, "dtype", dtype);
		PyDict_SetItemString(pyfield, "shape", shape);
		PyDict_SetItemString(pyfield, "names", names);
		PyDict_SetItemString(pyfield, "segments", segments);
		// And add this field to the fields list
		PyDict_SetItemString(fields, name.c_str(), pyfield);
	}
	PyDict_SetItemString(meta, "fields", fields);
	// Add the simple fields from the frames too
	PyHandle frames = expand_simple(sinfo.frames, buf); if(!frames) return NULL;
	PyDict_SetItemString(meta, "frames", frames);
	// Everything looks ok
	return meta.defuse();
}


PyObject * extract_py(PyObject * self, PyObject * args, PyObject * kwargs) {
	Py_buffer pybuf;
	PyObject *meta = NULL, *fields=NULL;
	static const char * kwlist[] = {"buffer", "meta", "fields", NULL};
	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "y*OO", (char**)kwlist, &pybuf, &meta, &fields)) return NULL;
	// Extract field dict from meta. GetItem gives borrowed ref, so no PyHandle
	PyObject * meta_fields = PyDict_GetItemString(meta, "fields"); if(!fields)   return NULL;
	PyObject * obj_nsamp   = PyDict_GetItemString(meta, "nsamp");  if(!obj_nsamp)return NULL;
	// fields = {name:{oarr:arr,rows:rows},...}
	// Build the work array, which specifies all the information we need to decode
	// and copy each segment
	WorkList worklist;
	PyHandle iterator = PyObject_GetIter(fields); if(!iterator) return NULL;
	PyObject * name;
	while((name = PyIter_Next(iterator))) {
		PyHandle del_name(name);
		// Look up this in metadata
		PyObject * finfo  = PyDict_GetItem(meta_fields, name);   if(!finfo) return NULL;
		PyObject * inames = PyDict_GetItemString(finfo, "names");if(!inames)return NULL;
		PyObject * ishape = PyDict_GetItemString(finfo, "shape");if(!ishape)return NULL;
		long ndim  = PyObject_Size(ishape); if(PyErr_Occurred()) return NULL;
		int  nname = PyObject_Size(inames); if(PyErr_Occurred()) return NULL;
		PyArray_Descr * dtype = (PyArray_Descr*)PyDict_GetItemString(finfo, "dtype"); if(!dtype) return NULL;
		int npy_type      = dtype->type_num;
		int itemsize      = PyDataType_ELSIZE(dtype);
		// fields will now be {name:{oarr:,(optional)rows:}}
		PyObject * oinfo  = PyDict_GetItem(fields, name); if(!oinfo) return NULL;
		// Selected rows
		PyHandle hrows;
		if(!contains(oinfo, "rows") || PyDict_GetItemString(oinfo, "rows") == Py_None)
			hrows.ptr = PyArray_Arange(0, nname, 1, NPY_INT);
		else hrows.ptr = PyArray_ContiguousFromAny(PyDict_GetItemString(oinfo, "rows"), NPY_INT, 1, 1);
		if(!hrows) return NULL;
		// Make a mapping from the full set of rows to the ones we ask for.
		// This will let us ensure the values are returned in the requested order
		std::unordered_map<int,int> rowmap;
		npy_intp nrow = PyArray_DIM((PyArrayObject*)hrows.ptr,0);
		for(npy_intp ri = 0; ri < nrow; ri++) {
			int row = *(int*)PyArray_GETPTR1((PyArrayObject*)hrows.ptr, ri);
			rowmap[row] = ri;
		}
		// Get the output array. Python side must ensure this is
		// contiguous and has the right shape and dtype
		PyArrayObject * arr = (PyArrayObject*)PyDict_GetItemString(oinfo, "oarr"); if(!arr)   return NULL;
		// We may want to read out just a sub-range [start:end] of the full
		// number of samples. We assume that oarr.shape[-1] = end-start, so
		// we just need an extra parameter to encode the start. We will call it
		// "skip". We allow "skip" to be missing
		int64_t ostart = 0;
		PyObject * obj_skip = PyDict_GetItemString(oinfo, "skip");
		if(obj_skip) {
			ostart = PyLong_AsLong(obj_skip); if(PyErr_Occurred()) return NULL;
		} else PyErr_Clear();
		int64_t oend = ostart + PyArray_SHAPE(arr)[PyArray_NDIM(arr)-1];
		// Loop through the segments
		PyArrayObject * segments = (PyArrayObject*)PyDict_GetItemString(finfo, "segments"); if(!segments) return NULL;
		if(PyArray_NDIM(segments) != 2 || PyArray_TYPE(segments) != NPY_INT64 || PyArray_DIM(segments,1) != 7) return NULL;
		npy_intp nseg = PyArray_DIM(segments,0);
		for(npy_intp seg = 0; seg < nseg; seg++) {
			int64_t row  = *(int64_t*)PyArray_GETPTR2(segments, seg, 0);
			int64_t ind  = 0;
			if(row >= 0) {
				auto it = rowmap.find(row);
				// Skip if this is an unwanted row
				if(it == rowmap.end()) continue;
				ind = it->second;
			}
			int64_t samp0 = *(int64_t*)PyArray_GETPTR2(segments, seg, 1);
			int64_t nsamp = *(int64_t*)PyArray_GETPTR2(segments, seg, 2);
			int64_t buf0  = *(int64_t*)PyArray_GETPTR2(segments, seg, 3);
			int64_t nbyte = *(int64_t*)PyArray_GETPTR2(segments, seg, 4);
			double quantum= *(double *)PyArray_GETPTR2(segments, seg, 5);
			uint8_t algo  = (uint8_t)*(int64_t*)PyArray_GETPTR2(segments, seg, 6);
			// Calculate the global sample range we want from this segment
			int64_t sstart    = std::max(samp0,      ostart);
			int64_t send      = std::min(samp0+nsamp,oend  );
			int64_t nsamp_get =  send-sstart;
			// Skip the segment if we don't want any of it
			if(nsamp_get > 0) {
				void * dest;
				if     (ndim == 1) dest = (void*)PyArray_GETPTR1(arr, sstart-ostart);
				else if(ndim == 2) dest = (void*)PyArray_GETPTR2(arr, ind, sstart-ostart);
				else return NULL;
				// This should be all we need to know
				Work work = { (char*)pybuf.buf+buf0, dest, nbyte, nsamp_get*itemsize, (sstart-samp0)*itemsize, quantum, npy_type, itemsize, algo };
				worklist.push_back(work);
			}
		}
	}
	// Phew! The work list is done. Do the actual work
	try { read_worklist(worklist); }
	catch (const std::exception & e) {
		PyErr_SetString(PyExc_IOError, e.what());
		return NULL;
	}
	Py_RETURN_NONE;
}

static PyObject * start_async_read_py(PyObject * self, PyObject * args) {
	const char * fname;
	Py_buffer pybuf;
	if(!PyArg_ParseTuple(args, "sy*", &fname, &pybuf)) return NULL;
	std::shared_ptr<AioTask> task;
	try { task = start_async_read(fname, (char*)pybuf.buf, pybuf.len); }
	catch(const std::exception & e) {
		PyErr_SetString(PyExc_IOError, e.what());
		return NULL;
	}
	// Return FILE* and ao* as tuple members
	PyObject * ret    = PyTuple_New(2);
	PyHandle delete_ret = ret;
	PyObject * pyfile = PyLong_FromVoidPtr(task->file); if(!pyfile) return NULL;
	PyTuple_SET_ITEM(ret, 0, pyfile);
	PyObject * pyao   = PyLong_FromVoidPtr(task->ao);   if(!pyao)   return NULL;
	PyTuple_SET_ITEM(ret, 1, pyao);
	// Now that python know about the file and ao, stop shared_ptr from
	// deleting them when it goes out of scope
	task->file = NULL;
	task->ao   = NULL;
	delete_ret.ptr = NULL;
	return ret;
}

static PyObject * end_async_read_py(PyObject * self, PyObject * args) {
	PyObject * info;
	if(!PyArg_ParseTuple(args, "O", &info)) return NULL;
	PyObject * pyfile = PyTuple_GetItem(info, 0); if(!pyfile) return NULL;
	PyObject * pyao   = PyTuple_GetItem(info, 1); if(!pyao)   return NULL;
	FILE  * file = (FILE *)PyLong_AsVoidPtr(pyfile); if(PyErr_Occurred()) return NULL;
	aiocb * ao   = (aiocb*)PyLong_AsVoidPtr(pyao);   if(PyErr_Occurred()) return NULL;
	AioTask task(file, ao);
	int status = end_async_read(task);
	if(status) {
		PyErr_SetString(PyExc_IOError, fmt("Async read error %d", status).c_str());
		return NULL;
	}
	return Py_None;
}

PyDoc_STRVAR(scan__doc,
	"scan(buffer)\n--\n"
	"\n"
	"Scan the bytes from a .g3 file. Returning the metadata needed to\n"
	"read from it using read(bytes buffer, meta)\n"
);

PyDoc_STRVAR(read__doc,
	"extract(buffer, meta, fields)\n--\n"
	"\n"
	"Extract the given fields from the .g3 bytes for the subset of\n"
	"detector indices specified, using the metadata provided by\n"
	"scan(buffer). fields is a dictionary of field_name:{\n"
	"\"oarr\":output_array,\"rows\":rows}, where the output array\n"
	"must have the right shape and dtype. The optional rows entry\n"
	"contains a list of rows to ask for for 2d fields, and the shape\n"
	"of the output array must reflect the number of rows requested\n"
);

PyDoc_STRVAR(start_async_read__doc,
	"start_async_read(fname, buffer)\n--\n"
	"\n"
	"Start an asynchronous read of the named file into the given\n"
	"buffer, which must be a bytes object with the correct length\n"
	"Returns an async task object that must be passed to\n"
	"end_async_read() to free up resources.\n"
);

PyDoc_STRVAR(end_async_read__doc,
	"end_async_read(async_task)\n--\n"
	"\n"
	"Wait for an asynchronous read started with start_async_read\n"
	"to finish, and free up resources when it does\n"
);

static PyMethodDef methods[] = {
	{"scan", (PyCFunction)scan_py, METH_VARARGS|METH_KEYWORDS, scan__doc},
	{"extract", (PyCFunction)extract_py, METH_VARARGS|METH_KEYWORDS, read__doc},
	{"start_async_read", start_async_read_py, METH_VARARGS, start_async_read__doc},
	{"end_async_read", end_async_read_py, METH_VARARGS, end_async_read__doc},
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
