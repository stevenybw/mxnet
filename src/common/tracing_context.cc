/*
 * tracing_context.cc
 *
 *  Created on: 2016年11月22日
 *      Author: Bowen Yu <yubowen15@foxmail.com>
 */

#include <dmlc/logging.h>
#include "tracing_context.h"

#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>


namespace mxnet {
namespace common {

#define WRITE_ITEM(TYPE, NAME) do { *((TYPE*)(_buf + offset)) = (NAME); offset += sizeof(TYPE); } while(0)
#define MAX_NUM_THREADS 1024

struct alignas(64) Serializer {
  char* _buf;
  std::ofstream* _fp;
  string _file_path;
  size_t offset;

  Serializer() : _buf(NULL), _fp(NULL), offset(0) {
  }

  virtual ~Serializer() {
    if(_buf != NULL) {
      FILE* fp;
      fp = fopen(_file_path.c_str(), "wb");
      fwrite(_buf, sizeof(char), offset, fp);
      fclose(fp);

      _fp->close();
      delete _fp;
      _fp = NULL;
      LOG(INFO) << "Successfully written to " << _file_path;
    }
  }

  void open(const char* file_path, size_t size) {
    _buf = new char[size];
    _file_path = file_path;
  }

  void write(double timestamp, int8_t ev, int8_t task_id, int32_t step_id, int8_t partition_id, int32_t node_id, uint64_t frame_id, int64_t input_iter) {
    WRITE_ITEM(double, timestamp);
    WRITE_ITEM(int8_t, ev);
    WRITE_ITEM(int8_t, task_id);
    WRITE_ITEM(int32_t, step_id);
    WRITE_ITEM(int8_t, partition_id);
    WRITE_ITEM(int32_t, node_id);
    WRITE_ITEM(uint64_t, frame_id);
    WRITE_ITEM(int64_t, input_iter);
  }

  void write(double timestamp, int8_t ev, int8_t task_id, int32_t step_id, int8_t partition_id, int32_t node_id) {
    WRITE_ITEM(double, timestamp);
    WRITE_ITEM(int8_t, ev);
    WRITE_ITEM(int8_t, task_id);
    WRITE_ITEM(int32_t, step_id);
    WRITE_ITEM(int8_t, partition_id);
    WRITE_ITEM(int32_t, node_id);
  }
};

static std::atomic_int_fast64_t _num_threads(0);
static thread_local int64_t _tid = -1;

// this is a cached trace writer for current thread
static thread_local Serializer* _trace_writer;

static Serializer _trace_writers[MAX_NUM_THREADS];

static double currentTimeMillisecond() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return 1e6 * tv.tv_sec + 1.0 * tv.tv_usec;
}

void TracingContext::init_thread_if_necessary() {
  if (_tid == -1) {
    _tid = _num_threads.fetch_add(1);

    char buf[256];
    sprintf(buf, "%s.trace.%ld", _trace_path.c_str(), _tid);
    _trace_writer = _trace_writers + _tid;
    _trace_writer->open(buf, 256*1024*1024);

    std::ofstream* _meta_fp = NULL;
    sprintf(buf, "%s.meta.%ld", _trace_path.c_str(), _tid);
    _meta_fp = new std::ofstream(buf, std::ios::out);
    if(!_meta_fp) {
      LOG(ERROR) << "Failed to open " << buf;
    }
    _trace_writer->_fp = _meta_fp;
  }
}


bool TracingContext::Enabled() {
  return _enabled;
}

void TracingContext::RecordSchedulerBegin(int64_t task_id, int64_t step_id,
    int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter) {
  if (! _enabled)
    return;
  init_thread_if_necessary();

  double timestamp = currentTimeMillisecond() - _tracing_context._program_start_time;

  if (frame_id != 0 || input_iter != 0) {
    _trace_writer->write(timestamp, TraceEventType::TRACE_SCHEDULER_BEGIN_WITH_ITER, task_id, step_id, partition_id, node_id, frame_id, input_iter);
  } else {
    _trace_writer->write(timestamp, TraceEventType::TRACE_SCHEDULER_BEGIN, task_id, step_id, partition_id, node_id);
  }
}

void TracingContext::RecordSchedulerEnd(int64_t task_id, int64_t step_id,
    int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter) {
  if (! _enabled)
    return;
  init_thread_if_necessary();

  double timestamp = currentTimeMillisecond() - _tracing_context._program_start_time;

  if (frame_id != 0 || input_iter != 0) {
    _trace_writer->write(timestamp, TraceEventType::TRACE_SCHEDULER_END_WITH_ITER, task_id, step_id, partition_id, node_id, frame_id, input_iter);
  } else {
    _trace_writer->write(timestamp, TraceEventType::TRACE_SCHEDULER_END, task_id, step_id, partition_id, node_id);
  }
}

void TracingContext::RecordComputeBegin(int64_t task_id, int64_t step_id,
    int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter,
    bool async) {
  if (! _enabled)
    return;
  init_thread_if_necessary();

  double timestamp = currentTimeMillisecond() - _tracing_context._program_start_time;

  if (frame_id != 0 || input_iter != 0) {
    _trace_writer->write(timestamp, TraceEventType::TRACE_COMPUTE_BEGIN_WITH_ITER, task_id, step_id, partition_id, node_id, frame_id, input_iter);
  } else {
    _trace_writer->write(timestamp, TraceEventType::TRACE_COMPUTE_BEGIN, task_id, step_id, partition_id, node_id);
  }
}

void TracingContext::RecordComputeEnd(int64_t task_id, int64_t step_id,
    int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter,
    bool async) {
  if (! _enabled)
    return;
  init_thread_if_necessary();

  double timestamp = currentTimeMillisecond() - _tracing_context._program_start_time;

  if (frame_id != 0 || input_iter != 0) {
    _trace_writer->write(timestamp, TraceEventType::TRACE_COMPUTE_END_WITH_ITER, task_id, step_id, partition_id, node_id, frame_id, input_iter);
  } else {
    _trace_writer->write(timestamp, TraceEventType::TRACE_COMPUTE_END, task_id, step_id, partition_id, node_id);
  }
}

int64_t TracingContext::nextTaskId() {
  return _num_tasks.fetch_add(1);
}

std::ostream& TracingContext::MetaStream() {
  init_thread_if_necessary();
  return *(_trace_writer->_fp);
}

TracingContext::TracingContext() : _num_tasks(0) {
  char* str;

  // Set MX_TRACE_PATH to enable tracing
  str = std::getenv("MX_TRACE_PATH");
  if(str == NULL) {
    LOG(INFO) << "MX_TRACE_PATH not set, default NULL";
  } else {
    _trace_path = string(str);
    _enabled = true;
    LOG(INFO) << "MX_TRACE_PATH=" << _trace_path;
  }
  _program_start_time = currentTimeMillisecond();
  if(sizeof(Serializer) % 64 != 0) {
    LOG(ERROR) << "ERROR: sizeof(Serializer) = " << sizeof(Serializer) << " which not a multiple of 64";
  }
}


TracingContext::~TracingContext() {
  // TODO Auto-generated destructor stub
}

TracingContext _tracing_context;

} /* namespace internal */
} /* namespace tensorflow */


