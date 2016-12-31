/*
 * tracing_context.h
 *
 *  Created on: 2016年11月22日
 *      Author: Bowen Yu <yubowen15@foxmail.com>
 */

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_TRACINGCONTEXT_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_TRACINGCONTEXT_H_

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <vector>
#include <atomic>

using std::string;

namespace mxnet {
namespace common {

const string SEP = " ";

extern std::atomic_int_fast64_t _num_tasks;
enum MetaEventType { META_NEW_TASK=1, META_NEW_PARTITION, META_NEW_NODE, META_ASSIGN_NODE_CHILDREN };
enum TraceEventType { TRACE_SCHEDULER_BEGIN=1, TRACE_SCHEDULER_END, TRACE_COMPUTE_BEGIN, TRACE_COMPUTE_END,
  TRACE_SCHEDULER_BEGIN_WITH_ITER, TRACE_SCHEDULER_END_WITH_ITER, TRACE_COMPUTE_BEGIN_WITH_ITER, TRACE_COMPUTE_END_WITH_ITER};

class TracingContext {
private:
  std::atomic_int_fast64_t _num_tasks;
  string _trace_path;
  bool _enabled = false;

  void init_thread_if_necessary();

public:
  double _program_start_time;
  bool Enabled();


  // Here we don't care about the specific information because (task_id, partition_id) can identify a graph,
  // and all the information about each node should be stored into that.
  void RecordSchedulerBegin(int64_t task_id, int64_t step_id, int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter);
  void RecordSchedulerEnd(int64_t task_id, int64_t step_id, int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter);
  void RecordComputeBegin(int64_t task_id, int64_t step_id, int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter, bool async);
  void RecordComputeEnd(int64_t task_id, int64_t step_id, int64_t partition_id, int node_id, uint64_t frame_id, int64_t input_iter, bool async);

  std::ostream& MetaStream();
  int64_t nextTaskId();

  /*
   * An extra param, full_key is concerned compared to record for SendRecv operation.
   * This information helps us connect _Send to corresponding _Recv when analyzing.
   */
  //void RecordSendRecvBegin(int node_id, int64_t step_id, const string& node_name, const string& type_string, const string& assigned_device_name, const std::vector<TracingNode>& in_node_id_list, const string& full_key, uintptr_t run_id);
  //void RecordSendRecvEnd(int node_id, int64_t step_id, const string& assigned_device_name);

  TracingContext();
  virtual ~TracingContext();
};

extern TracingContext _tracing_context;


} /* namespace internal */
} /* namespace tensorflow */

#endif /* TENSORFLOW_CORE_PLATFORM_DEFAULT_TRACINGCONTEXT_H_ */

