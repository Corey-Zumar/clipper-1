#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>

#define PROVIDES_EXECUTORS
#include <boost/exception_ptr.hpp>
#include <boost/optional.hpp>

#include <boost/thread/executors/basic_thread_pool.hpp>


#include <clipper/containers.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/exceptions.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/query_processor.hpp>
#include <clipper/task_executor.hpp>
// #include <clipper/timers.hpp>

#define UNREACHABLE() assert(false)

using std::vector;
using std::tuple;

namespace clipper {

  QueryProcessor::QueryProcessor() 
    : state_db_(std::make_shared<StateDB>()),
    request_rate_(metrics::MetricsRegistry::get_metrics().create_meter(
          "query_processor:request_rate"))
  {
  // Create selection policy instances
  selection_policies_.emplace(DefaultOutputSelectionPolicy::get_name(),
                              std::make_shared<DefaultOutputSelectionPolicy>());
  log_info(LOGGING_TAG_QUERY_PROCESSOR, "Query Processor started");
}

std::shared_ptr<StateDB> QueryProcessor::get_state_table() const {
  return state_db_;
}

void QueryProcessor::predict(Query query,
    std::function<void(Response)>&& on_response_callback) {
  request_rate_->mark(1);
  long query_id = query_counter_.fetch_add(1);
  auto current_policy_iter = selection_policies_.find(query.selection_policy_);
  if (current_policy_iter == selection_policies_.end()) {
    std::stringstream err_msg_builder;
    err_msg_builder << query.selection_policy_ << " "
                    << "is an invalid selection_policy.";
    const std::string err_msg = err_msg_builder.str();
    log_error(LOGGING_TAG_QUERY_PROCESSOR, err_msg);
    throw PredictError(err_msg);
  }
  std::shared_ptr<SelectionPolicy> current_policy = current_policy_iter->second;

  auto state_opt = state_db_->get(StateKey{query.label_, query.user_id_, 0});
  if (!state_opt) {
    std::stringstream err_msg_builder;
    err_msg_builder << "No selection state found for query with user_id: "
                    << query.user_id_ << " and label: " << query.label_;
    const std::string err_msg = err_msg_builder.str();
    log_error(LOGGING_TAG_QUERY_PROCESSOR, err_msg);
    throw PredictError(err_msg);
  }

  if (!selection_state_) {
    selection_state_ = current_policy->deserialize(*state_opt);
  }

  std::vector<PredictTask> tasks =
      current_policy->select_predict_tasks(selection_state_, query, query_id);

  log_info_formatted(LOGGING_TAG_QUERY_PROCESSOR, "Found {} tasks",
                     tasks.size());


  task_executor_.schedule_prediction(tasks[0], [
      response_callback = std::move(on_response_callback),
      query, query_id
      ] (Output output) mutable {
      std::chrono::time_point<std::chrono::high_resolution_clock> end =
      std::chrono::high_resolution_clock::now();
      long duration_micros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end - query.create_time_)
      .count();
      boost::optional<std::string> default_explanation;

      response_callback(Response{
          query,
          query_id,
          duration_micros,
          std::move(output),
          false,
          std::move(default_explanation)});
      });
}

}  // namespace clipper
