#ifndef CLIPPER_LIB_TASK_EXECUTOR_H
#define CLIPPER_LIB_TASK_EXECUTOR_H

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#include <boost/optional.hpp>

#include <redox.hpp>
#include <zmq.hpp>

#include <clipper/callback_threadpool.hpp>
#include <clipper/config.hpp>
#include <clipper/containers.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/redis.hpp>
#include <clipper/rpc_service.hpp>
#include <clipper/threadpool.hpp>
#include <clipper/util.hpp>

namespace clipper {

const std::string LOGGING_TAG_TASK_EXECUTOR = "TASKEXECUTOR";

class ModelMetrics {
 public:
  explicit ModelMetrics(VersionedModelId model)
      : model_(model),
        latency_(metrics::MetricsRegistry::get_metrics().create_histogram(
            "model:" + model.serialize() + ":prediction_latency", "microseconds", 4096)),
        // latency_list_(metrics::MetricsRegistry::get_metrics().create_data_list<long
        // long>(
        //     "model:" + model.serialize() + ":prediction_latencies_list",
        //     "microseconds"
        // )),
        throughput_(metrics::MetricsRegistry::get_metrics().create_meter(
            "model:" + model.serialize() + ":prediction_throughput")),
        num_predictions_(metrics::MetricsRegistry::get_metrics().create_counter(
            "model:" + model.serialize() + ":num_predictions")),
        // cache_hit_ratio_(
        //     metrics::MetricsRegistry::get_metrics().create_ratio_counter(
        //         "model:" + model.serialize() + ":cache_hit_ratio")),
        batch_size_(metrics::MetricsRegistry::get_metrics().create_histogram(
            "model:" + model.serialize() + ":batch_size", "queries", 4096)) {}
  ~ModelMetrics() = default;
  ModelMetrics(const ModelMetrics &) = default;
  ModelMetrics &operator=(const ModelMetrics &) = default;

  ModelMetrics(ModelMetrics &&) = default;
  ModelMetrics &operator=(ModelMetrics &&) = default;

  VersionedModelId model_;
  std::shared_ptr<metrics::Histogram> latency_;
  // std::shared_ptr<metrics::DataList<long long>> latency_list_;
  std::shared_ptr<metrics::Meter> throughput_;
  std::shared_ptr<metrics::Counter> num_predictions_;
  // std::shared_ptr<metrics::RatioCounter> cache_hit_ratio_;
  std::shared_ptr<metrics::Histogram> batch_size_;
};

class CacheEntry {
 public:
  CacheEntry();
  ~CacheEntry() = default;

  CacheEntry(const CacheEntry &) = delete;
  CacheEntry &operator=(const CacheEntry &) = delete;

  CacheEntry(CacheEntry &&) = default;
  CacheEntry &operator=(CacheEntry &&) = default;

  bool completed_ = false;
  bool used_ = true;
  Output value_;
  std::vector<std::function<void(Output)>> value_callbacks_;
};

// A cache page is a pair of <hash, entry_size>
using CachePage = std::pair<long, long>;

// NOTE: Prediction cache is now a query cache
class QueryCache {
 public:
  QueryCache(size_t size_bytes);
  bool fetch(const VersionedModelId &model, const QueryId query_id,
             std::function<void(Output)> callback);

  void put(const VersionedModelId &model, const QueryId query_id, Output output);

 private:
  size_t hash(const VersionedModelId &model, const QueryId query_id) const;
  void insert_entry(const long key, CacheEntry &value);
  void evict_entries(long space_needed_bytes);

  std::mutex m_;
  const size_t max_size_bytes_;
  size_t size_bytes_ = 0;
  // TODO cache needs a promise as well?
  std::unordered_map<long, CacheEntry> entries_;
  std::vector<long> page_buffer_;
  size_t page_buffer_index_ = 0;
  std::shared_ptr<metrics::Counter> lookups_counter_;
  std::shared_ptr<metrics::RatioCounter> hit_ratio_;
  // CallbackThreadPool callback_threadpool_;
};

struct DeadlineCompare {
  bool operator()(const std::pair<Deadline, PredictTask> &lhs,
                  const std::pair<Deadline, PredictTask> &rhs) {
    return lhs.first > rhs.first;
  }
};

// thread safe model queue
class ContainerQueue {
 public:
  ContainerQueue(std::string name)
      : queue_(ModelPQueue{}),
        queue_size_hist_(metrics::MetricsRegistry::get_metrics().create_histogram(
            name + ":queue_size", "microseconds", 1000)) {}
  // queue_size_list_(
  //     metrics::MetricsRegistry::get_metrics()
  //         .create_data_list<size_t>(name + ":queue_sizes", "queue size")),
  // queue_arrivals_list_(
  //     metrics::MetricsRegistry::get_metrics()
  //         .create_data_list<long long>(name + ":queue_arrivals",
  //         "timestamp")) {}

  // Disallow copy and assign
  ContainerQueue(const ContainerQueue &) = delete;
  ContainerQueue &operator=(const ContainerQueue &) = delete;

  ContainerQueue(ContainerQueue &&) = default;
  ContainerQueue &operator=(ContainerQueue &&) = default;

  ~ContainerQueue() = default;

  void add_task(PredictTask task) {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    std::chrono::time_point<std::chrono::system_clock> current_time =
        std::chrono::system_clock::now();

    Deadline deadline = current_time + std::chrono::microseconds(task.latency_slo_micros_);
    queue_.emplace(deadline, std::move(task));

    // long long curr_system_time =
    // clock::ClipperClock::get_clock().get_uptime();
    // queue_arrivals_list_->insert(curr_system_time);

    // queue_size_list_->insert(queue_.size());
    queue_not_empty_condition_.notify_one();
  }

  int get_size() {
    std::unique_lock<std::mutex> l(queue_mutex_);
    return queue_.size();
  }

  std::vector<PredictTask> get_batch(std::function<int(Deadline)> &&get_batch_size,
                                     bool full_batch) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_not_empty_condition_.wait(lock, [this]() { return !queue_.empty(); });
    Deadline deadline = queue_.top().first;
    int max_batch_size = get_batch_size(deadline);
    std::vector<PredictTask> batch;

    if (full_batch) {
      queue_not_empty_condition_.wait_for(
          lock, std::chrono::milliseconds(100),
          [this, max_batch_size]() { return queue_.size() >= max_batch_size; });
      while (batch.size() < (size_t)max_batch_size && queue_.size() > 0) {
        auto &task = queue_.top().second;
        batch.push_back(task);
        queue_.pop();
      }
      queue_size_hist_->insert(static_cast<int64_t>(queue_.size()));
      // queue_size_list_->insert(queue_.size());
      return batch;

    } else {
      while (batch.size() < (size_t)max_batch_size && queue_.size() > 0) {
        auto &task = queue_.top().second;
        batch.push_back(task);
        queue_.pop();
      }
      queue_size_hist_->insert(static_cast<int64_t>(queue_.size()));
      // queue_size_list_->insert(queue_.size());
      return batch;
    }
  }

  void drain_queue() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_ = ModelPQueue{};
  }

 private:
  // Min PriorityQueue so that the task with the earliest
  // deadline is at the front of the queue
  using ModelPQueue =
      std::priority_queue<std::pair<Deadline, PredictTask>,
                          std::vector<std::pair<Deadline, PredictTask>>, DeadlineCompare>;
  ModelPQueue queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_not_empty_condition_;
  std::shared_ptr<metrics::Histogram> queue_size_hist_;
  // std::shared_ptr<metrics::DataList<size_t>> queue_size_list_;
  // std::shared_ptr<metrics::DataList<long long>> queue_arrivals_list_;

  // Deletes tasks with deadlines prior or equivalent to the
  // current system time. This method should only be called
  // when a unique lock on the queue_mutex is held.
  void remove_tasks_with_elapsed_deadlines() {
    std::chrono::time_point<std::chrono::system_clock> current_time =
        std::chrono::system_clock::now();
    while (!queue_.empty()) {
      Deadline first_deadline = queue_.top().first;
      if (first_deadline <= current_time) {
        // If a task's deadline has already elapsed,
        // we should not process it
        queue_.pop();
      } else {
        break;
      }
    }
  }
};

class InflightMessage {
 public:
  InflightMessage(const std::chrono::time_point<std::chrono::system_clock> send_time,
                  const ContainerId container_id, const VersionedModelId model, int replica_id,
                  const InputVector input, const QueryId query_id)
      : send_time_(send_time),
        container_id_(container_id),
        model_(model),
        replica_id_(replica_id),
        input_(input),
        query_id_(query_id) {}

  // Default copy and move constructors
  InflightMessage(const InflightMessage &) = default;
  InflightMessage(InflightMessage &&) = default;

  // Default assignment operators
  InflightMessage &operator=(const InflightMessage &) = default;
  InflightMessage &operator=(InflightMessage &&) = default;

  std::chrono::time_point<std::chrono::system_clock> send_time_;
  ContainerId container_id_;
  VersionedModelId model_;
  int replica_id_;
  InputVector input_;
  QueryId query_id_;
};

void noop_free(void *data, void *hint);

void real_free(void *data, void *hint);

/**
 * Returns a pair consisting of ZMQ messages to be sent via the RPC layer,
 * as well as a collection of batch ids. The id at index n corresponds to
 * the specified predict task at index n. These ids will be useful when
 * deconstructing the batched response into individual query results.
 */
std::pair<std::vector<zmq::message_t>, std::vector<uint32_t>> construct_batch_message(
    std::vector<PredictTask> tasks);

class TaskExecutor {
 public:
  ~TaskExecutor() { active_->store(false); };
  explicit TaskExecutor()
      : active_(std::make_shared<std::atomic_bool>(true)),
        active_containers_(std::make_shared<ActiveContainers>()),
        rpc_(std::make_unique<rpc::RPCService>()),
        // cache_(std::make_unique<QueryCache>(0)),
        model_metrics_({}) {
    // Seed the random number generator used for container queue selection
    // TODO(czumar): Do something better than randomly generating a number
    // to select the container queue for a task
    std::srand(std::time(nullptr));

    log_info(LOGGING_TAG_TASK_EXECUTOR, "TaskExecutor started");
    rpc_->start("*", RPC_SERVICE_SEND_PORT, RPC_SERVICE_RECV_PORT,
                [ this, task_executor_valid = active_ ](ContainerId container_id) {
                  if (*task_executor_valid) {
                    on_container_ready(container_id);
                  } else {
                    log_info(LOGGING_TAG_TASK_EXECUTOR,
                             "Not running on_container_ready callback because "
                             "TaskExecutor has been destroyed.");
                  }
                },
                [ this, task_executor_valid = active_ ](rpc::RPCResponse response) {
                  if (*task_executor_valid) {
                    on_response_recv(std::move(response));
                  } else {
                    log_info(LOGGING_TAG_TASK_EXECUTOR,
                             "Not running on_response_recv callback because "
                             "TaskExecutor has been destroyed.");
                  }

                });
    Config &conf = get_config();
    while (!redis_connection_.connect(conf.get_redis_address(), conf.get_redis_port())) {
      log_error(LOGGING_TAG_TASK_EXECUTOR, "TaskExecutor failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    while (!redis_subscriber_.connect(conf.get_redis_address(), conf.get_redis_port())) {
      log_error(LOGGING_TAG_TASK_EXECUTOR, "TaskExecutor subscriber failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    redis::subscribe_to_model_changes(
        redis_subscriber_, [ this, task_executor_valid = active_ ](const std::string &key,
                                                                   const std::string &event_type) {
          if (event_type == "hset" && *task_executor_valid) {
            auto model_info = redis::get_model_by_key(redis_connection_, key);
            VersionedModelId model_id =
                VersionedModelId(model_info["model_name"], model_info["model_version"]);
            int batch_size = std::stoi(model_info["batch_size"]);
            active_containers_->register_batch_size(model_id, batch_size);
          }
        });

    std::vector<VersionedModelId> models = redis::get_all_models(redis_connection_);
    for (auto model_id : models) {
      auto model_info = redis::get_model(redis_connection_, model_id);
      // VersionedModelId model_id = VersionedModelId(
      //     model_info["model_name"], model_info["model_version"]);
      int batch_size = std::stoi(model_info["batch_size"]);
      active_containers_->register_batch_size(model_id, batch_size);
    }

    redis::send_cmd_no_reply<std::string>(redis_connection_,
                                          {"CONFIG", "SET", "notify-keyspace-events", "AKE"});
    redis::subscribe_to_container_changes(
        redis_subscriber_,
        // event_type corresponds to one of the Redis event types
        // documented in https://redis.io/topics/notifications.
        [ this, task_executor_valid = active_ ](const std::string &key,
                                                const std::string &event_type) {
          if (event_type == "hset" && *task_executor_valid) {
            auto container_info = redis::get_container_by_key(redis_connection_, key);

            std::vector<ContainerModelDataItem> &container_models = container_info.first;
            std::unordered_map<std::string, std::string> &container_metadata =
                container_info.second;

            ContainerId container_id = std::stoul(container_metadata["container_id"]);
            int connection_id = std::stoi(container_metadata["zmq_connection_id"]);
            active_containers_->add_container(container_id, container_models, connection_id);

            create_container_queue(container_id, container_models);

            TaskExecutionThreadPool::create_queue(container_id);
            TaskExecutionThreadPool::submit_job(
                container_id, [this, container_id]() { on_container_ready(container_id); });

          } else if (!*task_executor_valid) {
            log_info(LOGGING_TAG_TASK_EXECUTOR,
                     "Not running TaskExecutor's "
                     "subscribe_to_container_changes callback because "
                     "TaskExecutor has been destroyed.");
          }
        });
    throughput_meter_ =
        metrics::MetricsRegistry::get_metrics().create_meter("internal:aggregate_model_throughput");
    predictions_counter_ = metrics::MetricsRegistry::get_metrics().create_counter(
        "internal:aggregate_num_predictions");
  }

  // Disallow copy
  TaskExecutor(const TaskExecutor &other) = delete;
  TaskExecutor &operator=(const TaskExecutor &other) = delete;

  TaskExecutor(TaskExecutor &&other) = default;
  TaskExecutor &operator=(TaskExecutor &&other) = default;

  void schedule_prediction(PredictTask task,
                           std::function<void(Output)> &&task_completion_callback) {
    predictions_counter_->increment(1);
    // add each task to the queue corresponding to its associated model
    boost::shared_lock<boost::shared_mutex> lock(model_queues_mutex_);
    auto model_container_queues_entry = model_queues_.find(task.model_);
    if (model_container_queues_entry != model_queues_.end()) {
      std::vector<std::shared_ptr<ContainerQueue>> &container_queues =
          model_container_queues_entry->second;

      if (container_queues.empty()) {
        // TODO(czumar): This behavior should be allowed. Support it!
        throw std::runtime_error(
            "Scheduling a prediction for which there are no connected containers!");
      }

      int queue_idx = std::rand() % container_queues.size();
      std::shared_ptr<ContainerQueue> &selected_queue = container_queues[queue_idx];

      {
        std::unique_lock<std::mutex> l(prediction_callback_map_mutex_);
        prediction_callback_map_.emplace(
            std::make_pair(task.query_id_, std::move(task_completion_callback)));
      }
      // bool cached = cache_->fetch(task.model_, task.query_id_,
      //                             std::move(task_completion_callback));
      // if (!cached) {
      task.recv_time_ = std::chrono::system_clock::now();
      selected_queue->add_task(task);
      log_info_formatted(LOGGING_TAG_TASK_EXECUTOR, "Adding task to queue. QueryID: {}, model: {}",
                         task.query_id_, task.model_.serialize());

      // }
    } else {
      log_error_formatted(LOGGING_TAG_TASK_EXECUTOR, "Received task for unknown model: {} : {}",
                          task.model_.get_name(), task.model_.get_id());
    }
  }

  void drain_queues() {
    boost::unique_lock<boost::shared_mutex> lock(container_queues_mutex_);
    for (auto &entry : container_queues_) {
      entry.second->drain_queue();
    }
  }

  void set_full_batches() { use_full_batches_ = true; }

 private:
  // active_containers_ is shared with the RPC service so it can add new
  // containers to the collection when they connect
  std::shared_ptr<std::atomic_bool> active_;
  std::shared_ptr<ActiveContainers> active_containers_;
  std::unique_ptr<rpc::RPCService> rpc_;
  // std::unique_ptr<QueryCache> cache_;
  redox::Redox redis_connection_;
  redox::Subscriber redis_subscriber_;
  std::mutex inflight_messages_mutex_;
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, InflightMessage>> inflight_messages_;
  std::shared_ptr<metrics::Counter> predictions_counter_;
  std::shared_ptr<metrics::Meter> throughput_meter_;
  boost::shared_mutex model_queues_mutex_;
  boost::shared_mutex container_queues_mutex_;
  std::unordered_map<VersionedModelId, std::vector<std::shared_ptr<ContainerQueue>>> model_queues_;
  std::unordered_map<ContainerId, std::shared_ptr<ContainerQueue>> container_queues_;
  boost::shared_mutex model_metrics_mutex_;
  std::unordered_map<VersionedModelId, ModelMetrics> model_metrics_;
  static constexpr int INITIAL_MODEL_QUEUES_MAP_SIZE = 100;

  std::unordered_map<QueryId, std::function<void(Output)>> prediction_callback_map_;
  std::mutex prediction_callback_map_mutex_;
  std::atomic_bool use_full_batches_{false};

  void create_container_queue(ContainerId container_id,
                              std::vector<ContainerModelDataItem> &container_models) {
    auto container_queue = std::make_shared<ContainerQueue>(std::to_string(container_id));
    boost::unique_lock<boost::shared_mutex> container_queues_lock(container_queues_mutex_);
    container_queues_.emplace(container_id, container_queue);
    container_queues_lock.unlock();

    boost::unique_lock<boost::shared_mutex> model_queues_lock(model_queues_mutex_);
    for (auto &model : container_models) {
      VersionedModelId &model_id = model.first;
      auto model_container_queues_search = model_queues_.find(model_id);
      if (model_container_queues_search == model_queues_.end()) {
        std::vector<std::shared_ptr<ContainerQueue>> container_queues;
        container_queues.reserve(1);
        container_queues.push_back(container_queue);
        model_queues_.emplace(model_id, std::move(container_queues));
      } else {
        std::vector<std::shared_ptr<ContainerQueue>> &container_queues =
            model_container_queues_search->second;
        container_queues.push_back(container_queue);
      }
    }
  }

  void on_container_ready(ContainerId container_id) {
    std::shared_ptr<ModelContainer> container =
        active_containers_->get_container_by_id(container_id);

    if (!container) {
      throw std::runtime_error(
          "TaskExecutor failed to find previously registered active "
          "container!");
    }
    boost::unique_lock<boost::shared_mutex> l(model_queues_mutex_);

    auto container_queue_entry = container_queues_.find(container_id);
    if (container_queue_entry == container_queues_.end()) {
      throw std::runtime_error(
          "Failed to find container queue associated with a previously registered "
          "container!");
    }
    std::shared_ptr<ContainerQueue> container_queue = container_queue_entry->second;

    // NOTE: It is safe to unlock here because we copy the shared_ptr to
    // the ContainerQueue object so even if that entry in the map gets deleted,
    // the ContainerQueue object won't be destroyed until our copy of the pointer
    // goes out of scope.
    l.unlock();

    std::vector<PredictTask> batch = container_queue->get_batch(
        [container](Deadline deadline) { return container->get_batch_size(deadline); },
        use_full_batches_);

    if (batch.size() > 0) {
      // move the lock up here, so that nothing can pull from the
      // inflight_messages_
      // map between the time a message is sent and when it gets inserted
      // into the map
      std::unique_lock<std::mutex> l(inflight_messages_mutex_);
      auto batch_message = construct_batch_message(batch);
      std::vector<zmq::message_t> rpc_message = std::move(batch_message.first);
      std::vector<uint32_t> &batch_ids = batch_message.second;

      std::unordered_map<uint32_t, InflightMessage> outbound_messages;

      std::chrono::time_point<std::chrono::system_clock> current_time =
          std::chrono::system_clock::now();
      for (size_t i = 0; i < batch.size(); ++i) {
        PredictTask &b = batch[i];
        uint32_t batch_id = batch_ids[i];
        outbound_messages.emplace(
            std::piecewise_construct, std::forward_as_tuple(batch_id),
            std::forward_as_tuple(current_time, container->container_id_, b.model_,
                                  container->get_replica_id(b.model_), b.input_, b.query_id_));
      }

      int message_id = rpc_->send_message(std::move(rpc_message), container->connection_id_);
      inflight_messages_.emplace(message_id, std::move(outbound_messages));
    } else {
      // log_error_formatted(LOGGING_TAG_TASK_EXECUTOR,
      //                     "ContainerQueue returned empty batch for model {}, replica {}",
      //                     model_id.serialize(), std::to_string(replica_id));
    }
  }

  void on_response_recv(rpc::RPCResponse response) {
    int msg_id = std::get<0>(response);
    std::unordered_map<uint32_t, std::shared_ptr<OutputData>> &outputs = std::get<1>(response);

    std::unique_lock<std::mutex> inflight_messages_lock(inflight_messages_mutex_);
    std::unordered_map<uint32_t, InflightMessage> inbound_messages = inflight_messages_[msg_id];
    size_t batch_size = inbound_messages.size();
    assert(outputs.size() == batch_size);
    inflight_messages_.erase(msg_id);
    inflight_messages_lock.unlock();

    throughput_meter_->mark(batch_size);
    std::chrono::time_point<std::chrono::system_clock> current_time =
        std::chrono::system_clock::now();

    std::unique_lock<std::mutex> callbacks_lock(prediction_callback_map_mutex_);
    boost::shared_lock<boost::shared_mutex> metrics_lock(model_metrics_mutex_);
    long task_latency_micros = -1;
    for (auto &entry : outputs) {
      uint32_t batch_id = entry.first;
      std::shared_ptr<OutputData> &output_data = entry.second;
      auto completed_msg_search = inbound_messages.find(batch_id);
      if (completed_msg_search == inbound_messages.end()) {
        throw std::runtime_error(
            "Received prediction response with no corresponding inflight message!");
      }
      InflightMessage &completed_msg = completed_msg_search->second;
      VersionedModelId &cur_model = completed_msg.model_;

      if (task_latency_micros == -1) {
        int cur_replica_id = completed_msg.replica_id_;
        auto task_latency = current_time - completed_msg.send_time_;
        task_latency_micros =
            std::chrono::duration_cast<std::chrono::microseconds>(task_latency).count();

        std::shared_ptr<ModelContainer> processing_container =
            active_containers_->get_model_replica(cur_model, cur_replica_id);
        processing_container->latency_hist_.insert(task_latency_micros);
      }

      auto search = prediction_callback_map_.find(completed_msg.query_id_);
      if (search != prediction_callback_map_.end()) {
        search->second(Output{output_data, {completed_msg.model_}});
        prediction_callback_map_.erase(completed_msg.query_id_);
      }

      boost::optional<ModelMetrics> cur_model_metric;
      auto cur_model_metric_entry = model_metrics_.find(cur_model);
      if (cur_model_metric_entry != model_metrics_.end()) {
        cur_model_metric = cur_model_metric_entry->second;
      }
      if (cur_model_metric) {
        (*cur_model_metric).throughput_->mark(batch_size);
        (*cur_model_metric).num_predictions_->increment(batch_size);
        (*cur_model_metric).batch_size_->insert(batch_size);
        (*cur_model_metric).latency_->insert(static_cast<int64_t>(task_latency_micros));
      }
    }
  }
};

}  // namespace clipper

#endif  // CLIPPER_LIB_TASK_EXECUTOR_H
