#include <chrono>
#include <memory>
#include <random>
// uncomment to disable assert()
// #define NDEBUG
#include <cassert>
#include <sstream>

#include <clipper/callback_threadpool.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/metrics.hpp>
#include <clipper/task_executor.hpp>
#include <clipper/util.hpp>

namespace clipper {

// CacheEntry::CacheEntry() {}
//
// QueryCache::QueryCache(size_t size_bytes)
//     // : max_size_bytes_(size_bytes), callback_threadpool_("query_cache", 16) {}
//     : max_size_bytes_(size_bytes) {}
//
// bool QueryCache::fetch(
//     const VersionedModelId &model, const QueryId query_id,
//     std::function<void(Output, std::shared_ptr<QueryLineage>)> callback) {
//   std::unique_lock<std::mutex> l(m_);
//   auto key = hash(model, query_id);
//   auto search = entries_.find(key);
//   if (search != entries_.end()) {
//     // cache entry exists
//     if (search->second.completed_) {
//       // value already in cache
//       search->second.used_ = true;
//       // `makeFuture` takes an rvalue reference, so moving/forwarding
//       // the cache value directly would destroy it. Therefore, we use
//       // copy assignment to `value` and move the copied object instead
//       Output value = search->second.value_;
//       // callback_threadpool_.submit(callback, std::move(value),
//       //                             search->second.lineage_);
//       callback(std::move(value), search->second.lineage_);
//       return true;
//     } else {
//       // value not in cache yet
//       search->second.value_callbacks_.push_back(std::move(callback));
//       return false;
//     }
//   } else {
//     // cache entry doesn't exist yet, so create entry
//     CacheEntry new_entry;
//     // create promise/future pair for this request
//     new_entry.value_callbacks_.push_back(std::move(callback));
//     insert_entry(key, new_entry);
//     return false;
//   }
// }
//
// void QueryCache::put(const VersionedModelId &model, const QueryId query_id,
//                      Output output, std::shared_ptr<QueryLineage> lineage) {
//   std::unique_lock<std::mutex> l(m_);
//   auto key = hash(model, query_id);
//   auto search = entries_.find(key);
//   if (search != entries_.end()) {
//     CacheEntry &entry = search->second;
//     if (!entry.completed_) {
//       // Complete the outstanding promises
//       auto callbacks = std::move(search->second.value_callbacks_);
//       entry.completed_ = true;
//       entry.value_ = output;
//       entry.lineage_ = lineage;
//       size_bytes_ += output.y_hat_->size();
//       evict_entries(size_bytes_ - max_size_bytes_);
//       l.unlock();
//
//       for (auto &c : callbacks) {
//         c(output, lineage);
//         // callback_threadpool_.submit(c, output, lineage);
//       }
//     }
//   } else {
//     CacheEntry new_entry;
//     new_entry.value_ = output;
//     new_entry.completed_ = true;
//     new_entry.lineage_ = lineage;
//     insert_entry(key, new_entry);
//   }
// }
//
// size_t QueryCache::hash(const VersionedModelId &model,
//                         const QueryId query_id) const {
//   std::size_t seed = 0;
//   size_t model_hash = std::hash<clipper::VersionedModelId>()(model);
//   boost::hash_combine(seed, model_hash);
//   boost::hash_combine(seed, query_id);
//   return seed;
// }
//
// void QueryCache::insert_entry(const long key, CacheEntry &value) {
//   size_t entry_size_bytes = value.completed_ ? value.value_.y_hat_->size() : 0;
//   if (entry_size_bytes <= max_size_bytes_) {
//     evict_entries(size_bytes_ + entry_size_bytes - max_size_bytes_);
//     page_buffer_.insert(page_buffer_.begin() + page_buffer_index_, key);
//     page_buffer_index_ = (page_buffer_index_ + 1) % page_buffer_.size();
//     size_bytes_ += entry_size_bytes;
//     entries_.insert(std::make_pair(key, std::move(value)));
//   } else {
//     // This entry is too large to cache
//     log_error_formatted(LOGGING_TAG_TASK_EXECUTOR,
//                         "Received an output of size: {} bytes that exceeds "
//                         "cache size of: {} bytes",
//                         entry_size_bytes, max_size_bytes_);
//   }
// }
//
// void QueryCache::evict_entries(long space_needed_bytes) {
//   if (space_needed_bytes <= 0) {
//     return;
//   }
//   while (space_needed_bytes > 0 && !page_buffer_.empty()) {
//     long page_key = page_buffer_[page_buffer_index_];
//     auto page_entry_search = entries_.find(page_key);
//     if (page_entry_search == entries_.end()) {
//       throw std::runtime_error(
//           "Failed to find corresponding cache entry for a buffer page!");
//     }
//     CacheEntry &page_entry = page_entry_search->second;
//     if (page_entry.used_ || !page_entry.completed_) {
//       page_entry.used_ = false;
//       page_buffer_index_ = (page_buffer_index_ + 1) % page_buffer_.size();
//     } else {
//       page_buffer_.erase(page_buffer_.begin() + page_buffer_index_);
//       page_buffer_index_ = page_buffer_.size() > 0
//                                ? page_buffer_index_ % page_buffer_.size()
//                                : 0;
//       size_bytes_ -= page_entry.value_.y_hat_->size();
//       space_needed_bytes -= page_entry.value_.y_hat_->size();
//       entries_.erase(page_entry_search);
//     }
//   }
// }

void noop_free(void *data, void *hint) {
  // std::cout << "NOOP FREE CALLED" << std::endl;
}

void real_free(void *data, void *hint) { free(data); }

std::pair<std::vector<zmq::message_t>, std::vector<uint32_t>> construct_batch_message(
    std::vector<PredictTask> tasks) {
  std::vector<uint32_t> batch_ids;
  batch_ids.reserve(tasks.size());

  std::vector<zmq::message_t> messages;
  size_t request_metadata_size = 1 * sizeof(uint32_t);
  uint32_t *request_metadata = static_cast<uint32_t *>(malloc(request_metadata_size));
  request_metadata[0] = static_cast<uint32_t>(RequestType::PredictRequest);

  size_t input_metadata_size = (1 + (tasks.size() * 3)) * sizeof(uint32_t);
  uint32_t *input_metadata = static_cast<uint32_t *>(malloc(input_metadata_size));
  input_metadata[0] = static_cast<uint32_t>(tasks.size());

  uint32_t curr_batch_id = 0;
  for (size_t i = 0; i < tasks.size(); ++i) {
    batch_ids.push_back(curr_batch_id);
    input_metadata[(3 * i) + 1] = curr_batch_id;
    input_metadata[(3 * i) + 2] = static_cast<uint32_t>(tasks[i].input_.type_);
    input_metadata[(3 * i) + 3] = static_cast<uint32_t>(tasks[i].input_.size_bytes_);
    curr_batch_id += 1;
  }

  size_t input_metadata_size_buf_size = 1 * sizeof(long);
  long *input_metadata_size_buf = static_cast<long *>(malloc(input_metadata_size_buf_size));
  // Add the size of the input metadata in bytes. This will be
  // sent prior to the input metadata to allow for proactive
  // buffer allocation in the receiving container
  input_metadata_size_buf[0] = input_metadata_size;

  messages.emplace_back(
      zmq::message_t(reinterpret_cast<void *>(request_metadata), request_metadata_size, real_free));

  messages.emplace_back(zmq::message_t(reinterpret_cast<void *>(input_metadata_size_buf),
                                       input_metadata_size_buf_size, real_free));

  messages.emplace_back(
      zmq::message_t(reinterpret_cast<void *>(input_metadata), input_metadata_size, real_free));

  for (size_t i = 0; i < tasks.size(); ++i) {
    VersionedModelId &task_vm = tasks[i].model_;
    std::string serialized_vm = task_vm.serialize();
    size_t model_name_size = serialized_vm.size() * sizeof(char);
    zmq::message_t msg_model_name(model_name_size);
    memcpy(msg_model_name.data(), serialized_vm.data(), model_name_size);
    messages.emplace_back(std::move(msg_model_name));
    messages.emplace_back(zmq::message_t(reinterpret_cast<void *>(tasks[i].input_.data_),
                                         tasks[i].input_.size_bytes_, noop_free));
  }
  return std::make_pair(std::move(messages), std::move(batch_ids));
}

}  // namespace clipper
