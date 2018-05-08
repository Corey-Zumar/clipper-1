#include <chrono>
#include <iostream>
#include <memory>
#include <random>
// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

#include <clipper/constants.hpp>
#include <clipper/containers.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/util.hpp>

#include <boost/circular_buffer.hpp>
#include <boost/thread.hpp>

namespace clipper {

const std::string LOGGING_TAG_CONTAINERS = "CONTAINERS";

ContainerId get_container_id(const std::vector<ContainerModelDataItem> &container_data) {
  ContainerId container_id = 0;
  for (auto &model_item : container_data) {
    boost::hash_combine(container_id, model_item.first.get_name());
    boost::hash_combine(container_id, model_item.first.get_id());
    boost::hash_combine(container_id, model_item.second);
  }
  return container_id;
}

ModelContainer::ModelContainer(ContainerId container_id,
                               std::vector<ContainerModelDataItem> model_data, int connection_id,
                               int batch_size)
    : container_id_(container_id),
      connection_id_(connection_id),
      batch_size_(batch_size),
      latency_hist_("container:" + std::to_string(container_id) + ":prediction_latency",
                    "microseconds", HISTOGRAM_SAMPLE_SIZE) {
  log_info_formatted(LOGGING_TAG_CONTAINERS, "Creating new ModelContainer with id {}",
                     std::to_string(container_id));

  for (auto &data_item : model_data) {
    model_data_.emplace(data_item.first, data_item.second);
  }
}

size_t ModelContainer::get_batch_size(Deadline deadline) {
  std::lock_guard<std::mutex> lock(batch_size_mtx_);
  return batch_size_;
}

void ModelContainer::set_batch_size(int batch_size) {
  std::lock_guard<std::mutex> lock(batch_size_mtx_);
  batch_size_ = batch_size;
}

int ModelContainer::get_replica_id(VersionedModelId model_id) const {
  auto id_search = model_data_.find(model_id);
  if (id_search == model_data_.end()) {
    std::stringstream ss;
    ss << "Attempted to find replica id for unregistered model " << model_id.get_name() << ":"
       << model_id.get_id();
    throw std::runtime_error(ss.str());
  }
  return id_search->second;
}

void ActiveContainers::add_container(std::vector<ContainerModelDataItem> model_data,
                                     int connection_id) {
  ContainerId container_id = get_container_id(model_data);
  add_container(container_id, std::move(model_data), connection_id);
}

void ActiveContainers::add_container(ContainerId container_id,
                                     std::vector<ContainerModelDataItem> model_data,
                                     int connection_id) {
  log_info_formatted(LOGGING_TAG_CONTAINERS,
                     "Adding new container - container ID: {},"
                     "connection ID: {}",
                     container_id, connection_id);
  boost::unique_lock<boost::shared_mutex> l{m_};

  // Set a default batch size of 30
  int batch_size = 30;
  // auto batch_size_search = batch_sizes_.find(model);
  // if (batch_size_search != batch_sizes_.end()) {
  //   batch_size = batch_size_search->second;
  // }

  auto new_container = std::make_shared<ModelContainer>(container_id, std::move(model_data),
                                                        connection_id, batch_size);

  by_id_containers_.emplace(container_id, new_container);

  for (const auto &model_data_item : new_container->model_data_) {
    const VersionedModelId &vm = model_data_item.first;
    int replica_id = model_data_item.second;
    auto &entry = by_model_containers_[vm];
    entry.emplace(replica_id, new_container);
    assert(by_model_containers_[vm].size() > 0);
  }

  std::stringstream log_msg;
  log_msg << "\nActive containers:\n";
  for (auto model : by_model_containers_) {
    log_msg << "\tModel: " << model.first.serialize() << "\n";
    for (auto r : model.second) {
      log_msg << "\t\trep_id: " << r.first << ", container_id: " << r.second->container_id_ << "\n";
    }
  }
  log_info(LOGGING_TAG_CONTAINERS, log_msg.str());
}

std::shared_ptr<ModelContainer> ActiveContainers::get_model_replica(const VersionedModelId &model,
                                                                    const int replica_id) {
  boost::shared_lock<boost::shared_mutex> l{m_};

  auto replicas_map_entry = by_model_containers_.find(model);
  if (replicas_map_entry == by_model_containers_.end()) {
    log_error_formatted(LOGGING_TAG_CONTAINERS, "Requested replica {} for model {} NOT FOUND",
                        replica_id, model.serialize());
    return nullptr;
  }

  std::map<int, std::shared_ptr<ModelContainer>> replicas_map = replicas_map_entry->second;
  auto replica_entry = replicas_map.find(replica_id);
  if (replica_entry != replicas_map.end()) {
    return replica_entry->second;
  } else {
    log_error_formatted(LOGGING_TAG_CONTAINERS, "Requested replica {} for model {} NOT FOUND",
                        replica_id, model.serialize());
    return nullptr;
  }
}

std::shared_ptr<ModelContainer> ActiveContainers::get_container_by_id(
    const ContainerId container_id) {
  boost::shared_lock<boost::shared_mutex> l{m_};

  auto container_entry = by_id_containers_.find(container_id);
  if (container_entry == by_id_containers_.end()) {
    log_error_formatted(LOGGING_TAG_CONTAINERS, "Could not find container with id {}",
                        container_id);
    return nullptr;
  }
  return container_entry->second;
}

std::vector<VersionedModelId> ActiveContainers::get_known_models() {
  boost::shared_lock<boost::shared_mutex> l{m_};
  std::vector<VersionedModelId> keys;
  for (auto m : by_model_containers_) {
    keys.push_back(m.first);
  }
  return keys;
}

void ActiveContainers::register_batch_size(VersionedModelId model, int batch_size) {
  boost::shared_lock<boost::shared_mutex> l{m_};

  auto batch_size_entry = batch_sizes_.find(model);
  if (batch_size_entry != batch_sizes_.end()) {
    batch_sizes_.erase(model);
  }
  batch_sizes_.emplace(model, batch_size);
  auto matching_containers_entry = by_model_containers_.find(model);
  if (matching_containers_entry != by_model_containers_.end()) {
    for (auto &container : matching_containers_entry->second) {
      container.second->set_batch_size(batch_size);
    }
  }
}
}
