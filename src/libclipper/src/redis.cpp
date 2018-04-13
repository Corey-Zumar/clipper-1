#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <clipper/constants.hpp>
#include <clipper/logging.hpp>
#include <clipper/redis.hpp>
#include <redox.hpp>

using redox::Command;
using redox::Redox;
using redox::Subscriber;
using std::string;
using std::vector;
using std::unordered_map;

namespace clipper {
namespace redis {

const std::string VERSION_METADATA_PREFIX = "CURRENT_MODEL_VERSION:";

bool contains_prohibited_chars_for_group(std::string value) {
  for (std::string prohibited_str : clipper::redis::prohibited_group_strings) {
    if (value.find(prohibited_str) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::unordered_map<string, string> parse_redis_map(const std::vector<string>& redis_data) {
  std::unordered_map<string, string> parsed_map;
  for (auto m = redis_data.begin(); m != redis_data.end(); ++m) {
    auto key = *m;
    m += 1;
    auto value = *m;
    log_info_formatted(LOGGING_TAG_REDIS, "\t {}: {}", key, value);
    parsed_map[key] = value;
  }
  return parsed_map;
}

std::string gen_model_replica_key(const VersionedModelId& key, int model_replica_id) {
  std::stringstream ss;
  ss << key.get_name();
  ss << ITEM_DELIMITER;
  ss << key.get_id();
  ss << ITEM_DELIMITER;
  ss << std::to_string(model_replica_id);
  return ss.str();
}

// Expects keys of format "model_name:model_version:replica_id"
std::vector<ContainerModelDataItem> parse_models_key(const std::string& key) {
  std::vector<std::string> model_keys = str_to_labels(key);
  std::vector<ContainerModelDataItem> model_data_items;
  model_data_items.reserve(model_keys.size());
  for (auto& model_key : model_keys) {
    size_t pos = model_key.find(ITEM_DELIMITER);
    if (pos == std::string::npos) {
      throw std::invalid_argument("Couldn't parse model replica key \"" + key + "\"");
    }
    std::string model_name = model_key.substr(0, pos);
    model_key.erase(0, pos + ITEM_DELIMITER.length());

    pos = model_key.find(ITEM_DELIMITER);
    if (pos == std::string::npos) {
      throw std::invalid_argument("Couldn't parse model replica key \"" + key + "\"");
    }
    std::string model_version = model_key.substr(0, pos);
    model_key.erase(0, pos + ITEM_DELIMITER.length());
    int replica_id = std::stoi(model_key);
    VersionedModelId model = VersionedModelId(model_name, model_version);
    model_data_items.push_back(std::make_pair(model, replica_id));
  }

  return model_data_items;
}

std::string gen_versioned_model_key(const VersionedModelId& key) {
  std::stringstream ss;
  ss << key.get_name();
  ss << ":";
  ss << key.get_id();
  return ss.str();
}

std::string gen_model_current_version_key(const std::string& model_name) {
  std::stringstream ss;
  ss << VERSION_METADATA_PREFIX;
  ss << model_name;
  return ss.str();
}

// Update `prohibited_group_strings` when changing the set of delimeters and/or
// other generic substrings used
string labels_to_str(const vector<string>& labels) {
  if (labels.empty()) return "";

  std::ostringstream ss;
  for (auto l = labels.begin(); l != labels.end() - 1; ++l) {
    ss << *l << ITEM_DELIMITER;
  }
  // don't forget to save the last label
  ss << *(labels.end() - 1);
  return ss.str();
}

string model_names_to_str(const vector<string>& names) { return labels_to_str(names); }

// String parsing taken from http://stackoverflow.com/a/14267455/814642
vector<string> str_to_labels(const string& label_str) {
  auto start = 0;
  auto end = label_str.find(ITEM_DELIMITER);
  vector<string> labels;

  while (end != string::npos) {
    labels.push_back(label_str.substr(start, end - start));
    start = end + ITEM_DELIMITER.length();
    end = label_str.find(ITEM_DELIMITER, start);
  }
  // don't forget to parse the last label
  labels.push_back(label_str.substr(start, end - start));
  return labels;
}

std::string models_to_str(const std::vector<VersionedModelId>& models) {
  if (models.empty()) return "";

  std::ostringstream ss;
  for (auto m = models.begin(); m != models.end() - 1; ++m) {
    ss << m->get_name() << ITEM_PART_CONCATENATOR << m->get_id() << ITEM_DELIMITER;
  }
  // don't forget to save the last label
  ss << (models.end() - 1)->get_name() << ITEM_PART_CONCATENATOR << (models.end() - 1)->get_id();
  log_info_formatted(LOGGING_TAG_REDIS, "models_to_str result: {}", ss.str());
  return ss.str();
}

std::vector<VersionedModelId> str_to_models(const std::string& model_str) {
  auto start = 0;
  auto end = model_str.find(ITEM_DELIMITER);
  vector<VersionedModelId> models;

  while (end != string::npos) {
    auto split = start + model_str.substr(start, end - start).find(ITEM_PART_CONCATENATOR);
    std::string model_name = model_str.substr(start, split - start);
    std::string model_version = model_str.substr(split + 1, end - split - 1);
    models.push_back(VersionedModelId(model_name, model_version));
    start = end + ITEM_DELIMITER.length();
    end = model_str.find(ITEM_DELIMITER, start);
  }

  // don't forget to parse the last model
  auto split = start + model_str.substr(start, end - start).find(ITEM_PART_CONCATENATOR);
  std::string model_name = model_str.substr(start, split - start);
  std::string model_version = model_str.substr(split + 1, end - split - 1);
  models.push_back(VersionedModelId(model_name, model_version));

  return models;
}

bool set_current_model_version(redox::Redox& redis, const std::string& model_name,
                               const std::string& version) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_METADATA_DB_NUM)})) {
    std::string key = gen_model_current_version_key(model_name);
    const vector<string> cmd_vec{"SET", key, version};

    return send_cmd_no_reply<string>(redis, cmd_vec);
  } else {
    return false;
  }
}

boost::optional<std::string> get_current_model_version(redox::Redox& redis,
                                                       const std::string& model_name) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_METADATA_DB_NUM)})) {
    std::string key = gen_model_current_version_key(model_name);
    auto result = send_cmd_with_reply<string>(redis, {"GET", key});
    if (result) {
      std::string version = *result;
      if (version.size() == 0) {
        log_error_formatted(LOGGING_TAG_REDIS, "Versions cannot be empty string. Found version {}",
                            version);
      } else {
        return version;
      }
    }
  }
  log_error_formatted(LOGGING_TAG_REDIS, "No versions found for model {}", model_name);
  return boost::none;
}

std::vector<std::string> get_linked_models(redox::Redox& redis, const std::string& app_name) {
  std::vector<std::string> linked_models;
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_APP_MODEL_LINKS_DB_NUM)})) {
    auto result = send_cmd_with_reply<std::vector<string>>(redis, {"SMEMBERS", app_name});
    if (result) {
      linked_models = *result;
    } else {
      log_error_formatted(LOGGING_TAG_REDIS, "Found no linked models for app {}", app_name);
    }
  } else {
    log_error_formatted(LOGGING_TAG_REDIS,
                        "Redis encountered an error in searching for app links for {}", app_name);
  }

  return linked_models;
}

bool add_model(Redox& redis, const VersionedModelId& model_id, const DataType& input_type,
               const vector<string>& labels, const std::string& container_name,
               const std::string& model_data_path, const int batch_size) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_MODEL_DB_NUM)})) {
    std::string model_id_key = gen_versioned_model_key(model_id);
    // clang-format off
    const vector<string> cmd_vec{
      "HMSET",            model_id_key,
      "model_name",       model_id.get_name(),
      "model_version",    model_id.get_id(),
      "load",             std::to_string(0.0),
      "input_type",       get_readable_input_type(input_type),
      "labels",           labels_to_str(labels),
      "container_name",   container_name,
      "model_data_path",  model_data_path,
      "batch_size",       std::to_string(batch_size)};
    // clang-format on
    return send_cmd_no_reply<string>(redis, cmd_vec);
  } else {
    return false;
  }
}

bool delete_model(Redox& redis, const VersionedModelId& model_id) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_MODEL_DB_NUM)})) {
    std::string model_id_key = gen_versioned_model_key(model_id);
    return send_cmd_no_reply<int>(redis, {"DEL", model_id_key});
  } else {
    return false;
  }
}

unordered_map<string, string> get_model(Redox& redis, const VersionedModelId& model_id) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_MODEL_DB_NUM)})) {
    std::string model_id_key = gen_versioned_model_key(model_id);

    std::vector<std::string> model_data;
    auto result = send_cmd_with_reply<vector<string>>(redis, {"HGETALL", model_id_key});
    if (result) {
      model_data = *result;
    }
    return parse_redis_map(model_data);
  } else {
    return unordered_map<string, string>{};
  }
}

std::unordered_map<std::string, std::string> get_model_by_key(redox::Redox& redis,
                                                              const std::string& key) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_MODEL_DB_NUM)})) {
    std::vector<std::string> model_data;
    auto result = send_cmd_with_reply<vector<string>>(redis, {"HGETALL", key});
    if (result) {
      model_data = *result;
    }
    return parse_redis_map(model_data);
  } else {
    return unordered_map<string, string>{};
  }
}

std::vector<std::string> get_model_versions(redox::Redox& redis, const std::string& model_name) {
  std::vector<std::string> versions;
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_MODEL_DB_NUM)})) {
    std::stringstream ss;
    ss << model_name;
    ss << ":*";
    auto key_regex = ss.str();
    auto result = send_cmd_with_reply<vector<string>>(redis, {"KEYS", key_regex});
    if (result) {
      std::vector<std::string> model_keys;
      model_keys = *result;
      for (auto model_str : model_keys) {
        std::vector<VersionedModelId> parsed_model = str_to_models(model_str);
        versions.push_back(parsed_model.front().get_id());
      }
    }
  }
  return versions;
}

std::vector<std::string> get_all_model_names(redox::Redox& redis) {
  std::vector<std::string> model_names;
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_MODEL_DB_NUM)})) {
    // Use wildcard argument for KEYS command to get all key names.
    // The number of keys is assumed to be within reasonable limits.
    auto result = send_cmd_with_reply<vector<string>>(redis, {"KEYS", "*"});
    if (result) {
      // De-duplicate and return the key names.
      std::set<std::string> model_name_set;
      for (auto model_str : *result) {
        std::vector<VersionedModelId> parsed_model = str_to_models(model_str);
        model_name_set.insert(parsed_model.front().get_name());
      }
      model_names.insert(model_names.end(), model_name_set.begin(), model_name_set.end());
    }
  }
  return model_names;
}

std::vector<VersionedModelId> get_all_models(redox::Redox& redis) {
  std::vector<VersionedModelId> models;
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_MODEL_DB_NUM)})) {
    // Use wildcard argument for KEYS command to get all key names.
    // The number of keys is assumed to be within reasonable limits.
    auto result = send_cmd_with_reply<vector<string>>(redis, {"KEYS", "*"});
    if (result) {
      for (auto model_str : *result) {
        std::vector<VersionedModelId> parsed_model = str_to_models(model_str);
        models.push_back(parsed_model.front());
      }
    }
  }
  return models;
}

bool add_container(redox::Redox& redis,
                   const std::vector<ContainerModelDataItem> container_model_data,
                   const int zmq_connection_id) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_CONTAINER_DB_NUM)})) {
    size_t container_id = get_container_id(container_model_data);

    std::vector<std::string> models_key_components;
    models_key_components.reserve(container_model_data.size());
    for (const auto& model_data_item : container_model_data) {
      const VersionedModelId& model_id = model_data_item.first;
      int model_replica_id = model_data_item.second;
      std::string replica_key = gen_model_replica_key(model_id, model_replica_id);
      models_key_components.push_back(std::move(replica_key));
    }

    std::string models_key = labels_to_str(models_key_components);

    const vector<string> cmd_vec{"HMSET",
                                 std::to_string(container_id),
                                 "models",
                                 models_key,
                                 "zmq_connection_id",
                                 std::to_string(zmq_connection_id),
                                 "batch_size",
                                 std::to_string(1),
                                 "container_id",
                                 std::to_string(container_id)};
    return send_cmd_no_reply<string>(redis, cmd_vec);
  } else {
    return false;
  }
}

bool delete_container(Redox& redis, const ContainerId container_id) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_CONTAINER_DB_NUM)})) {
    std::string container_key = std::to_string(container_id);
    return send_cmd_no_reply<int>(redis, {"DEL", container_key});
  } else {
    return false;
  }
}

std::pair<std::vector<ContainerModelDataItem>, std::unordered_map<std::string, std::string>>
get_container(Redox& redis, const ContainerId container_id) {
  std::string container_key = std::to_string(container_id);
  return get_container_by_key(redis, container_key);
}

std::pair<std::vector<ContainerModelDataItem>, std::unordered_map<std::string, std::string>> get_container_by_key(Redox& redis, const std::string& key) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_CONTAINER_DB_NUM)})) {
    std::vector<std::string> container_data;
    auto result = send_cmd_with_reply<vector<string>>(redis, {"HGETALL", key});
    if (result) {
      container_data = *result;
    }
    std::unordered_map<std::string, std::string> parsed_container_data =
        parse_redis_map(container_data);

    std::string models = parsed_container_data.find("models")->second;
    std::vector<ContainerModelDataItem> container_models_data = parse_models_key(models);

    return std::make_pair(std::move(container_models_data), parsed_container_data);
  } else {
    return std::make_pair(std::vector<ContainerModelDataItem>{},
                          std::unordered_map<std::string, std::string>{});
  }
}

std::vector<std::vector<ContainerModelDataItem>> get_all_containers(redox::Redox& redis) {
  std::vector<std::vector<ContainerModelDataItem>> containers;
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_CONTAINER_DB_NUM)})) {
    // Use wildcard argument for KEYS command to get all key names.
    // The number of keys is assumed to be within reasonable limits.
    auto result = send_cmd_with_reply<vector<string>>(redis, {"KEYS", "*"});
    if (result) {
      auto container_keys = *result;
      for (auto c : container_keys) {
        // NOTE: This is not efficient, but we don't expect this method to be
        // invoked frequently
        std::pair<std::vector<ContainerModelDataItem>, std::unordered_map<std::string, std::string>>
            container_info = get_container_by_key(redis, c);
        containers.push_back(std::move(container_info.first));
      }
    }
  }
  return containers;
}

bool add_application(redox::Redox& redis, const std::string& appname, const DataType& input_type,
                     const std::string& policy, const std::string& default_output,
                     const long latency_slo_micros) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_APPLICATION_DB_NUM)})) {
    const vector<string> cmd_vec{"HMSET",
                                 appname,
                                 "input_type",
                                 get_readable_input_type(input_type),
                                 "policy",
                                 policy,
                                 "default_output",
                                 default_output,
                                 "latency_slo_micros",
                                 std::to_string(latency_slo_micros)};
    return send_cmd_no_reply<string>(redis, cmd_vec);
  } else {
    return false;
  }
}

bool add_model_links(redox::Redox& redis, const std::string& appname,
                     const std::vector<std::string>& model_names) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_APP_MODEL_LINKS_DB_NUM)})) {
    for (auto model_name : model_names) {
      if (!send_cmd_no_reply<int>(redis, vector<string>{"SADD", appname, model_name})) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

bool delete_application(redox::Redox& redis, const std::string& appname) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_APPLICATION_DB_NUM)})) {
    return send_cmd_no_reply<int>(redis, {"DEL", appname});
  } else {
    return false;
  }
}

std::unordered_map<std::string, std::string> get_application(redox::Redox& redis,
                                                             const std::string& appname) {
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_APPLICATION_DB_NUM)})) {
    std::vector<std::string> container_data;
    auto result = send_cmd_with_reply<vector<string>>(redis, {"HGETALL", appname});
    if (result) {
      container_data = *result;
    }

    return parse_redis_map(container_data);
  } else {
    return unordered_map<string, string>{};
  }
}

std::unordered_map<std::string, std::string> get_application_by_key(redox::Redox& redis,
                                                                    const std::string& key) {
  // Applications just use their appname as a key.
  // We keep the get_*_by_key() to preserve the symmetry of the
  // API.
  return get_application(redis, key);
}

std::vector<string> get_all_application_names(redox::Redox& redis) {
  std::vector<std::string> app_names;
  if (send_cmd_no_reply<string>(redis, {"SELECT", std::to_string(REDIS_APPLICATION_DB_NUM)})) {
    // Use wildcard argument for KEYS command to get all key names.
    // The number of keys is assumed to be within reasonable limits.
    auto result = send_cmd_with_reply<vector<string>>(redis, {"KEYS", "*"});
    if (result) {
      app_names = *result;
    }
  }
  return app_names;
}

void subscribe_to_keyspace_changes(
    int db, std::string prefix, Subscriber& subscriber,
    std::function<void(const std::string&, const std::string&)> callback) {
  std::ostringstream subscription;
  subscription << "__keyspace@" << std::to_string(db) << "__:" << prefix << "*";
  std::string sub_str = subscription.str();
  log_info_formatted(LOGGING_TAG_REDIS, "SUBSCRIPTION STRING: {}", sub_str);
  subscriber.psubscribe(sub_str,
                        [callback, prefix](const std::string& topic, const std::string& msg) {
                          size_t split_idx = topic.find_first_of(":");
                          std::string key = topic.substr(split_idx + 1 + prefix.size());
                          log_info_formatted(LOGGING_TAG_REDIS, "MESSAGE: {}", msg);
                          callback(key, msg);
                        });
}

void subscribe_to_model_changes(
    Subscriber& subscriber, std::function<void(const std::string&, const std::string&)> callback) {
  std::string prefix = "";
  subscribe_to_keyspace_changes(REDIS_MODEL_DB_NUM, prefix, subscriber, std::move(callback));
}

void subscribe_to_container_changes(
    Subscriber& subscriber, std::function<void(const std::string&, const std::string&)> callback) {
  std::string prefix = "";
  subscribe_to_keyspace_changes(REDIS_CONTAINER_DB_NUM, prefix, subscriber, std::move(callback));
}

void subscribe_to_application_changes(
    redox::Subscriber& subscriber,
    std::function<void(const std::string&, const std::string&)> callback) {
  std::string prefix = "";
  subscribe_to_keyspace_changes(REDIS_APPLICATION_DB_NUM, prefix, subscriber, std::move(callback));
}

void subscribe_to_model_link_changes(
    redox::Subscriber& subscriber,
    std::function<void(const std::string&, const std::string&)> callback) {
  std::string prefix = "";
  subscribe_to_keyspace_changes(REDIS_APP_MODEL_LINKS_DB_NUM, prefix, subscriber,
                                std::move(callback));
}

void subscribe_to_model_version_changes(
    redox::Subscriber& subscriber,
    std::function<void(const std::string&, const std::string&)> callback) {
  subscribe_to_keyspace_changes(REDIS_METADATA_DB_NUM, VERSION_METADATA_PREFIX, subscriber,
                                std::move(callback));
}

}  // namespace redis
}  // namespace clipper
