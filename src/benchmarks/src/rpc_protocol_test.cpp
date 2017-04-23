#include <cxxopts.hpp>
#include <redox.hpp>

#include <unordered_map>
#include <mutex>
#include <vector>
#include <sstream>

#include <clipper/rpc_service.hpp>
#include <clipper/config.hpp>
#include <clipper/constants.hpp>
#include <clipper/redis.hpp>
#include <clipper/datatypes.hpp>

using namespace clipper;

const std::string LOGGING_TAG_RPC_TEST = "RPCTEST";

constexpr int MESSAGE_HISTORY_SENT_HEARTBEAT = 1;
constexpr int MESSAGE_HISTORY_RECEIVED_HEARTBEAT = 2;
constexpr int MESSAGE_HISTORY_SENT_CONTAINER_METADATA = 3;
constexpr int MESSAGE_HISTORY_RECEIVED_CONTAINER_METADATA = 4;
constexpr int MESSAGE_HISTORY_SENT_CONTAINER_CONTENT = 5;
constexpr int MESSAGE_HISTORY_RECEIVED_CONTAINER_CONTENT = 6;

// From the container's perspective, we expect at least the following activity:
// 1. Send a heartbeat message to Clipper
// 2. Receive a heartbeat response from Clipper requesting container metadata
// 3. Send container metadata to Clipper
// 4. Receive a container content message from Clipper
constexpr int MESSAGE_HISTORY_MINIMUM_LENGTH = 4;

using RPCValidationResult = std::pair<bool, std::string>;

class Tester {
 public:
  explicit Tester(const int num_containers) : rpc_(std::make_unique<rpc::RPCService>()),
                                              num_containers_(num_containers) {}

  Tester(const Tester &other) = delete;
  Tester &operator=(const Tester &other) = delete;

  Tester(Tester &&other) = default;
  Tester &operator=(Tester &&other) = default;
  ~Tester() {
    std::unique_lock<std::mutex> l(test_completed_cv_mutex_);
    test_completed_cv_.wait(l, [this]() { return test_completed_ == true; });
  }

  void start() {
    rpc_->start("127.0.0.1",
               RPC_SERVICE_PORT,
               [](VersionedModelId /*model*/, int /*container_id*/) {},
               [this](rpc::RPCResponse response) {
                 on_response_received(response);
               });
    Config& conf = get_config();
    while (!redis_connection_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      log_error(LOGGING_TAG_RPC_TEST, "RPCTest failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    while (!redis_subscriber_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      log_error(LOGGING_TAG_RPC_TEST,
                "RPCBench subscriber failed to connect to redis",
                "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    redis::send_cmd_no_reply<std::string>(
        redis_connection_, {"CONFIG", "SET", "notify-keyspace-events", "AKE"});
    redis::subscribe_to_container_changes(
        redis_subscriber_,
        [this](const std::string &key, const std::string &event_type) {
          if (event_type == "hset") {
            // Detected a new container
            auto container_info =
                redis::get_container_by_key(redis_connection_, key);
            int container_id = std::stoi(container_info["zmq_connection_id"]);
            std::string model_name = container_info["model_name"];
            // Add a validation entry for the new connected container
            // indicating that it has not yet been validated
            std::unique_lock<std::mutex> lock(container_maps_mutex);
            container_names_map_.emplace(container_id, model_name);
            const int validation_msg_id = send_validation_message(container_id);
            msg_id_to_container_map_.emplace(validation_msg_id, container_id);
          }
        });
  }

  bool succeeded() {
    std::unique_lock<std::mutex> lock(container_maps_mutex);
    for(auto const& container_entry : container_validation_map_) {
      bool rpc_valid = container_entry.second.first;
      if(!rpc_valid) {
        return false;
      }
    }
    return true;
  }

  int send_validation_message(int container_id) {
    std::vector<double> data;
    data.push_back(1);
    std::shared_ptr<Input> input = std::make_shared<DoubleVector>(data);
    rpc::PredictionRequest request(InputType::Doubles);
    request.add_input(input);
    auto serialized_request = request.serialize();
    int msg_id = rpc_->send_message(serialized_request, container_id);
    return msg_id;
  }

  std::condition_variable_any test_completed_cv_;
  std::mutex test_completed_cv_mutex_;
  std::atomic<bool> test_completed_;

 private:
  std::unique_ptr<rpc::RPCService> rpc_;
  redox::Subscriber redis_subscriber_;
  redox::Redox redis_connection_;
  std::atomic<int> containers_validated_{0};
  int num_containers_;

  // Mutex used to ensure stability of container-related
  // maps in asynchronous environment
  std::mutex container_maps_mutex;
  // Maintains a mapping between a container's connection id
  // and its associated model's name for debugging purposes
  std::unordered_map<int, std::string> container_names_map_;
  // Maintains a mapping from a container's id to a boolean
  // flag indicating whether or not its RPC protocol is valid
  std::unordered_map<int, RPCValidationResult> container_validation_map_;
  // Mapping of message id to the connection id of its
  // corresponding container
  std::unordered_map<int, int> msg_id_to_container_map_;

  void on_response_received(rpc::RPCResponse response) {
    int msg_id = response.first;
    std::unique_lock<std::mutex> lock(container_maps_mutex);
    auto container_id_entry = msg_id_to_container_map_.find(msg_id);
    if(container_id_entry == msg_id_to_container_map_.end()) {
      // TODO: Throw error
    }
    int container_id = container_id_entry->second;
    auto container_valid_entry = container_validation_map_.find(container_id);
    RPCValidationResult container_rpc_protocol_valid;
    if(container_valid_entry == container_validation_map_.end()) {
      // Container has not yet been validated
      std::vector<uint8_t> msg_history_bytes = response.second;
      size_t message_history_size = static_cast<size_t>(msg_history_bytes.size() / sizeof(int));
      std::vector<int> parsed_message_history;
      float* msg_history_floats = reinterpret_cast<float*>(msg_history_bytes.data());
      for(int i = 0; static_cast<size_t>(i) < message_history_size; i++) {
        int value = static_cast<int>(msg_history_floats[i]);
        parsed_message_history.push_back(value);
      }
      container_rpc_protocol_valid = validate_rpc_protocol(parsed_message_history);
      container_validation_map_[container_id] = container_rpc_protocol_valid;
      containers_validated_ += 1;
      if(containers_validated_ == num_containers_) {
        test_completed_ = true;
        test_completed_cv_.notify_all();
      }
    } else {
      container_rpc_protocol_valid =
          RPCValidationResult(false, "Container sent excessive container content messages (expected 1)");
      container_validation_map_[container_id] = container_rpc_protocol_valid;
    }
    log_validation_result(container_id, container_rpc_protocol_valid);
  }

  RPCValidationResult validate_rpc_protocol(std::vector<int>& rpc_message_history) {
    if(rpc_message_history.size() < MESSAGE_HISTORY_MINIMUM_LENGTH) {
      return RPCValidationResult(false, "Protocol failed to exchange minimally required messages!");
    }
    bool initial_messages_correct =
        (rpc_message_history[0] == MESSAGE_HISTORY_SENT_HEARTBEAT)
            && (rpc_message_history[1] == MESSAGE_HISTORY_RECEIVED_HEARTBEAT)
            && (rpc_message_history[2] == MESSAGE_HISTORY_SENT_CONTAINER_METADATA);
    if(!initial_messages_correct) {
      return RPCValidationResult(false, "Initial protocol messages are of incorrect types!");
    }
    int received_container_content_count = 0;
    for(int i = 3; i < static_cast<int>(rpc_message_history.size()); i++) {
      if(rpc_message_history[i] == MESSAGE_HISTORY_RECEIVED_CONTAINER_CONTENT) {
        received_container_content_count++;
      }
      if(rpc_message_history[i] == MESSAGE_HISTORY_RECEIVED_CONTAINER_METADATA) {
        // The container should never receive container metadata from Clipper
        return RPCValidationResult(false, "Clipper sent an erroneous container metadata message!");
      }
    }
    if(received_container_content_count > 1) {
      std::stringstream ss;
      ss << "Clipper sent excessive container content messages! " << std::endl;
      ss << "Expected: 1, Sent: " << received_container_content_count;
      return RPCValidationResult(false, ss.str());
    } else if(received_container_content_count < 1) {
      // The container definitely received a container content message,
      // but its missing from the log
      return RPCValidationResult(false, "Container log is missing reception of container content message!");
    }
    return RPCValidationResult(true, "");
  }

  void log_validation_result(int container_id, RPCValidationResult& result) {
    std::string container_name = container_names_map_.find(container_id)->second;
    if(result.first) {
      log_info_formatted(LOGGING_TAG_RPC_TEST, "Successfully validated container: \"{}\"", container_name);
    } else {
      log_error_formatted(
          LOGGING_TAG_RPC_TEST, "Failed to validate container: \"{}\". Error: {}", container_name, result.second);
    }
  }
};

int main(int argc, char *argv[]) {
  cxxopts::Options options("rpc_test", "Clipper RPC Correctness Test");
  // clang-format off
  options.add_options()
      ("redis_ip", "Redis address",
       cxxopts::value<std::string>()->default_value("localhost"))
      ("redis_port", "Redis port",
       cxxopts::value<int>()->default_value("6379"))
      ("num_containers", "Number of containers to validate",
       cxxopts::value<int>()->default_value("1"));
  // clang-format on
  options.parse(argc, argv);

  get_config().set_redis_address(options["redis_ip"].as<std::string>());
  get_config().set_redis_port(options["redis_port"].as<int>());
  get_config().ready();

  Tester tester(options["num_containers"].as<int>());
  tester.start();

  std::unique_lock<std::mutex> l(tester.test_completed_cv_mutex_);
  tester.test_completed_cv_.wait(
      l, [&tester]() { return tester.test_completed_ == true; });
  exit(tester.succeeded() ? EXIT_SUCCESS : EXIT_FAILURE);
}