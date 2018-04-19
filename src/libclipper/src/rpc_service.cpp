#include <boost/bimap.hpp>
#include <boost/functional/hash.hpp>

#include <chrono>
#include <cmath>
#include <iostream>

#include <concurrentqueue.h>
#include <redox.hpp>

#include <clipper/config.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/redis.hpp>
#include <clipper/rpc_service.hpp>
#include <clipper/task_executor.hpp>
#include <clipper/threadpool.hpp>
#include <clipper/util.hpp>

using zmq::socket_t;
using zmq::message_t;
using zmq::context_t;
using std::shared_ptr;
using std::string;
using std::vector;

namespace clipper {

namespace rpc {

constexpr int INITIAL_REPLICA_ID_SIZE = 100;

RPCService::RPCService()
    : request_queue_(
          std::make_shared<moodycamel::ConcurrentQueue<RPCRequest>>(sizeof(RPCRequest) * 10000)),
      response_queue_(
          std::make_shared<moodycamel::ConcurrentQueue<RPCResponse>>(sizeof(RPCResponse) * 10000)),
      active_(false),
      // The version of the unordered_map constructor that allows
      // you to specify your own hash function also requires you
      // to provide the initial size of the map. We define the initial
      // size of the map somewhat arbitrarily as 100.
      replica_ids_(std::unordered_map<VersionedModelId, int>({})) {
  msg_queueing_hist_ = metrics::MetricsRegistry::get_metrics().create_histogram(
      "internal:rpc_request_queueing_delay", "microseconds", 2056);
  // model_send_times_ =
  // metrics::MetricsRegistry::get_metrics().create_data_list<long
  // long>("send_times", "timestamp");
}

RPCService::~RPCService() { stop(); }

void RPCService::start(const string ip, int send_port, int recv_port,
                       std::function<void(ContainerId)> &&container_ready_callback,
                       std::function<void(RPCResponse)> &&new_response_callback) {
  container_ready_callback_ = container_ready_callback;
  new_response_callback_ = new_response_callback;
  if (active_) {
    throw std::runtime_error("Attempted to start RPC Service when it is already running!");
  }
  // 7000
  const string send_address = "tcp://" + ip + ":" + std::to_string(send_port);
  // 7001
  const string recv_address = "tcp://" + ip + ":" + std::to_string(recv_port);
  active_ = true;
  rpc_send_thread_ = std::thread([this, send_address]() { manage_send_service(send_address); });
  rpc_recv_thread_ = std::thread([this, recv_address]() { manage_recv_service(recv_address); });
}

void RPCService::manage_send_service(const string address) {
  context_t context = context_t(1);
  socket_t socket = socket_t(context, ZMQ_ROUTER);
  socket.bind(address);
  // Indicate that we will poll our zmq service socket for new inbound messages
  zmq::pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
  int zmq_connection_id = 0;

  auto redis_connection = std::make_shared<redox::Redox>();
  Config &conf = get_config();
  while (!redis_connection->connect(conf.get_redis_address(), conf.get_redis_port())) {
    log_error(LOGGING_TAG_RPC, "RPCService failed to connect to Redis", "Retrying in 1 second...");
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  int num_send = conf.get_rpc_max_send();
  while (active_) {
    zmq_poll(items, 1, 0);
    if (items[0].revents & ZMQ_POLLIN) {
      handle_new_connection(socket, zmq_connection_id, redis_connection);
    }
    send_messages(socket, num_send);
  }
  shutdown_service(socket);
}

void RPCService::manage_recv_service(const string address) {
  context_t context = context_t(1);
  socket_t socket = socket_t(context, ZMQ_ROUTER);
  socket.bind(address);
  // Indicate that we will poll our zmq service socket for new inbound messages
  zmq::pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};

  while (active_) {
    zmq_poll(items, 1, 1);
    if (items[0].revents & ZMQ_POLLIN) {
      receive_message(socket);
    }
  }
  shutdown_service(socket);
}

void RPCService::stop() {
  if (active_) {
    active_ = false;
    rpc_send_thread_.join();
    rpc_recv_thread_.join();
  }
}

uint32_t RPCService::send_message(std::vector<zmq::message_t> items, const int zmq_connection_id) {
  if (!active_) {
    log_error(LOGGING_TAG_RPC, "Cannot send message to inactive RPCService instance",
              "Dropping Message");
    return -1;
  }
  int id = message_id_;
  message_id_ += 1;
  long current_time_micros = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count();
  RPCRequest request(zmq_connection_id, id, std::move(items), current_time_micros);
  request_queue_->enqueue(std::move(request));
  return id;
}

vector<RPCResponse> RPCService::try_get_responses(const int max_num_responses) {
  std::vector<RPCResponse> vec(response_queue_->size_approx());
  size_t num_dequeued = response_queue_->try_dequeue_bulk(vec.begin(), vec.size());
  vec.resize(num_dequeued);
  return vec;
}

void RPCService::shutdown_service(socket_t &socket) {
  size_t buf_size = 32;
  std::vector<char> buf(buf_size);
  socket.getsockopt(ZMQ_LAST_ENDPOINT, (void *)buf.data(), &buf_size);
  std::string last_endpoint = std::string(buf.begin(), buf.end());
  socket.unbind(last_endpoint);
  socket.close();
}

void noop_free(void *data, void *hint) {}

void real_free(void *data, void *hint) { free(data); }

void RPCService::send_messages(socket_t &socket, int max_num_messages) {
  if (max_num_messages == -1) {
    max_num_messages = request_queue_->size_approx();
  }

  std::vector<RPCRequest> requests(max_num_messages);
  size_t num_requests = request_queue_->try_dequeue_bulk(requests.begin(), max_num_messages);

  for (size_t i = 0; i < num_requests; i++) {
    RPCRequest &request = requests[i];

    int zmq_connection_id = std::get<0>(request);
    std::lock_guard<std::mutex> routing_lock(connection_routing_mutex_);
    auto routing_id_search = connection_routing_map_.find(zmq_connection_id);
    if (routing_id_search == connection_routing_map_.end()) {
      std::stringstream ss;
      ss << "Received a send request associated with a client id " << zmq_connection_id
         << " that has no associated routing identity";
      throw std::runtime_error(ss.str());
    }
    const std::vector<uint8_t> &routing_id = routing_id_search->second;
    message_t type_message(sizeof(int));
    static_cast<int *>(type_message.data())[0] = static_cast<int>(MessageType::ContainerContent);
    message_t id_message(sizeof(int));
    memcpy(id_message.data(), &std::get<1>(request), sizeof(int));

    socket.send(routing_id.data(), routing_id.size(), ZMQ_SNDMORE);
    socket.send("", 0, ZMQ_SNDMORE);
    socket.send(type_message, ZMQ_SNDMORE);
    socket.send(id_message, ZMQ_SNDMORE);
    int cur_msg_num = 0;
    // subtract 1 because we start counting at 0
    int last_msg_num = std::get<2>(request).size() - 1;
    for (auto &cur_msg : std::get<2>(request)) {
      // send the sndmore flag unless we are on the last message part
      if (cur_msg_num < last_msg_num) {
        socket.send(cur_msg, ZMQ_SNDMORE);
      } else {
        socket.send(cur_msg);
      }
      cur_msg_num += 1;
    }
  }
}

void RPCService::receive_message(socket_t &socket) {
  message_t msg_routing_identity;
  message_t msg_delimiter;
  message_t msg_zmq_connection_id;
  message_t msg_type;
  socket.recv(&msg_routing_identity, 0);
  socket.recv(&msg_delimiter, 0);
  socket.recv(&msg_zmq_connection_id, 0);
  socket.recv(&msg_type, 0);

  MessageType type = static_cast<MessageType>(static_cast<int *>(msg_type.data())[0]);

  int zmq_connection_id = static_cast<int *>(msg_zmq_connection_id.data())[0];
  if (type != MessageType::ContainerContent) {
    throw std::runtime_error("Received wrong message type");
  }
  // This message is a response to a container query
  message_t msg_id;
  message_t msg_output_header;

  socket.recv(&msg_id, 0);
  socket.recv(&msg_output_header, 0);

  uint64_t *output_header = static_cast<uint64_t *>(msg_output_header.data());
  uint64_t num_outputs = output_header[0];
  output_header += 1;

  uint64_t output_data_size =
      static_cast<uint64_t>(std::accumulate(output_header, output_header + num_outputs, 0));

  std::shared_ptr<void> output_data(malloc(output_data_size), free);
  uint8_t *output_data_raw = static_cast<uint8_t *>(output_data.get());

  std::unordered_map<uint32_t, std::shared_ptr<OutputData>> outputs;
  outputs.reserve(num_outputs);

  uint64_t curr_start = 0;
  for (uint64_t i = 0; i < (num_outputs * 3); i += 3) {
    uint64_t output_size = output_header[i];
    uint32_t output_batch_id = output_header[i + 1];
    DataType output_type = static_cast<DataType>(output_header[i + 2]);

    socket.recv(output_data_raw + curr_start, output_size, 0);
    outputs.emplace(output_batch_id, OutputData::create_output(output_type, output_data, curr_start,
                                                               curr_start + output_size));

    curr_start += output_size;
    output_data_raw += output_size;
  }

  log_info(LOGGING_TAG_RPC, "response received");
  uint32_t id = static_cast<uint32_t *>(msg_id.data())[0];
  RPCResponse response(id, std::move(outputs));

  std::lock_guard<std::mutex> connections_container_map_lock(connections_containers_map_mutex_);
  auto container_info_entry = connections_containers_map_.find(zmq_connection_id);
  if (container_info_entry == connections_containers_map_.end()) {
    std::stringstream ss;
    ss << "Failed to find container with ID " << zmq_connection_id;
    ss << " that was previously registered via RPC.";
    throw std::runtime_error(ss.str());
  }

  std::vector<ContainerModelDataItem> container_info = container_info_entry->second;
  ContainerId container_id = get_container_id(container_info);

  TaskExecutionThreadPool::submit_job(container_id, new_response_callback_, response);
  TaskExecutionThreadPool::submit_job(container_id, container_ready_callback_, container_id);

  response_queue_->enqueue(response);
}

void RPCService::handle_new_connection(socket_t &socket, int &zmq_connection_id,
                                       std::shared_ptr<redox::Redox> redis_connection) {
  std::cout << "New connection detected" << std::endl;

  message_t msg_routing_identity;
  message_t msg_delimiter;
  message_t msg_type;

  socket.recv(&msg_routing_identity, 0);
  socket.recv(&msg_delimiter, 0);
  socket.recv(&msg_type, 0);

  MessageType type = static_cast<MessageType>(static_cast<int *>(msg_type.data())[0]);

  if (type != MessageType::NewContainer) {
    std::stringstream ss;
    ss << "Wrong message type in RPCService::HandleNewConnection. Expected ";
    ss << static_cast<std::underlying_type<MessageType>::type>(MessageType::NewContainer);
    ss << ". Found " << static_cast<std::underlying_type<MessageType>::type>(type);
    throw std::runtime_error(ss.str());
  }

  const vector<uint8_t> routing_id(
      (uint8_t *)msg_routing_identity.data(),
      (uint8_t *)msg_routing_identity.data() + msg_routing_identity.size());

  int curr_zmq_connection_id = zmq_connection_id;
  std::lock_guard<std::mutex> lock(connection_routing_mutex_);
  connection_routing_map_.emplace(curr_zmq_connection_id, std::move(routing_id));

  message_t msg_num_models;
  socket.recv(&msg_num_models, 0);
  uint32_t num_models = static_cast<uint32_t *>(msg_num_models.data())[0];

  std::vector<ContainerModelDataItem> container_models;
  container_models.reserve(num_models);

  for (size_t i = 0; i < num_models; ++i) {
    message_t msg_model_name;
    message_t msg_model_version;

    socket.recv(&msg_model_name, 0);
    socket.recv(&msg_model_version, 0);

    std::string model_name(static_cast<char *>(msg_model_name.data()), msg_model_name.size());
    std::string model_version(static_cast<char *>(msg_model_version.data()),
                              msg_model_version.size());

    VersionedModelId model_id(model_name, model_version);

    // Note that if the map does not have an entry for this model,
    // a new entry will be created with the default value (0).
    // This use of operator[] avoids the boilerplate of having to
    // check if the key is present in the map.
    int model_replica_id = replica_ids_[model_id];
    replica_ids_[model_id] = model_replica_id + 1;
    container_models.push_back(std::make_pair(model_id, model_replica_id));
  }

  redis::add_container(*redis_connection, container_models, curr_zmq_connection_id);
  std::lock_guard<std::mutex> connections_container_map_lock(connections_containers_map_mutex_);
  connections_containers_map_.emplace(curr_zmq_connection_id, container_models);

  size_t container_id = get_container_id(container_models);

  TaskExecutionThreadPool::create_queue(container_id);

  zmq::message_t msg_zmq_connection_id(sizeof(int));
  memcpy(msg_zmq_connection_id.data(), &curr_zmq_connection_id, sizeof(int));
  socket.send(msg_routing_identity, ZMQ_SNDMORE);
  socket.send("", 0, ZMQ_SNDMORE);
  socket.send(msg_zmq_connection_id, 0);
  zmq_connection_id += 1;
}

// void RPCService::send_heartbeat_response(socket_t &socket,
//                                          const vector<uint8_t>
//                                          &connection_id,
//                                          bool request_container_metadata) {
//   message_t type_message(sizeof(int));
//   message_t heartbeat_type_message(sizeof(int));
//   static_cast<int *>(type_message.data())[0] =
//       static_cast<int>(MessageType::Heartbeat);
//   static_cast<int *>(heartbeat_type_message.data())[0] = static_cast<int>(
//       request_container_metadata ? HeartbeatType::RequestContainerMetadata
//                                  : HeartbeatType::KeepAlive);
//   socket.send(connection_id.data(), connection_id.size(), ZMQ_SNDMORE);
//   socket.send("", 0, ZMQ_SNDMORE);
//   socket.send(type_message, ZMQ_SNDMORE);
//   socket.send(heartbeat_type_message);
// }

}  // namespace rpc

}  // namespace clipper
