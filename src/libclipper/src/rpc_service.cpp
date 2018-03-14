#include <boost/bimap.hpp>
#include <boost/functional/hash.hpp>

#include <chrono>
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
    : request_queue_(std::make_shared<moodycamel::ConcurrentQueue<RPCRequest>>(
          sizeof(RPCRequest) * 10000)),
      response_queue_(
          std::make_shared<moodycamel::ConcurrentQueue<RPCResponse>>(
              sizeof(RPCResponse) * 10000)),
      active_(false),
      // The version of the unordered_map constructor that allows
      // you to specify your own hash function also requires you
      // to provide the initial size of the map. We define the initial
      // size of the map somewhat arbitrarily as 100.
      replica_ids_(std::unordered_map<VersionedModelId, int>({})) {
  msg_queueing_hist_ = metrics::MetricsRegistry::get_metrics().create_histogram(
      "internal:rpc_request_queueing_delay", "microseconds", 2056);
}

RPCService::~RPCService() { stop(); }

void RPCService::start(
    const string ip, int send_port, int recv_port,
    std::function<void(VersionedModelId, int)> &&container_ready_callback,
    std::function<void(RPCResponse)> &&new_response_callback) {
  container_ready_callback_ = container_ready_callback;
  new_response_callback_ = new_response_callback;
  if (active_) {
    throw std::runtime_error(
        "Attempted to start RPC Service when it is already running!");
  }
  // 7000
  const string send_address = "tcp://" + ip + ":" + std::to_string(send_port);
  // 7001
  const string recv_address = "tcp://" + ip + ":" + std::to_string(recv_port);
  active_ = true;
  rpc_send_thread_ = std::thread(
      [this, send_address]() { manage_send_service(send_address); });
  rpc_recv_thread_ = std::thread(
      [this, recv_address]() { manage_recv_service(recv_address); });
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
  while (!redis_connection->connect(conf.get_redis_address(),
                                    conf.get_redis_port())) {
    log_error(LOGGING_TAG_RPC, "RPCService failed to connect to Redis",
              "Retrying in 1 second...");
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

int RPCService::send_message(std::vector<RPCRequestItem> msg,
                             const int zmq_connection_id) {
  if (!active_) {
    log_error(LOGGING_TAG_RPC,
              "Cannot send message to inactive RPCService instance",
              "Dropping Message");
    return -1;
  }
  int id = message_id_;
  message_id_ += 1;
  long current_time_micros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  RPCRequest request(zmq_connection_id, id, std::move(msg),
                     current_time_micros);
  request_queue_->enqueue(std::move(request));
  return id;
}

int RPCService::send_model_message(std::string model_name,
                                   std::vector<RPCRequestItem> msg,
                                   const int zmq_connection_id) {
  // Duplicated code in order to avoid potential race conditions
  if (!active_) {
    log_error(LOGGING_TAG_RPC,
              "Cannot send message to inactive RPCService instance",
              "Dropping Message");
    return -1;
  }
  int id = message_id_;
  message_id_ += 1;
  long current_time_micros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  RPCRequest request(zmq_connection_id, id, std::move(msg),
                     current_time_micros);
  auto model_metrics_search = model_processing_latencies_.find(model_name);
  if (model_metrics_search == model_processing_latencies_.end()) {
    auto model_metric =
        metrics::MetricsRegistry::get_metrics().create_data_list<long>(
            model_name + ":processing_latency", "milliseconds");
    model_processing_latencies_.emplace(model_name, model_metric);
  }

  msg_id_models_map_.emplace(id, model_name);
  request_queue_->enqueue(std::move(request));
  return id;
}

vector<RPCResponse> RPCService::try_get_responses(const int max_num_responses) {
  std::vector<RPCResponse> vec(response_queue_->size_approx());
  size_t num_dequeued =
      response_queue_->try_dequeue_bulk(vec.begin(), vec.size());
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
  size_t num_requests =
      request_queue_->try_dequeue_bulk(requests.begin(), max_num_messages);

  for (size_t i = 0; i < num_requests; i++) {
    RPCRequest &request = requests[i];

    int zmq_connection_id = std::get<0>(request);
    std::lock_guard<std::mutex> routing_lock(connection_routing_mutex_);
    auto routing_id_search = connection_routing_map_.find(zmq_connection_id);
    if (routing_id_search == connection_routing_map_.end()) {
      std::stringstream ss;
      ss << "Received a send request associated with a client id "
         << zmq_connection_id << " that has no associated routing identity";
      throw std::runtime_error(ss.str());
    }
    const std::vector<uint8_t> &routing_id = routing_id_search->second;
    message_t type_message(sizeof(int));
    static_cast<int *>(type_message.data())[0] =
        static_cast<int>(MessageType::ContainerContent);
    message_t id_message(sizeof(int));
    memcpy(id_message.data(), &std::get<1>(request), sizeof(int));

    long long curr_system_time = clock::ClipperClock.get_clock().get_uptime();

    socket.send(routing_id.data(), routing_id.size(), ZMQ_SNDMORE);
    socket.send("", 0, ZMQ_SNDMORE);
    socket.send(type_message, ZMQ_SNDMORE);
    socket.send(id_message, ZMQ_SNDMORE);
    int cur_msg_num = 0;
    // subtract 1 because we start counting at 0
    int last_msg_num = std::get<2>(request).size() - 1;
    for (RPCRequestItem &cur_item : std::get<2>(request)) {
      // message_t cur_buffer(m.first, m.second, noop_free);
      // send the sndmore flag unless we are on the last message part
      boost::optional<int> &query_id = cur_item.first;
      if (query_id) {
        metrics::TSLineageTracker::get_tracker().add_entry(
            query_id.get(), curr_system_time, "QUERY SENT VIA RPC");
      }
      zmq::message_t &cur_msg = cur_item.second;
      if (cur_msg_num < last_msg_num) {
        socket.send(cur_msg, ZMQ_SNDMORE);
      } else {
        socket.send(cur_msg);
      }
      cur_msg_num += 1;
    }

    int msg_id = std::get<1>(request);
    auto outbound_timestamp = std::chrono::system_clock::now();
    msg_id_timestamp_map_.emplace(msg_id, std::move(outbound_timestamp));
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

  MessageType type =
      static_cast<MessageType>(static_cast<int *>(msg_type.data())[0]);

  int zmq_connection_id = static_cast<int *>(msg_zmq_connection_id.data())[0];
  if (type != MessageType::ContainerContent) {
    throw std::runtime_error("Received wrong message type");
  }
  // This message is a response to a container query
  message_t msg_id;
  message_t msg_content_type;
  message_t msg_content_size;
  message_t msg_content;
  socket.recv(&msg_id, 0);
  socket.recv(&msg_content_type, 0);
  socket.recv(&msg_content_size, 0);

  DataType content_data_type =
      static_cast<DataType>(static_cast<int *>(msg_content_type.data())[0]);
  uint32_t content_size = static_cast<uint32_t *>(msg_content_size.data())[0];

  std::shared_ptr<void> msg_content_buffer(malloc(content_size), free);

  socket.recv(msg_content_buffer.get(), content_size, 0);
  log_info(LOGGING_TAG_RPC, "response received");
  int id = static_cast<int *>(msg_id.data())[0];
  RPCResponse response(id, content_data_type, msg_content_buffer);

  std::lock_guard<std::mutex> connections_container_map_lock(
      connections_containers_map_mutex_);
  auto container_info_entry =
      connections_containers_map_.find(zmq_connection_id);
  if (container_info_entry == connections_containers_map_.end()) {
    std::stringstream ss;
    ss << "Failed to find container with ID " << zmq_connection_id;
    ss << " that was previously registered via RPC.";
    throw std::runtime_error(ss.str());
  }

  auto outbound_timestamp = msg_id_timestamp_map_.find(id)->second;
  std::string model_name = msg_id_models_map_.find(id)->second;
  auto model_latencies_list =
      model_processing_latencies_.find(model_name)->second;

  auto inbound_timestamp = std::chrono::system_clock::now();
  long model_processing_latency =
      std::chrono::duration_cast<std::chrono::milliseconds>(inbound_timestamp -
                                                            outbound_timestamp)
          .count();

  model_latencies_list->insert(model_processing_latency);
  msg_id_timestamp_map_.erase(id);
  msg_id_models_map_.erase(id);

  std::pair<VersionedModelId, int> container_info =
      container_info_entry->second;

  VersionedModelId vm = container_info.first;
  int replica_id = container_info.second;
  TaskExecutionThreadPool::submit_job(vm, replica_id, new_response_callback_,
                                      response);
  TaskExecutionThreadPool::submit_job(vm, replica_id, container_ready_callback_,
                                      vm, replica_id);

  response_queue_->enqueue(response);
}

void RPCService::handle_new_connection(
    socket_t &socket, int &zmq_connection_id,
    std::shared_ptr<redox::Redox> redis_connection) {
  std::cout << "New connection detected" << std::endl;

  message_t msg_routing_identity;
  message_t msg_delimiter;
  message_t msg_type;

  socket.recv(&msg_routing_identity, 0);
  socket.recv(&msg_delimiter, 0);
  socket.recv(&msg_type, 0);

  MessageType type =
      static_cast<MessageType>(static_cast<int *>(msg_type.data())[0]);

  if (type != MessageType::NewContainer) {
    std::stringstream ss;
    ss << "Wrong message type in RPCService::HandleNewConnection. Expected ";
    ss << static_cast<std::underlying_type<MessageType>::type>(
        MessageType::NewContainer);
    ss << ". Found "
       << static_cast<std::underlying_type<MessageType>::type>(type);
    throw std::runtime_error(ss.str());
  }

  const vector<uint8_t> routing_id(
      (uint8_t *)msg_routing_identity.data(),
      (uint8_t *)msg_routing_identity.data() + msg_routing_identity.size());

  int curr_zmq_connection_id = zmq_connection_id;
  std::lock_guard<std::mutex> lock(connection_routing_mutex_);
  connection_routing_map_.emplace(curr_zmq_connection_id,
                                  std::move(routing_id));

  message_t model_name;
  message_t model_version;
  message_t model_input_type;
  socket.recv(&model_name, 0);
  socket.recv(&model_version, 0);
  socket.recv(&model_input_type, 0);

  std::string name(static_cast<char *>(model_name.data()), model_name.size());
  std::string version(static_cast<char *>(model_version.data()),
                      model_version.size());
  std::string input_type_str(static_cast<char *>(model_input_type.data()),
                             model_input_type.size());

  DataType input_type = static_cast<DataType>(std::stoi(input_type_str));

  VersionedModelId model = VersionedModelId(name, version);

  // Note that if the map does not have an entry for this model,
  // a new entry will be created with the default value (0).
  // This use of operator[] avoids the boilerplate of having to
  // check if the key is present in the map.
  int cur_replica_id = replica_ids_[model];
  replica_ids_[model] = cur_replica_id + 1;
  redis::add_container(*redis_connection, model, cur_replica_id,
                       curr_zmq_connection_id, input_type);
  std::lock_guard<std::mutex> connections_container_map_lock(
      connections_containers_map_mutex_);
  connections_containers_map_.emplace(
      curr_zmq_connection_id,
      std::pair<VersionedModelId, int>(model, cur_replica_id));

  TaskExecutionThreadPool::create_queue(model, cur_replica_id);

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
