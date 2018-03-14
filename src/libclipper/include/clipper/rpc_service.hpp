#ifndef CLIPPER_RPC_SERVICE_HPP
#define CLIPPER_RPC_SERVICE_HPP

#include <list>
#include <queue>
#include <string>
#include <vector>

#include <concurrentqueue.h>
#include <boost/bimap.hpp>
#include <redox.hpp>
#include <zmq.hpp>

#include <clipper/containers.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/metrics.hpp>
#include <clipper/util.hpp>

using zmq::socket_t;
using std::string;
using std::shared_ptr;
using std::vector;
using std::list;

namespace clipper {

namespace rpc {

const std::string LOGGING_TAG_RPC = "RPC";

// Tuple of msg_id, data_type, binary data
using RPCResponse = std::tuple<int, DataType, std::shared_ptr<void>>;

// Tuple of query_id, zmq message contents
using RPCRequestItem = std::pair<boost::optional<int>, zmq::message_t>;

/// Tuple of zmq_connection_id, message_id, vector of messages, creation time
using RPCRequest = std::tuple<int, int, std::vector<RPCRequestItem>, long>;

enum class RPCEvent {
  SentHeartbeat = 1,
  ReceivedHeartbeat = 2,
  SentContainerMetadata = 3,
  ReceivedContainerMetadata = 4,
  SentContainerContent = 5,
  ReceivedContainerContent = 6
};

enum class MessageType {
  NewContainer = 0,
  ContainerContent = 1,
  Heartbeat = 2
};

enum class HeartbeatType { KeepAlive = 0, RequestContainerMetadata = 1 };

class RPCService {
 public:
  explicit RPCService();
  ~RPCService();
  // Disallow copy
  RPCService(const RPCService &) = delete;
  RPCService &operator=(const RPCService &) = delete;
  vector<RPCResponse> try_get_responses(const int max_num_responses);
  /**
   * Starts the RPC Service. This must be called explicitly, as it is not
   * invoked during construction.
   */
  void start(
      const string ip, int send_port, int recv_port,
      std::function<void(VersionedModelId, int)> &&container_ready_callback,
      std::function<void(RPCResponse)> &&new_response_callback);
  /**
   * Stops the RPC Service. This is called implicitly within the RPCService
   * destructor.
   */
  void stop();

  int send_message(std::vector<RPCRequestItem> msg,
                   const int zmq_connection_id);

  int send_model_message(std::string model_name,
                         std::vector<RPCRequestItem> msg,
                         const int zmq_connection_id);

 private:
  void manage_send_service(const string address);
  void manage_recv_service(const string address);
  void send_messages(socket_t &socket, int max_num_messages);
  void receive_message(socket_t &socket);

  void handle_new_connection(socket_t &socket, int &zmq_connection_id,
                             std::shared_ptr<redox::Redox> redis_connection);

  void shutdown_service(socket_t &socket);

  std::thread rpc_send_thread_;
  std::thread rpc_recv_thread_;
  shared_ptr<moodycamel::ConcurrentQueue<RPCRequest>> request_queue_;
  shared_ptr<moodycamel::ConcurrentQueue<RPCResponse>> response_queue_;
  // Flag indicating whether rpc service is active
  std::atomic_bool active_;
  // The next available message id
  int message_id_ = 0;
  std::unordered_map<VersionedModelId, int> replica_ids_;
  std::shared_ptr<metrics::Histogram> msg_queueing_hist_;
  std::unordered_map<std::string, std::shared_ptr<metrics::DataList<long>>>
      model_processing_latencies_;
  std::unordered_map<int, std::string> msg_id_models_map_;
  std::unordered_map<int, std::chrono::time_point<std::chrono::system_clock>>
      msg_id_timestamp_map_;

  std::function<void(VersionedModelId, int)> container_ready_callback_;
  std::function<void(RPCResponse)> new_response_callback_;

  // Mapping from zmq_connection_id to routing id (for sending)
  std::unordered_map<int, const std::vector<uint8_t>> connection_routing_map_;
  std::mutex connection_routing_mutex_;

  // Map from zmq_connection_id to container metadata.
  // Values are pairs of model id and integer replica id.
  std::unordered_map<int, std::pair<VersionedModelId, int>>
      connections_containers_map_;
  std::mutex connections_containers_map_mutex_;
};

}  // namespace rpc

}  // namespace clipper

#endif  // CLIPPER_RPC_SERVICE_HPP
