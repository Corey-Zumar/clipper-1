#ifndef CLIPPER_FRONTEND_RPC_SERVICE_HPP
#define CLIPPER_FRONTEND_RPC_SERVICE_HPP

#include <mutex>

// #include <folly/ProducerConsumerQueue.h>
#include <concurrentqueue.h>
#include <clipper/callback_threadpool.hpp>
#include <clipper/datatypes.hpp>
#include <zmq.hpp>

namespace zmq_frontend {

using namespace clipper;

const std::string LOGGING_TAG_ZMQ_FRONTEND = "ZMQ_FRONTEND";

// We may have up to 50,000 outstanding requests
constexpr size_t RESPONSE_QUEUE_SIZE = 50000;
constexpr size_t NUM_REQUESTS_RECV = 100;
constexpr size_t NUM_RESPONSES_SEND = 100;

constexpr size_t TOTAL_DATA_BYTES = 299 * 299 * 3 * sizeof(float) * 50000;

// Tuple of input, request id, client id
typedef std::tuple<InputVector, int, int> FrontendRPCRequest;
// Tuple of output, request id, client id. Request id and client ids
// should match corresponding ids of a FrontendRPCRequest object
typedef std::tuple<Output, int, int> FrontendRPCResponse;

class FrontendRPCService {
 public:
  FrontendRPCService();
  ~FrontendRPCService();

  FrontendRPCService(const FrontendRPCService &) = delete;
  FrontendRPCService &operator=(const FrontendRPCService &) = delete;

  void start(const std::string address, int send_port, int recv_port);
  void stop();
  void send_response(FrontendRPCResponse response);
  void add_application(std::string name,
                       std::function<void(FrontendRPCRequest)> app_function);

 private:
  void manage_send_service(const std::string ip, int port);
  void manage_recv_service(const std::string ip, int port);
  void handle_new_connection(zmq::socket_t &socket, int &client_id);
  void shutdown_service(zmq::socket_t &socket);
  void receive_request(zmq::socket_t &socket);
  void send_responses(zmq::socket_t &socket, size_t num_responses);

  std::shared_ptr<moodycamel::ConcurrentQueue<FrontendRPCResponse>>
      response_queue_;
  // std::shared_ptr<clipper::CallbackThreadPool> prediction_executor_;
  std::atomic_bool active_;
  std::mutex app_functions_mutex_;
  std::mutex client_routing_mutex_;
  // Mapping from app name to prediction function
  std::unordered_map<std::string, std::function<void(FrontendRPCRequest)>>
      app_functions_;
  // Mapping from client id to routing id
  std::unordered_map<size_t, const std::vector<uint8_t>> client_routing_map_;
  std::thread rpc_send_thread_;
  std::thread rpc_recv_thread_;

  std::shared_ptr<metrics::Meter> request_enqueue_meter_;

  std::shared_ptr<metrics::Meter> response_enqueue_meter_;
  std::shared_ptr<metrics::Meter> response_dequeue_meter_;
  // std::shared_ptr<metrics::Histogram> malloc_latency_;
  std::shared_ptr<metrics::Histogram> recv_latency_;

  void *alloc_data(size_t size_bytes);

  std::mutex data_mutex_;
  size_t next_data_offset_;
  uint8_t *recv_data_buffer_;
};

}  // namespace zmq_frontend

#endif  // CLIPPER_FRONTEND_RPC_SERVICE_HPP
