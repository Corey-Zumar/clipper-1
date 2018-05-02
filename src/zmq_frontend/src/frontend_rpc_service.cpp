#include "frontend_rpc_service.hpp"

#include <cstdlib>
#include <mutex>

#include <boost/functional/hash.hpp>
#include <zmq.hpp>

#include <clipper/callback_threadpool.hpp>
#include <clipper/config.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/redis.hpp>
#include <clipper/rpc_service.hpp>
#include <clipper/task_executor.hpp>
#include <clipper/threadpool.hpp>
#include <clipper/util.hpp>

namespace zmq_frontend {

FrontendRPCService::FrontendRPCService()
    : response_queue_(
          std::make_shared<moodycamel::ConcurrentQueue<FrontendRPCResponse>>(
              RESPONSE_QUEUE_SIZE)),
      // prediction_executor_(
      //     std::make_shared<clipper::CallbackThreadPool>("frontend", 15)),
      active_(false),
      request_enqueue_meter_(
          metrics::MetricsRegistry::get_metrics().create_meter(
              "frontend_rpc:request_enqueue")),
      response_enqueue_meter_(
          metrics::MetricsRegistry::get_metrics().create_meter(
              "frontend_rpc:response_enqueue")),
      response_dequeue_meter_(
          metrics::MetricsRegistry::get_metrics().create_meter(
              "frontend_rpc:response_dequeue")),
      recv_latency_(metrics::MetricsRegistry::get_metrics().create_histogram(
          "frontend_rpc:recv_latency", "microseconds", 4096)),
      next_data_offset_(0) {
  std::chrono::time_point<std::chrono::system_clock> start_time =
      std::chrono::system_clock::now();
  recv_data_buffer_ = static_cast<uint8_t *>(std::calloc(1, TOTAL_DATA_BYTES));
  std::chrono::time_point<std::chrono::system_clock> end_time =
      std::chrono::system_clock::now();
  auto calloc_latency = end_time - start_time;
  long calloc_latency_micros =
      std::chrono::duration_cast<std::chrono::microseconds>(calloc_latency)
          .count();

  std::cout << "Memory allocation time (us): "
            << std::to_string(calloc_latency_micros) << std::endl;
}

FrontendRPCService::~FrontendRPCService() {
  stop();
  std::free(recv_data_buffer_);
}

void FrontendRPCService::start(const std::string address, int send_port,
                               int recv_port) {
  active_ = true;
  rpc_send_thread_ = std::thread([this, address, send_port]() {
    manage_send_service(address, send_port);
  });
  rpc_recv_thread_ = std::thread([this, address, recv_port]() {
    manage_recv_service(address, recv_port);
  });
}

void FrontendRPCService::stop() {
  if (active_) {
    active_ = false;
    rpc_send_thread_.join();
    rpc_recv_thread_.join();
  }
}

void FrontendRPCService::add_application(
    std::string name, std::function<void(FrontendRPCRequest)> app_function) {
  std::lock_guard<std::mutex> lock(app_functions_mutex_);
  app_functions_.emplace(name, app_function);
}

void FrontendRPCService::send_response(FrontendRPCResponse response) {
  response_enqueue_meter_->mark(1);
  response_queue_->enqueue(response);
}

void FrontendRPCService::manage_send_service(const std::string ip, int port) {
  std::string send_address = "tcp://" + ip + ":" + std::to_string(port);
  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_ROUTER);
  socket.bind(send_address);
  zmq::pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
  int client_id = 0;
  while (active_) {
    zmq_poll(items, 1, 1);
    if (items[0].revents & ZMQ_POLLIN) {
      handle_new_connection(socket, client_id);
    }
    send_responses(socket, NUM_RESPONSES_SEND);
  }
  shutdown_service(socket);
}

void FrontendRPCService::manage_recv_service(const std::string ip, int port) {
  std::string recv_address = "tcp://" + ip + ":" + std::to_string(port);
  zmq::context_t context(2);
  zmq::socket_t socket(context, ZMQ_ROUTER);
  socket.bind(recv_address);
  zmq::pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
  while (active_) {
    zmq_poll(items, 1, 1);
    if (items[0].revents & ZMQ_POLLIN) {
      receive_request(socket);
    }
  }
  shutdown_service(socket);
}

void FrontendRPCService::handle_new_connection(zmq::socket_t &socket,
                                               int &client_id) {
  zmq::message_t msg_routing_identity;
  zmq::message_t msg_delimiter;
  zmq::message_t msg_establish_connection;

  socket.recv(&msg_routing_identity, 0);
  socket.recv(&msg_delimiter, 0);
  socket.recv(&msg_establish_connection, 0);

  const vector<uint8_t> routing_id(
      (uint8_t *)msg_routing_identity.data(),
      (uint8_t *)msg_routing_identity.data() + msg_routing_identity.size());
  int curr_client_id = client_id;
  client_id++;
  std::lock_guard<std::mutex> lock(client_routing_mutex_);
  client_routing_map_.emplace(curr_client_id, std::move(routing_id));

  zmq::message_t msg_client_id(sizeof(int));
  memcpy(msg_client_id.data(), &curr_client_id, sizeof(int));
  socket.send(msg_routing_identity, ZMQ_SNDMORE);
  socket.send("", 0, ZMQ_SNDMORE);
  socket.send(msg_client_id, 0);
}

void FrontendRPCService::shutdown_service(zmq::socket_t &socket) {
  size_t buf_size = 32;
  std::vector<char> buf(buf_size);
  socket.getsockopt(ZMQ_LAST_ENDPOINT, (void *)buf.data(), &buf_size);
  std::string last_endpoint = std::string(buf.begin(), buf.end());
  socket.unbind(last_endpoint);
  socket.close();
}

void FrontendRPCService::receive_request(zmq::socket_t &socket) {
  zmq::message_t msg_routing_identity;
  zmq::message_t msg_delimiter;
  zmq::message_t msg_client_id;
  zmq::message_t msg_request_id;
  zmq::message_t msg_app_name;
  zmq::message_t msg_data_type;
  zmq::message_t msg_data_size_typed;

  socket.recv(&msg_routing_identity, 0);
  socket.recv(&msg_delimiter, 0);
  socket.recv(&msg_client_id, 0);
  socket.recv(&msg_request_id, 0);
  socket.recv(&msg_app_name, 0);
  socket.recv(&msg_data_type, 0);
  socket.recv(&msg_data_size_typed, 0);

  std::string app_name(static_cast<char *>(msg_app_name.data()),
                       msg_app_name.size());
  DataType input_type =
      static_cast<DataType>(static_cast<int *>(msg_data_type.data())[0]);
  int input_size_typed = static_cast<int *>(msg_data_size_typed.data())[0];

  // NOTE(dcrankshaw): It would make more sense to just have the RPC message
  // include the size in bytes, but to maintain backwards compatibility I'm
  // leaving it as the typed_size for now.
  size_t bytes_per_input;
  switch (input_type) {
    case DataType::Bytes: {
      bytes_per_input = sizeof(uint8_t);
    } break;
    case DataType::Floats: {
      bytes_per_input = sizeof(float);
    } break;
    case DataType::Doubles: {
      bytes_per_input = sizeof(double);
    } break;
    case DataType::Invalid:
    default: {
      std::stringstream ss;
      ss << "Received a request with an input with invalid type: "
         << get_readable_input_type(input_type);
      throw std::runtime_error(ss.str());
    }
  }

  size_t input_size_bytes = ((size_t)input_size_typed) * bytes_per_input;

  std::chrono::time_point<std::chrono::system_clock> start_time =
      std::chrono::system_clock::now();

  // uint8_t *input_buffer =
  //     reinterpret_cast<uint8_t *>(malloc(input_size_bytes));

  uint8_t *input_buffer =
      reinterpret_cast<uint8_t *>(alloc_data(input_size_bytes));

  socket.recv(input_buffer, input_size_bytes, 0);
  std::chrono::time_point<std::chrono::system_clock> recv_end_time =
      std::chrono::system_clock::now();
  InputVector input(input_buffer, input_size_typed, input_size_bytes,
                    input_type);

  auto recv_latency = recv_end_time - start_time;
  long recv_latency_micros =
      std::chrono::duration_cast<std::chrono::microseconds>(recv_latency)
          .count();
  recv_latency_->insert(recv_latency_micros);

  std::lock_guard<std::mutex> lock(app_functions_mutex_);
  auto app_functions_search = app_functions_.find(app_name);
  if (app_functions_search == app_functions_.end()) {
    log_error_formatted(
        LOGGING_TAG_ZMQ_FRONTEND,
        "Received a request for an unknown application with name {}", app_name);
  } else {
    auto app_function = app_functions_search->second;

    int request_id = static_cast<int *>(msg_request_id.data())[0];

    int client_id = static_cast<int *>(msg_client_id.data())[0];

    // The app_function should be very cheap, it's just constructing
    // an object and putting it on the task queue. Try executing it directly.
    app_function(std::make_tuple(input, request_id, client_id));

    request_enqueue_meter_->mark(1);
    // prediction_executor_->submit(
    //     [app_function, input, request_id, client_id]() {
    //       app_function(std::make_tuple(input, request_id, client_id));
    //     });
  }
}

void FrontendRPCService::send_responses(zmq::socket_t &socket,
                                        size_t num_responses) {
  FrontendRPCResponse response;
  size_t sent_responses = 0;
  while (sent_responses < num_responses &&
         response_queue_->try_dequeue(response)) {
    response_dequeue_meter_->mark(1);
    Output &output = std::get<0>(response);
    int request_id = std::get<1>(response);
    int client_id = std::get<2>(response);

    std::lock_guard<std::mutex> routing_lock(client_routing_mutex_);
    auto routing_id_search = client_routing_map_.find(client_id);
    if (routing_id_search == client_routing_map_.end()) {
      std::stringstream ss;
      ss << "Received a response associated with a client id " << client_id
         << " that has no associated routing identity";
      throw std::runtime_error(ss.str());
    }

    const std::vector<uint8_t> &routing_id = routing_id_search->second;

    int output_type = static_cast<int>(output.y_hat_->type());

    // TODO(czumar): If this works, include other relevant output data (default
    // bool, default expl, etc)
    socket.send(routing_id.data(), routing_id.size(), ZMQ_SNDMORE);
    socket.send("", 0, ZMQ_SNDMORE);
    socket.send(&request_id, sizeof(int), ZMQ_SNDMORE);
    socket.send(&output_type, sizeof(int), ZMQ_SNDMORE);
    socket.send(output.y_hat_->get_data(), output.y_hat_->byte_size());

    sent_responses += 1;
  }
}

// WARNING: THIS IS A QUICK AND DIRTY HACK. IT'S TOTALLY NOT SAFE TO ACTUALLY
// USE.
void *FrontendRPCService::alloc_data(size_t size_bytes) {
  if (size_bytes > TOTAL_DATA_BYTES) {
    throw std::runtime_error("Requested a memory allocation that was too big");
  }
  std::lock_guard<std::mutex> l(data_mutex_);
  // Check if we've reached end of buffer and need to wrap back
  if ((next_data_offset_ + size_bytes) > TOTAL_DATA_BYTES) {
    std::cout << "Wrapping around to front of buffer" << std::endl;
    next_data_offset_ = 0;
  }

  void *alloc_ptr = static_cast<void *>(recv_data_buffer_ + next_data_offset_);
  next_data_offset_ += size_bytes;
  return alloc_ptr;
}

}  // namespace zmq_frontend
