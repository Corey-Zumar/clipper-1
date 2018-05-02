#include "zmq_frontend.hpp"
// #include "zmq_frontend_no_queries.hpp"

#include <clipper/config.hpp>
#include <clipper/constants.hpp>
#include <cxxopts.hpp>
#include <server_http.hpp>

using HttpServer = SimpleWeb::Server<SimpleWeb::HTTP>;

const std::string GET_METRICS = "^/metrics$";

void respond_http(std::string content, std::string message,
                  std::shared_ptr<HttpServer::Response> response) {
  *response << "HTTP/1.1 " << message << "\r\nContent-Type: application/json"
            << "\r\nContent-Length: " << content.length() << "\r\n\r\n"
            << content << "\n";
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("zmq_frontend", "Clipper query processing frontend");
  // clang-format off
  options.add_options()
      ("redis_ip", "Redis address",
       cxxopts::value<std::string>()->default_value("localhost"))
      ("redis_port", "Redis port",
       cxxopts::value<int>()->default_value("6379"))
      ("num_rpc_threads_size", "Number of threads for the grpc frontend",
       cxxopts::value<int>()->default_value("2"))
      ("rpc_recv_max", "", cxxopts::value<int>()->default_value("1"))
      ("rpc_send_max", "", cxxopts::value<int>()->default_value("-1"));
  // clang-format on
  options.parse(argc, argv);

  clipper::Config& conf = clipper::get_config();
  conf.set_redis_address(options["redis_ip"].as<std::string>());
  conf.set_redis_port(options["redis_port"].as<int>());
  conf.set_rpc_max_recv(options["rpc_recv_max"].as<int>());
  conf.set_rpc_max_send(options["rpc_send_max"].as<int>());
  // conf.set_task_execution_threadpool_size(options["threadpool_size"].as<int>());
  conf.ready();

  zmq_frontend::ServerImpl zmq_server("0.0.0.0", 4455, 4456);

  HttpServer metrics_server("0.0.0.0", clipper::QUERY_FRONTEND_PORT, 1);

  metrics_server.add_endpoint(
      GET_METRICS, "GET", [](std::shared_ptr<HttpServer::Response> response,
                             std::shared_ptr<HttpServer::Request> /*request*/) {
        clipper::metrics::MetricsRegistry& registry =
            clipper::metrics::MetricsRegistry::get_metrics();
        std::string metrics_report =
            // registry.report_metrics(false);
            registry.report_metrics(true);
        std::cout << metrics_report << std::endl;
        respond_http(metrics_report, "200 OK", response);
      });
  metrics_server.start();
}
