#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "frontend_rpc_service.hpp"

#include <clipper/config.hpp>
#include <clipper/constants.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/exceptions.hpp>
#include <clipper/json_util.hpp>
#include <clipper/logging.hpp>
#include <clipper/metrics.hpp>
#include <clipper/redis.hpp>
#include <clipper/task_executor.hpp>

using namespace clipper;
using clipper::redis::labels_to_str;

namespace zmq_frontend {

const std::string LOGGING_TAG_RPC_FRONTEND = "RPC FRONTEND";

class AppMetrics {
 public:
  explicit AppMetrics(std::string app_name)
      : app_name_(app_name),
        latency_(
            clipper::metrics::MetricsRegistry::get_metrics().create_histogram(
                "app:" + app_name + ":prediction_latency", "microseconds",
                4096)),
        throughput_(
            clipper::metrics::MetricsRegistry::get_metrics().create_meter(
                "app:" + app_name + ":prediction_throughput")),
        num_predictions_(
            clipper::metrics::MetricsRegistry::get_metrics().create_counter(
                "app:" + app_name + ":num_predictions")),
        default_pred_ratio_(
            clipper::metrics::MetricsRegistry::get_metrics()
                .create_ratio_counter("app:" + app_name +
                                      ":default_prediction_ratio")) {}
  ~AppMetrics() = default;

  AppMetrics(const AppMetrics&) = default;

  AppMetrics& operator=(const AppMetrics&) = default;

  AppMetrics(AppMetrics&&) = default;
  AppMetrics& operator=(AppMetrics&&) = default;

  std::string app_name_;
  std::shared_ptr<clipper::metrics::Histogram> latency_;
  std::shared_ptr<clipper::metrics::Meter> throughput_;
  std::shared_ptr<clipper::metrics::Counter> num_predictions_;
  std::shared_ptr<clipper::metrics::RatioCounter> default_pred_ratio_;
};

class ServerImpl {
 public:
  ServerImpl(const std::string ip, int send_port, int recv_port)
      : rpc_service_(std::make_shared<FrontendRPCService>()),
        task_executor_(),
        request_rate_(metrics::MetricsRegistry::get_metrics().create_meter(
            "zmq_frontend:request_rate")) {
    // Start the frontend rpc service
    rpc_service_->start(ip, send_port, recv_port);

    // std::string server_address = address + std::to_string(portno);
    clipper::Config& conf = clipper::get_config();
    while (!redis_connection_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      clipper::log_error(LOGGING_TAG_RPC_FRONTEND,
                         "Query frontend failed to connect to Redis",
                         "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    while (!redis_subscriber_.connect(conf.get_redis_address(),
                                      conf.get_redis_port())) {
      clipper::log_error(LOGGING_TAG_RPC_FRONTEND,
                         "Query frontend subscriber failed to connect to Redis",
                         "Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    clipper::redis::subscribe_to_application_changes(
        redis_subscriber_,
        [this](const std::string& key, const std::string& event_type) {
          clipper::log_info_formatted(
              LOGGING_TAG_RPC_FRONTEND,
              "APPLICATION EVENT DETECTED. Key: {}, event_type: {}", key,
              event_type);
          if (event_type == "hset") {
            std::string name = key;
            clipper::log_info_formatted(LOGGING_TAG_RPC_FRONTEND,
                                        "New application detected: {}", key);
            auto app_info =
                clipper::redis::get_application_by_key(redis_connection_, key);
            DataType input_type =
                clipper::parse_input_type(app_info["input_type"]);
            std::string policy = app_info["policy"];
            std::string default_output = app_info["default_output"];
            int latency_slo_micros = std::stoi(app_info["latency_slo_micros"]);
            add_application(name, input_type, policy, default_output,
                            latency_slo_micros);
          }
        });

    clipper::redis::subscribe_to_model_link_changes(
        redis_subscriber_,
        [this](const std::string& key, const std::string& event_type) {
          std::string app_name = key;
          clipper::log_info_formatted(
              LOGGING_TAG_RPC_FRONTEND,
              "APP LINKS EVENT DETECTED. App name: {}, event_type: {}",
              app_name, event_type);
          if (event_type == "sadd") {
            clipper::log_info_formatted(LOGGING_TAG_RPC_FRONTEND,
                                        "New model link detected for app: {}",
                                        app_name);
            auto linked_model_names =
                clipper::redis::get_linked_models(redis_connection_, app_name);
            set_linked_models_for_app(app_name, linked_model_names);
          }
        });

    clipper::redis::subscribe_to_model_version_changes(
        redis_subscriber_,
        [this](const std::string& key, const std::string& event_type) {
          clipper::log_info_formatted(
              LOGGING_TAG_RPC_FRONTEND,
              "MODEL VERSION CHANGE DETECTED. Key: {}, event_type: {}", key,
              event_type);
          if (event_type == "set") {
            std::string model_name = key;
            boost::optional<std::string> new_version =
                clipper::redis::get_current_model_version(redis_connection_,
                                                          key);
            if (new_version) {
              std::unique_lock<std::mutex> l(current_model_versions_mutex_);
              current_model_versions_[key] = *new_version;
            } else {
              clipper::log_error_formatted(
                  LOGGING_TAG_RPC_FRONTEND,
                  "Model version change for model {} was invalid.", key);
            }
          }
        });

    // Read from Redis configuration tables and update models/applications.
    // (1) Iterate through applications and set up predict/update endpoints.
    std::vector<std::string> app_names =
        clipper::redis::get_all_application_names(redis_connection_);
    for (std::string app_name : app_names) {
      auto app_info =
          clipper::redis::get_application_by_key(redis_connection_, app_name);

      auto linked_model_names =
          clipper::redis::get_linked_models(redis_connection_, app_name);
      set_linked_models_for_app(app_name, linked_model_names);

      DataType input_type = clipper::parse_input_type(app_info["input_type"]);
      std::string policy = app_info["policy"];
      std::string default_output = app_info["default_output"];
      int latency_slo_micros = std::stoi(app_info["latency_slo_micros"]);

      add_application(app_name, input_type, policy, default_output,
                      latency_slo_micros);
    }
    if (app_names.size() > 0) {
      clipper::log_info_formatted(
          LOGGING_TAG_RPC_FRONTEND,
          "Found {} existing applications registered in Clipper: {}.",
          app_names.size(), labels_to_str(app_names));
    }
    // (2) Update current_model_versions_ with (model, version) pairs.
    std::vector<std::string> model_names =
        clipper::redis::get_all_model_names(redis_connection_);
    // Record human-readable model names for logging
    std::vector<std::string> model_names_with_version;
    for (std::string model_name : model_names) {
      auto model_version = clipper::redis::get_current_model_version(
          redis_connection_, model_name);
      if (model_version) {
        std::unique_lock<std::mutex> l(current_model_versions_mutex_);
        current_model_versions_[model_name] = *model_version;
        model_names_with_version.push_back(model_name + "@" + *model_version);
      } else {
        clipper::log_error_formatted(
            LOGGING_TAG_RPC_FRONTEND,
            "Found model {} with missing current version.", model_name);
        throw std::runtime_error("Invalid model version");
      }
    }
    if (model_names.size() > 0) {
      clipper::log_info_formatted(
          LOGGING_TAG_RPC_FRONTEND, "Found {} models deployed to Clipper: {}.",
          model_names.size(), labels_to_str(model_names_with_version));
    }
  }

  ~ServerImpl() {
    redis_connection_.disconnect();
    redis_subscriber_.disconnect();
    rpc_service_->stop();
  }

  void set_linked_models_for_app(std::string name,
                                 std::vector<std::string> models) {
    std::unique_lock<std::mutex> l(linked_models_for_apps_mutex_);
    linked_models_for_apps_[name] = models;
  }

  std::vector<std::string> get_linked_models_for_app(std::string name) {
    std::unique_lock<std::mutex> l(linked_models_for_apps_mutex_);
    return linked_models_for_apps_[name];
  }

  void add_application(std::string name, DataType input_type,
                       std::string policy, std::string default_output,
                       long latency_slo_micros) {
    AppMetrics app_metrics(name);

    auto predict_fn = [
      this, name, application_input_type = input_type, latency_slo_micros,
      app_metrics
    ](FrontendRPCRequest request) {
      std::vector<std::string> models = get_linked_models_for_app(name);
      std::vector<VersionedModelId> versioned_models;
      {
        std::unique_lock<std::mutex> l(current_model_versions_mutex_);
        for (auto m : models) {
          auto version = current_model_versions_.find(m);
          if (version != current_model_versions_.end()) {
            versioned_models.emplace_back(m, version->second);
          }
        }
      }

      int request_id = std::get<1>(request);
      int client_id = std::get<2>(request);
      long query_id = query_counter_.fetch_add(1);
      std::chrono::time_point<std::chrono::system_clock> create_time =
          std::chrono::system_clock::now();

      task_executor_.schedule_prediction(
          PredictTask{std::get<0>(request), versioned_models.front(), 1.0,
                      query_id, latency_slo_micros},
          [this, app_metrics, request_id, client_id,
           create_time](Output output) mutable {
            std::chrono::time_point<std::chrono::system_clock> end =
                std::chrono::system_clock::now();
            long duration_micros =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end - create_time)
                    .count();

            app_metrics.latency_->insert(duration_micros);
            app_metrics.num_predictions_->increment(1);

            rpc_service_->send_response(
                std::make_tuple(std::move(output), request_id, client_id));
          });
    };

    rpc_service_->add_application(name, predict_fn);
  }

  std::string get_metrics() const {
    clipper::metrics::MetricsRegistry& registry =
        clipper::metrics::MetricsRegistry::get_metrics();
    std::string metrics_report = registry.report_metrics();
    // clipper::log_info(LOGGING_TAG_RPC_FRONTEND, "METRICS", metrics_report);
    return metrics_report;
  }

 private:
  std::shared_ptr<FrontendRPCService> rpc_service_;
  TaskExecutor task_executor_;
  std::atomic<long> query_counter_{0};
  redox::Redox redis_connection_;
  redox::Subscriber redis_subscriber_;
  std::mutex current_model_versions_mutex_;
  std::unordered_map<std::string, std::string> current_model_versions_;

  std::mutex linked_models_for_apps_mutex_;
  std::unordered_map<std::string, std::vector<std::string>>
      linked_models_for_apps_;
  std::shared_ptr<metrics::Meter> request_rate_;
};

}  // namespace zmq_frontend
