#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include <boost/functional/hash.hpp>
#include <clipper/constants.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/logging.hpp>
#include <clipper/util.hpp>

namespace clipper {

template <typename T>
size_t serialize_to_buffer(const std::shared_ptr<T> &data, const size_t size,
                           uint8_t *buf) {
  size_t amt_to_write = size * (sizeof(T) / sizeof(uint8_t));
  memcpy(buf, data.get(), amt_to_write);
  return amt_to_write;
}

std::string get_readable_input_type(DataType type) {
  switch (type) {
    case DataType::Bytes:
      return std::string("bytes");
    // case DataType::Ints: return std::string("integers");
    case DataType::Floats: return std::string("floats");
    case DataType::Doubles:
      return std::string("doubles");
    // case DataType::Strings: return std::string("strings");
    case DataType::Invalid:
    default: return std::string("Invalid input type");
  }
}

DataType parse_input_type(std::string type_string) {
  if (type_string == "bytes" || type_string == "byte" || type_string == "b") {
    return DataType::Bytes;
    // } else if (type_string == "integers" || type_string == "ints" ||
    //            type_string == "integer" || type_string == "int" ||
    //            type_string == "i") {
    //   return DataType::Ints;
  } else if (type_string == "floats" || type_string == "float" ||
             type_string == "f") {
    return DataType::Floats;
  } else if (type_string == "doubles" || type_string == "double" ||
             type_string == "d") {
    return DataType::Doubles;
    // } else if (type_string == "strings" || type_string == "string" ||
    //            type_string == "str" || type_string == "strs" ||
    //            type_string == "s") {
    //   return DataType::Strings;
  } else {
    throw std::invalid_argument(type_string + " is not a valid input string");
  }
}

VersionedModelId::VersionedModelId(const std::string name, const std::string id)
    : name_(name), id_(id) {}

std::string VersionedModelId::get_name() const { return name_; }

std::string VersionedModelId::get_id() const { return id_; }

std::string VersionedModelId::serialize() const {
  std::stringstream ss;
  ss << name_;
  ss << ITEM_PART_CONCATENATOR;
  ss << id_;
  return ss.str();
}

VersionedModelId VersionedModelId::deserialize(std::string str) {
  auto split = str.find(ITEM_PART_CONCATENATOR);
  std::string model_name = str.substr(0, split);
  std::string model_version = str.substr(split + 1, str.size());
  return VersionedModelId(model_name, model_version);
}

bool VersionedModelId::operator==(const VersionedModelId &rhs) const {
  return (name_ == rhs.name_ && id_ == rhs.id_);
}

bool VersionedModelId::operator!=(const VersionedModelId &rhs) const {
  return !(name_ == rhs.name_ && id_ == rhs.id_);
}

Output::Output(const std::shared_ptr<OutputData> y_hat,
               const std::vector<VersionedModelId> models_used)
    : y_hat_(std::move(y_hat)), models_used_(models_used) {}

bool Output::operator==(const Output &rhs) const {
  return (y_hat_->hash() == rhs.y_hat_->hash() &&
          models_used_ == rhs.models_used_);
}

bool Output::operator!=(const Output &rhs) const {
  return !(y_hat_->hash() == rhs.y_hat_->hash() &&
           models_used_ == rhs.models_used_);
}

InputVector::InputVector(void *data, size_t size_typed, size_t size_bytes,
                         DataType type)
    : data_(data),
      size_typed_(size_typed),
      size_bytes_(size_bytes),
      type_(type) {}

std::shared_ptr<OutputData> OutputData::create_output(
    DataType type, std::shared_ptr<void> data, size_t start, size_t end) {
  switch (type) {
    case DataType::Bytes:
      return std::make_shared<ByteVectorOutput>(
          std::static_pointer_cast<uint8_t>(data), start, end);
    case DataType::Ints:
      return std::make_shared<IntVectorOutput>(
          std::static_pointer_cast<int>(data), start / sizeof(int),
          end / sizeof(int));
    case DataType::Floats:
      return std::make_shared<FloatVectorOutput>(
          std::static_pointer_cast<float>(data), start / sizeof(float),
          end / sizeof(float));
    case DataType::Strings:
      return std::make_shared<StringOutput>(
          std::static_pointer_cast<char>(data), start / sizeof(char),
          end / sizeof(char));
    case DataType::Doubles:
    case DataType::Invalid:
    default:
      std::stringstream ss;
      ss << "Attempted to create an output of an unsupported data type: "
         << get_readable_input_type(type);
      throw std::runtime_error(ss.str());
  }
}

ByteVectorOutput::ByteVectorOutput(std::shared_ptr<uint8_t> data, size_t start,
                                   size_t end)
    : data_(data), start_(start), end_(end) {}

size_t ByteVectorOutput::size() const { return end_ - start_; }

size_t ByteVectorOutput::byte_size() const { return end_ - start_; }

size_t ByteVectorOutput::hash() const {
  return boost::hash_range(data_.get() + start_, data_.get() + end_);
}

DataType ByteVectorOutput::type() const { return DataType::Bytes; }

size_t ByteVectorOutput::serialize(void *buf) const {
  memcpy(buf, data_.get() + start_, end_ - start_);
  return end_ - start_;
}

void *ByteVectorOutput::get_data() const { return data_.get() + start_; }

IntVectorOutput::IntVectorOutput(std::shared_ptr<int> data, size_t start,
                                 size_t end)
    : data_(data), start_(start), end_(end) {}

size_t IntVectorOutput::size() const { return end_ - start_; }

size_t IntVectorOutput::byte_size() const {
  return (end_ - start_) * sizeof(int);
}

size_t IntVectorOutput::hash() const {
  return boost::hash_range(data_.get() + start_, data_.get() + end_);
}

DataType IntVectorOutput::type() const { return DataType::Ints; }

size_t IntVectorOutput::serialize(void *buf) const {
  memcpy(buf, data_.get() + start_, (end_ - start_) * sizeof(int));
  return end_ - start_;
}

void *IntVectorOutput::get_data() const { return data_.get() + start_; }

FloatVectorOutput::FloatVectorOutput(std::shared_ptr<float> data, size_t start,
                                     size_t end)
    : data_(data), start_(start), end_(end) {}

size_t FloatVectorOutput::size() const { return end_ - start_; }

size_t FloatVectorOutput::byte_size() const {
  return (end_ - start_) * sizeof(float);
}

size_t FloatVectorOutput::hash() const {
  return boost::hash_range(data_.get() + start_, data_.get() + end_);
}

DataType FloatVectorOutput::type() const { return DataType::Floats; }

size_t FloatVectorOutput::serialize(void *buf) const {
  memcpy(buf, data_.get() + start_, (end_ - start_) * sizeof(float));
  return end_ - start_;
}

void *FloatVectorOutput::get_data() const { return data_.get() + start_; }

StringOutput::StringOutput(std::shared_ptr<char> data, size_t start, size_t end)
    : data_(data), start_(start), end_(end) {}

size_t StringOutput::size() const { return end_ - start_; }

size_t StringOutput::byte_size() const {
  return (end_ - start_) * sizeof(char);
}

size_t StringOutput::hash() const {
  return boost::hash_range(data_.get() + start_, data_.get() + end_);
}

DataType StringOutput::type() const { return DataType::Strings; }

size_t StringOutput::serialize(void *buf) const {
  memcpy(buf, data_.get() + start_, (end_ - start_) * sizeof(char));
  return end_ - start_;
}

void *StringOutput::get_data() const { return data_.get() + start_; }

// rpc::PredictionRequest::PredictionRequest(DataType input_type)
//     : input_type_(input_type) {}
//
// rpc::PredictionRequest::PredictionRequest(
//     std::vector<std::shared_ptr<Input>> inputs, DataType input_type)
//     : inputs_(inputs), input_type_(input_type) {
//   for (int i = 0; i < (int)inputs.size(); i++) {
//     validate_input_type(inputs[i]);
//     input_data_size_ += inputs[i]->byte_size();
//   }
// }
//
// void rpc::PredictionRequest::validate_input_type(
//     std::shared_ptr<Input> &input) const {
//   if (input->type() != input_type_) {
//     std::ostringstream ss;
//     ss << "Attempted to add an input of type "
//        << get_readable_input_type(input->type())
//        << " to a prediction request with input type "
//        << get_readable_input_type(input_type_);
//     log_error(LOGGING_TAG_CLIPPER, ss.str());
//     throw std::invalid_argument(ss.str());
//   }
// }
//
// void rpc::PredictionRequest::add_input(std::shared_ptr<Input> input) {
//   validate_input_type(input);
//   inputs_.push_back(input);
//   input_data_size_ += input->byte_size();
// }

// std::vector<ByteBuffer> rpc::PredictionRequest::serialize() {
//   if (input_data_size_ <= 0) {
//     throw std::length_error(
//         "Attempted to serialize a request with no input data!");
//   }
//
//   size_t request_metadata_size = 1 * sizeof(uint32_t);
//   uint32_t *request_metadata =
//       static_cast<uint32_t *>(malloc(request_metadata_size));
//   request_metadata[0] = static_cast<uint32_t>(RequestType::PredictRequest);
//
//   // std::shared_ptr<uint8_t> request_metadata(
//   //     static_cast<uint8_t *>(malloc(request_metadata_size)), free);
//   // uint32_t *request_metadata_raw =
//   //     reinterpret_cast<uint32_t *>(request_metadata.get());
//   // request_metadata_raw[0] =
//   // static_cast<uint32_t>(RequestType::PredictRequest);
//
//   size_t input_metadata_size = (2 + (inputs_.size() - 1)) * sizeof(uint32_t);
//   uint32_t *input_metadata =
//       static_cast<uint32_t *>(malloc(input_metadata_size));
//   input_metadata[0] = static_cast<uint32_t>(input_type_);
//   input_metadata[1] = static_cast<uint32_t>(inputs_.size());
//
//   for (size_t i = 0; i < inputs_.size(); ++i) {
//     input_metadata[i + 2] = static_cast<uint32_t>(inputs_[i]->byte_size());
//   }
//
//   // std::shared_ptr<uint8_t> input_metadata(
//   //     static_cast<uint8_t *>(malloc(input_metadata_size)), free);
//   // uint32_t *input_metadata_raw =
//   //     reinterpret_cast<uint32_t *>(input_metadata.get());
//   // input_metadata_raw[0] = static_cast<uint32_t>(input_type_);
//   // input_metadata_raw[1] = static_cast<uint32_t>(inputs_.size());
//   //
//   // std::vector<std::shared_ptr<void>> input_bufs;
//   // for (size_t i = 0; i < inputs_.size(); i++) {
//   //   inputs_[i]->serialize(input_bufs);
//   //   input_metadata_raw[i + 2] =
//   //   static_cast<uint32_t>(inputs_[i]->byte_size());
//   // }
//
//   size_t input_metadata_size_buf_size = 1 * sizeof(long);
//   long *input_metadata_size_buf =
//       static_cast<long *>(malloc(input_metadata_size_buf_size));
//   // Add the size of the input metadata in bytes. This will be
//   // sent prior to the input metadata to allow for proactive
//   // buffer allocation in the receiving container
//   input_metadata_size_buf[0] = input_metadata_size;
//
//   // size_t input_metadata_size_buf_size = 1 * sizeof(long);
//   // std::shared_ptr<uint8_t> input_metadata_size_buf(
//   //     static_cast<uint8_t *>(malloc(input_metadata_size_buf_size)), free);
//   // long *input_metadata_size_buf_raw =
//   //     reinterpret_cast<long *>(input_metadata_size_buf.get());
//   // // Add the size of the input metadata in bytes. This will be
//   // // sent prior to the input metadata to allow for proactive
//   // // buffer allocation in the receiving container
//   // input_metadata_size_buf_raw[0] = input_metadata_size;
//
//   std::vector<ByteBuffer> serialized_request;
//   serialized_request.push_back(std::make_pair(
//       reinterpret_cast<void *>(request_metadata), request_metadata_size));
//   serialized_request.push_back(
//       std::make_pair(reinterpret_cast<void *>(input_metadata_size_buf),
//                      input_metadata_size_buf_size));
//   serialized_request.push_back(std::make_pair(
//       reinterpret_cast<void *>(input_metadata), input_metadata_size));
//
//   for (size_t i = 0; i < inputs_.size(); ++i) {
//     serialized_request.push_back(
//         std::make_pair(inputs_[i]->get_data(), inputs_[i]->byte_size()));
//   }
//   return serialized_request;
// }

rpc::PredictionResponse::PredictionResponse(
    const std::vector<std::shared_ptr<OutputData>> outputs)
    : outputs_(outputs) {}

rpc::PredictionResponse
rpc::PredictionResponse::deserialize_prediction_response(
    DataType data_type, std::shared_ptr<void> &data) {
  std::vector<std::shared_ptr<OutputData>> outputs;
  uint32_t *output_lengths_data = reinterpret_cast<uint32_t *>(data.get());
  uint32_t num_outputs = output_lengths_data[0];
  output_lengths_data++;
  size_t curr_output_index =
      sizeof(uint32_t) + (num_outputs * sizeof(uint32_t));
  for (uint32_t i = 0; i < num_outputs; i++) {
    uint32_t output_length = output_lengths_data[i];
    std::shared_ptr<OutputData> output = OutputData::create_output(
        data_type, data, curr_output_index, curr_output_index + output_length);
    outputs.push_back(std::move(output));
    curr_output_index += output_length;
  }
  return PredictionResponse(outputs);
}

// Query::Query(std::string label, long user_id, std::shared_ptr<Input> input,
//              long latency_budget_micros, std::string selection_policy,
//              std::vector<VersionedModelId> candidate_models)
//     : label_(label),
//       user_id_(user_id),
//       input_(input),
//       latency_budget_micros_(latency_budget_micros),
//       selection_policy_(selection_policy),
//       candidate_models_(candidate_models),
//       create_time_(std::chrono::high_resolution_clock::now()) {}
//
// Response::Response(Query query, QueryId query_id, const long duration_micros,
//                    Output output, const bool output_is_default,
//                    const boost::optional<std::string> default_explanation)
//     : query_(std::move(query)),
//       query_id_(query_id),
//       duration_micros_(duration_micros),
//       output_(std::move(output)),
//       output_is_default_(output_is_default),
//       default_explanation_(std::move(default_explanation)) {}

PredictTask::PredictTask(InputVector input, VersionedModelId model,
                         float utility, QueryId query_id,
                         long latency_slo_micros)
    : input_(input),
      model_(model),
      utility_(utility),
      query_id_(query_id),
      latency_slo_micros_(latency_slo_micros) {}

}  // namespace clipper
