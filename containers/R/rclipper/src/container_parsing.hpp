#ifndef CLIPPER_CONTAINER_PARSING_HPP
#define CLIPPER_CONTAINER_PARSING_HPP

#include <cstring>
#include <vector>

#include "datatypes.hpp"

namespace container {

template <typename T>
struct input_parser_type {
  static const bool is_string_parser = false;
};

template <>
struct input_parser_type<char> {
  static const bool is_string_parser = true;
};

template <typename D>
class InputParser {
 public:
  InputParser(std::vector<D>& buffer) : buffer_(buffer) {}
  /**
   * Exposes a buffer into which request data should be
   * read directly from a socket. The buffer contents
   * can then be parsed via `get_inputs`
   */
  std::vector<D>& get_data_buffer(long min_size_bytes) {
    resize_if_necessary(buffer_, min_size_bytes);
    return buffer_;
  }

  const std::vector<Input<D>> get_inputs(const std::vector<int>& input_header,
                                         long input_content_size) {
    if (input_parser_type<D>::is_string_parser) {
      return get_string_inputs(input_header, input_content_size);
    } else {
      return get_primitive_inputs(input_header, input_content_size);
    }
  }

 private:
  std::vector<D>& buffer_;

  void resize_if_necessary(std::vector<D>& buffer, long required_buffer_size) {
    if (static_cast<long>((buffer.size() * sizeof(D))) < required_buffer_size) {
      buffer.reserve((2 * required_buffer_size) / sizeof(D));
    }
  }

  const std::vector<Input<D>> get_primitive_inputs(
      const std::vector<int>& input_header, long input_content_size) {
    // For a prediction request containing `n` inputs, there are `n - 1` split
    // indices
    int num_splits = input_header[1] - 1;
    std::vector<Input<D>> inputs;
    int prev_split = 0;
    // Iterate from the beginning of the input data content
    // (the first two elements of the header are metadata)
    for (int i = 2; i < num_splits + 2; i++) {
      int curr_split = input_header[i];
      Input<D> input = construct_input(buffer_, prev_split, curr_split);
      inputs.push_back(input);
      prev_split = curr_split;
    }
    int typed_input_content_size =
        static_cast<int>(input_content_size / static_cast<long>(sizeof(D)));
    Input<D> tail_input =
        construct_input(buffer_, prev_split, typed_input_content_size);
    inputs.push_back(tail_input);
    return inputs;
  }

  const Input<D> construct_input(std::vector<D>& data_buffer, size_t data_start,
                                 size_t data_end) {
    D* input_data = data_buffer.data() + data_start;
    size_t length = data_end - data_start;
    Input<D> input(input_data, length);
    return input;
  }

  const std::vector<Input<D>> get_string_inputs(
      const std::vector<int>& /*input_header*/, long input_content_size) {
    std::vector<Input<D>> inputs;
    char* concat_content = reinterpret_cast<char*>(buffer_.data());
    long curr_pos = 0;
    while (curr_pos < input_content_size) {
      char* input_content = concat_content + curr_pos;
      size_t input_length = std::strlen(input_content);
      Input<D> input(reinterpret_cast<D*>(input_content), input_length);
      inputs.push_back(input);
      // advance past the null terminator to the beginning of the next input
      curr_pos += input_length + 1;
    }
    return inputs;
  }
};

}  // namespace container

#endif  // CLIPPER_CONTAINER_PARSING_HPP
