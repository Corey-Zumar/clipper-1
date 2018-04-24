INCEPTION_IMAGE_SIZE_BYTES = 299 * 299 * 3 * 4

GRPC_OPTIONS=[('grpc.max_message_length', 200 * INCEPTION_IMAGE_SIZE_BYTES),
              ('grpc.max_send_message_length', 200 * INCEPTION_IMAGE_SIZE_BYTES),
              ('grpc.max_receive_message_length', 200 * INCEPTION_IMAGE_SIZE_BYTES)]
