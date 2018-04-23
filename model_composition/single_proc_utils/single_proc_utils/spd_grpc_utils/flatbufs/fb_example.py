import numpy as np
import FloatsInput
import PredictRequest
import flatbuffers

INCEPTION_IMAGE_SIZE = 299 * 299 * 3 * 4

def test():
    img = np.random.rand(1024).astype(np.float32)
    img_bytes = memoryview(img.view(np.uint8))
    builder = flatbuffers.Builder(len(img_bytes))
    FloatsInput.FloatsInputStartDataVector(builder, len(img_bytes))
    builder.Bytes[builder.head : (builder.head + len(img_bytes))] = img_bytes
    data = builder.EndVector(len(img_bytes))
    FloatsInput.FloatsInputStart(builder)
    FloatsInput.FloatsInputAddData(builder, data)
    out_idx = FloatsInput.FloatsInputEnd(builder)
    print("IDX TYPE", type(out_idx))
    builder.Finish(out_idx)
    out = builder.Output()

    inp1 = FloatsInput.FloatsInput.GetRootAsFloatsInput(out, 0)
    reconstructed_img = inp1.DataAsNumpy()
    print(reconstructed_img.dtype)
    print("LENGTHS", len(reconstructed_img), len(img))
    print(reconstructed_img[0], img[0])
    print(img)


def _create_predict_request(inputs, msg_ids):
    batch_size = len(inputs)
    builder_size = (batch_size + 5) * INCEPTION_IMAGE_SIZE
    builder = flatbuffers.Builder(builder_size)
    floats_input_idxs = []
    for inp in inputs:
        inp_bytes = memoryview(inp.view(np.uint8))
        inp_bytes_len = len(inp_bytes)
        FloatsInput.FloatsInputStartDataVector(builder, inp_bytes_len) 
        builder.Bytes[builder.head : (builder.head + inp_bytes_len)] = inp_bytes
        data = builder.EndVector(inp_bytes_len)
        FloatsInput.FloatsInputStart(builder)
        FloatsInput.FloatsInputAddData(builder, data)
        floats_input_idx = FloatsInput.FloatsInputEnd(builder)
        floats_input_idxs.append(floats_input_idx)

    msg_ids_bytes = memoryview(msg_ids.view(np.uint8))
    msg_ids_len = len(msg_ids_bytes)
    PredictRequest.PredictRequestStartMsgIdsVector(builder, msg_ids_len)
    builder.Bytes[builder.head : (builder.head + msg_ids_len)] = msg_ids_bytes 
    msg_ids_idx = builder.EndVector(msg_ids_len)

    PredictRequest.PredictRequestStartInputsVector(builder, len(floats_input_idxs))
    for floats_input_idx in floats_input_idxs:
        curr_offset = builder.PrependUOffsetTRelative(floats_input_idx)
    inputs_vector_idx = builder.EndVector(len(floats_input_idxs))
    

    PredictRequest.PredictRequestStart(builder)
    PredictRequest.PredictRequestAddInputs(builder, inputs_vector_idx)
    PredictRequest.PredictRequestAddMsgIds(builder, msg_ids_idx)
    request_idx = PredictRequest.PredictRequestEnd(builder)
    builder.Finish(request_idx)
    request = builder.Output()

    parsed = PredictRequest.PredictRequest.GetRootAsPredictRequest(request, 0)
    print(parsed.InputsLength())
    print(parsed.Inputs(5).DataAsNumpy())


    return request

if __name__ == "__main__":
    inps = np.random.rand(60, 299, 299, 3)
    msg_ids = np.array(range(60))
    _create_predict_request(inps, msg_ids)

    # test()
