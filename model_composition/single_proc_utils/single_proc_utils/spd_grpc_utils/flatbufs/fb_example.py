import numpy as np
import FloatsInput
import flatbuffers


def test():
    img = np.random.rand(1024).astype(np.float32)
    img_bytes = img.tobytes()
    builder = flatbuffers.Builder(len(img_bytes))
    FloatsInput.FloatsInputStartDataVector(builder, len(img_bytes))
    builder.Bytes[builder.head : (builder.head + len(img_bytes))] = img_bytes
    data = builder.EndVector(len(img_bytes))
    FloatsInput.FloatsInputStart(builder)
    FloatsInput.FloatsInputAddData(builder, data)
    out_idx = FloatsInput.FloatsInputEnd(builder)
    builder.Finish(out_idx)
    out = builder.Output()

    inp1 = FloatsInput.FloatsInput.GetRootAsFloatsInput(out, 0)
    reconstructed_img = inp1.DataAsNumpy()
    print(reconstructed_img.dtype)
    print(len(reconstructed_img))
    print(reconstructed_img[0], img[0])
    print(img)

if __name__ == "__main__":
    test()
