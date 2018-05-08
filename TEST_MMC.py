import numpy as np
from containerized_utils.zmq_client import Client

if __name__ == "__main__":
    client = Client("localhost", 4456, 4455)
    client.start()

    def continuation(output):
        print(output)

    for i in range(10):
        app_name = "app1"
        # if i % 2 == 0:
        #     app_name = "app1"
        # else:
        #     app_name = "app2"
        x = np.random.rand(1024)
        # print(x)
        client.send_request(app_name, x).then(continuation)
