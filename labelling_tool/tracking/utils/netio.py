import json

BUFFER_SIZE = 4096

def recv_from_connection(connection):
    buff = ''
    while True:
        data = connection.recv(BUFFER_SIZE).decode()
        buff += data
        if '%%END%%' in buff:
            buff = buff.split('%%END%%')[0]
            break
    return json.loads(buff)


def send_msg_to_connection(msg, connection):
    connection.send((json.dumps(msg) + " %%END%%").encode())

class ServerLog(object):
    def __init__(self):
        self.logs = []
        self.new_logs = []

    def add_log(self, text):
        self.logs.append(text)
        self.new_logs.append(text)

    def get_logs(self):
        return "\n".join(self.logs)

    def get_new_logs(self):
        out = "\n".join(self.new_logs)
        self.new_logs = []
        return out

