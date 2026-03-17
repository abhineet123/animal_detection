import json, sys, time

BUFFER_SIZE = 4096


def bindToPort(sock, port, name):
    server_address = ('', port)
    print_msg = True
    while True:
        try:
            sock.bind(server_address)
            break
        except:
            time.sleep(1)
            if print_msg:
                sys.stdout.write('Waiting to bind to the {:s} port'.format(name))
                sys.stdout.flush()
                print_msg = False
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
            continue
            # cmd = 'lsof -t -i tcp:{:d} | xargs kill -9'.format(self.params.port)
            # pid = subprocess.check_output(cmd, shell=True)
            # print('Process {} using port'.format(pid))
    if not print_msg:
        sys.stdout.write('Done\n')
        sys.stdout.flush()


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

