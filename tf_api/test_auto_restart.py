import sys
import psutil
from subprocess import PIPE, Popen
from threading  import Thread
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x

cmd = 'python3 test.py'

for _arg in sys.argv[1:]:
    cmd = '{} {}'.format(cmd, _arg)

print('Running: \n{}'.format(cmd))
# process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

ON_POSIX = 'posix' in sys.builtin_module_names

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

process = Popen(cmd, stdout=PIPE, stderr=PIPE, bufsize=1, close_fds=ON_POSIX, shell=True)
q1 = Queue()
t1 = Thread(target=enqueue_output, args=(process.stdout, q1))
t1.daemon = True # thread dies with the program
t1.start()

q2 = Queue()
t2 = Thread(target=enqueue_output, args=(process.stderr, q2))
t2.daemon = True # thread dies with the program
t2.start()

while True:
    # output, error = process.communicate()
    # output = output[0].decode("utf-8")
    # error = error[0].decode("utf-8")
    # sys.stdout.write(output)
    # sys.stdout.write(error)
    # sys.stdout.flush()

    # read line without blocking
    try:
        line1 = q1.get_nowait()  # or q.get(timeout=.1)
    except Empty:
        try:
            line2 = q2.get_nowait()  # or q.get(timeout=.1)
        except Empty:
            # print('no output yet')
            continue
        else:
            line2 = line2.decode("utf-8")
            sys.stdout.write(line2)
            sys.stdout.flush()
            if 'Segmentation fault' not in line2 and process.pid in psutil.pids():
                continue
    else:
        line1 = line1.decode("utf-8")
        sys.stdout.write(line1)
        sys.stdout.flush()
        if 'Segmentation fault' not in line1 and process.pid in psutil.pids():
            continue

    print('process.pid: ', process.pid)
    print('psutil.pids(): ', psutil.pids())

    print('\nRestarting...\n')

    process = Popen(cmd, stdout=PIPE, stderr=PIPE, bufsize=1, close_fds=ON_POSIX, shell=True)
    q1 = Queue()
    t1 = Thread(target=enqueue_output, args=(process.stdout, q1))
    t1.daemon = True  # thread dies with the program
    t1.start()

    q2 = Queue()
    t2 = Thread(target=enqueue_output, args=(process.stderr, q2))
    t2.daemon = True  # thread dies with the program
    t2.start()


