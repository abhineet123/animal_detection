import datetime, sys

while True:
    try:
        sys.stdout.write('\r{}'.format(datetime.datetime.time(datetime.datetime.now())))
    except KeyboardInterrupt:
        break
