#!/usr/bin/python2
import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("portnumber", help="The number of the port you want free (Eg. 8000)",
                    type=int, default=None)
parser.add_argument("--yes", default=False, help="confirm free port", action="store_true")
args = parser.parse_args()
port = args.portnumber
errMsg = None

try:
    errMsg = 'Enter integer value for port number'
    port = int(port)
    cmd = 'lsof -t -i:{0}'.format(port)
    pid = subprocess.check_output(cmd, shell=True)
    pid = int(pid)
except ValueError:
    pid = None
    print(errMsg)
    exit()
except Exception as e:
    # print('exception: ', e)
    print("No process running on port {0}".format(port))
    exit()
processTypeCmd = 'ps -p {0} -o comm='.format(pid)
processType = subprocess.check_output(processTypeCmd, shell=True).rstrip('\n')
confirm = ''
if processType and not args.yes:
    while True:
        confirm = raw_input("Process Type: '{0}'  Port: {1}. Kill?[yes/no]".format(processType, port))
        confirm = confirm.lower()
        if confirm == 'yes' or confirm == 'no':
            break

if confirm == 'yes' or args.yes:

    killcmd = 'kill -9 {0}'.format(pid) if pid else None
    isKilled = os.system('kill -9 {0}'.format(pid)) if pid else None
    if isKilled == 0:
        print("Port {0} is free. Processs {1} killed successfully".format(port, pid))
    else:
        print("Cannot free port {0}.Failed to kill process {1}, err code:{2}".format(port, pid, isKilled))
