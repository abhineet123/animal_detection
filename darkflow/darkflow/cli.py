from .defaults import argHandler #Import the default arguments
import os, sys
from .net.build import TFNet

sys.path.append('..')
from tf_api.utilities import sortKey

def cliHandler(args):
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    file_list = [os.path.join(FLAGS.dataset, name) for name in os.listdir(FLAGS.dataset) if
                 os.path.isdir(os.path.join(FLAGS.dataset, name))]
    file_list.sort(key=sortKey)

    FLAGS.dataset = file_list
    FLAGS.annotation = [os.path.join(f, 'annotations') for f in file_list]

    print('FLAGS.dataset:\n', FLAGS.dataset)
    print('FLAGS.annotation:\n', FLAGS.annotation)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    
    requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir,'out')]
    if FLAGS.summary:
        requiredDirectories.append(FLAGS.summary)

    _get_dir(requiredDirectories)

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

    tfnet = TFNet(FLAGS)
    
    if FLAGS.demo:
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train:
        print('Enter training ...'); tfnet.train()
        if not FLAGS.savepb: 
            exit('Training finished, exit.')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb(); exit('Done')

    # print('FLAGS.dataset: {}'.format(FLAGS.dataset))
    # file_list = [os.path.join(FLAGS.dataset, name) for name in os.listdir(FLAGS.dataset) if
    #              os.path.isdir(os.path.join(FLAGS.dataset, name))]
    # file_list.sort(key=sortKey)

    FLAGS.imgdir = FLAGS.dataset

    tfnet.predict(FLAGS.dataset)
