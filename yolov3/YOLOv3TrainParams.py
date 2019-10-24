# Hyperparameters: train.py --evolve --epochs 2 --img-size 320, Metrics: 0.204      0.302      0.175      0.234 (square smart)
# hyp = {'xy': 0.2,  # xy loss gain
#        'wh': 0.1,  # wh loss gain
#        'cls': 0.04,  # cls loss gain
#        'conf': 4.5,  # conf loss gain
#        'iou_t': 0.5,  # iou target-anchor training threshold
#        'lr0': 0.001,  # initial learning rate
#        'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.90,  # SGD momentum
#        'weight_decay': 0.0005,  # optimizer weight decay
#        }

# paramparse.fromDict(hyp, "hyp")

# Hyperparameters: Original, Metrics: 0.172      0.304      0.156      0.205 (square)
# hyp = {'xy': 0.5,  # xy loss gain
#        'wh': 0.0625,  # wh loss gain
#        'cls': 0.0625,  # cls loss gain
#        'conf': 4,  # conf loss gain
#        'iou_t': 0.1,  # iou target-anchor training threshold
#        'lr0': 0.001,  # initial learning rate
#        'lrf': -5.,  # final learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.9,  # SGD momentum
#        'weight_decay': 0.0005}  # optimizer weight decay

# Hyperparameters: train.py --evolve --epochs 2 --img-size 320, Metrics: 0.225      0.251      0.145      0.218 (rect)
# hyp = {'xy': 0.4499,  # xy loss gain
#        'wh': 0.05121,  # wh loss gain
#        'cls': 0.04207,  # cls loss gain
#        'conf': 2.853,  # conf loss gain
#        'iou_t': 0.2487,  # iou target-anchor training threshold
#        'lr0': 0.0005301,  # initial learning rate
#        'lrf': -5.,  # final learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.8823,  # SGD momentum
#        'weight_decay': 0.0004149}  # optimizer weight decay

# Hyperparameters: train.py --evolve --epochs 2 --img-size 320, Metrics: 0.178      0.313      0.167      0.212 (square)
# hyp = {'xy': 0.4664,  # xy loss gain
#        'wh': 0.08437,  # wh loss gain
#        'cls': 0.05145,  # cls loss gain
#        'conf': 4.244,  # conf loss gain
#        'iou_t': 0.09121,  # iou target-anchor training threshold
#        'lr0': 0.0004938,  # initial learning rate
#        'lrf': -5.,  # final learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.9025,  # SGD momentum
#        'weight_decay': 0.0005417}  # optimizer weight decay


class HyperParams:
    """
    :param float cls:
    :param float conf:
    :param float iou_t:
    :param float lr0:
    :param float lrf:
    :param float momentum:
    :param float weight_decay:
    :param float wh:
    :param float xy:
    """

    def __init__(self):
        self.cfg = ''
        self.cls = 0.04
        self.conf = 4.5
        self.iou_t = 0.5
        self.lr0 = 0.001
        self.lrf = -4.0
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.wh = 0.1
        self.xy = 0.2
        self.help = {
            'cls': '',
            'conf': '',
            'iou_t': '',
            'lr0': '',
            'lrf': '',
            'momentum': '',
            'weight_decay': '',
            'wh': '',
            'xy': '',
        }


class YOLOv3TrainParams:
    """
    :param int accumulate: accumulate gradient x batches before optimizing
    :param str backend: distributed backend
    :param int batch_size: size of each image batch
    :param str net_cfg: cfg file path
    :param str data_cfg: coco.data file path
    :param str dist_url: distributed training init method
    :param int epochs: number of epochs
    :param bool evolve: run hyperparameter evolution
    :param int img_size: inference size (pixels)
    :param int mixed_precision: mixed_precision training
    :param bool multi_scale: random image sizes per batch 320 - 608
    :param bool nosave: do not save training results
    :param bool notest: only test final epoch
    :param int num_workers: number of Pytorch DataLoader workers
    :param str pretrained_weights: pretrained_weights path
    :param int rank: distributed training node rank
    :param bool resume: resume training flag
    :param bool transfer: transfer learning flag
    :param int var: debug variable
    :param str weights: weights path
    :param int world_size: number of nodes for distributed training
    :param HyperParams hyp: HyperParams
    """

    def __init__(self):
        self.accumulate = 1
        self.backend = 'nccl'
        self.batch_size = 16
        self.net_cfg = 'cfg/yolov3-spp.cfg'
        self.data_cfg = 'data/coco.data'
        self.dist_url = 'tcp://127.0.0.1:9999'
        self.epochs = 500
        self.evolve = False
        self.img_size = 416
        self.mixed_precision = 0
        self.multi_scale = False
        self.nosave = False
        self.notest = False
        self.num_workers = 4
        self.pretrained_weights = 'pretrained_weights'
        self.rank = 0
        self.resume = False
        self.transfer = False
        self.var = 0
        self.weights = 'weights'
        self.hyp = HyperParams()
        self.world_size = 1
        self.load_sep = ' '
        self.help = {
            'accumulate': 'accumulate gradient x batches before optimizing',
            'backend': 'distributed backend',
            'batch_size': 'size of each image batch',
            'cfg': 'cfg file path',
            'data_cfg': 'coco.data file path',
            'dist_url': 'distributed training init method',
            'epochs': 'number of epochs',
            'evolve': 'run hyperparameter evolution',
            'img_size': 'inference size (pixels)',
            'mixed_precision': 'mixed_precision training',
            'multi_scale': 'random image sizes per batch 320 - 608',
            'nosave': 'do not save training results',
            'notest': 'only test final epoch',
            'num_workers': 'number of Pytorch DataLoader workers',
            'pretrained_weights': 'pretrained_weights path',
            'rank': 'distributed training node rank',
            'resume': 'resume training flag',
            'transfer': 'transfer learning flag',
            'var': 'debug variable',
            'weights': 'weights path',
            'world_size': 'number of nodes for distributed training',
            'hyp': 'HyperParams',
        }
