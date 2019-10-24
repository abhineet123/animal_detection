from labelling_tool.tracking.DaSiamRPN.DaSiamRPN import DaSiamRPNParams
from labelling_tool.tracking.siamfc.SiamFC import SiamFCParams
from labelling_tool.tracking.SiamMask.SiamMask import SiamMaskParams


class YOLOv3TestParams:
    """
    :param float assoc_thresh: iou threshold for associating tracked and detected boxes
    :param int batch_size: size of each image batch
    :param str net_cfg: cfg file path
    :param float conf_thresh: object confidence threshold
    :param str data_cfg: coco.data file path
    :param int img_size: inference size (pixels)
    :param float nms_thresh: iou threshold for non-maximum suppression
    :param str nms_type: NMS method: 'OR', 'AND', 'MERGE', 'SOFT')
    :param str out_suffix: out_suffix
    :param str save_dir: save_dir path
    :param str test_path: test_path
    :param int track_diff: minimum frame difference between initializing two trackers
    :param float track_thresh: tracking confidence threshold
    :param str tracker_type: enable Siamese tracking to reduce false negatives
    :param int unassoc_thresh: unassoc_thresh before tracker is removed
    :param int vis: enable visualization
    :param str weights: path to weights file
    """

    def __init__(self):
        self.assoc_thresh = 0.5
        self.batch_size = 16
        self.net_cfg = 'cfg/yolov3-spp.cfg'
        self.conf_thresh = 0.001
        self.data_cfg = 'data/coco.data'
        self.img_size = 416
        self.nms_thresh = 0.5
        self.nms_type = "merge"
        self.out_suffix = ''
        self.save_dir = ''
        self.test_path = ''
        self.track_diff = 1
        self.track_thresh = 0.001
        self.tracker_type = ''
        self.unassoc_thresh = 2
        self.filter_unassociated = 1
        self.max_trackers = 0
        self.vis = 0
        self.verbose = 0
        self.weights = 'weights/yolov3-spp.weights'

        self.siam_fc = SiamFCParams()
        self.siam_mask = SiamMaskParams()
        self.da_siam_rpn = DaSiamRPNParams()

        self.help = {
            'assoc_thresh': 'iou threshold for associating tracked and detected boxes',
            'batch_size': 'size of each image batch',
            'cfg': 'cfg file path',
            'conf_thresh': 'object confidence threshold',
            'data_cfg': 'coco.data file path',
            'img_size': 'inference size (pixels)',
            'nms_thresh': 'iou threshold for non-maximum suppression',
            'nms_type': 'NMS method: OR, AND, MERGE, SOFT)',
            'out_suffix': 'out_suffix',
            'save_dir': 'save_dir path',
            'test_path': 'test_path',
            'track_diff': 'minimum frame difference between creating two trackers',
            'track_thresh': 'tracking confidence threshold',
            'tracker_type': {
                'SiamFC': (0, 'fc', 'siam_fc', 'SiamFC'),
                'DaSiamRPN': (1, 'rpn', 'da_rpn', 'da_siam_rpn', 'DaSiamRPN'),
                'SiamMask': (2, 'mask', 'siam_mask', 'SiamMask'),
            },
            'max_trackers': 'maximum no. of allowed trackers; '
                            'least confident tracker is removed on exceeding this; '
                            '0 means that there is no bound',
            'filter_unassociated': 'filter unassociated detections based on tracking scores',
            'unassoc_thresh': 'unassoc_thresh before tracker is removed',
            'vis': 'enable visualization',
            'weights': 'path to weights file',
        }

# parser = argparse.ArgumentParser(prog='test.py')
    # parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    # parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    # parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    # parser.add_argument('--test_path', type=str, default='', help='test_path')
    # parser.add_argument('--save_dir', type=str, default='', help='save_dir path')
    # parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    # parser.add_argument('--conf_thresh', type=float, default=0.001, help='object confidence threshold')
    # parser.add_argument('--nms_thresh', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    # parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    # parser.add_argument('--vis', type=int, default=0, help='enable visualization')
    # parser.add_argument('--tracker_type', type=int, default=0, help='enable Siamese tracking to reduce false negatives')
    # parser.add_argument('--unassoc_thresh', type=int, default=2,
    #                     help='unassoc_thresh before tracker is removed')
    # parser.add_argument('--assoc_thresh', type=float, default=0.5,
    #                     help='iou threshold for associating tracked and detected boxes')
    # parser.add_argument('--track_thresh', type=float, default=0.001,
    #                     help='tracking confidence threshold')
    # parser.add_argument('--track_diff', type=int, default=10,
    #                     help='minimum frame difference between initializing two trackers')
    # parser.add_argument('--out_suffix', type=str, default='', help='out_suffix')
    #
    # paramparse.fromParser(parser, 'YOLOv3Params')
    # opt = parser.parse_args()
    # print(opt)