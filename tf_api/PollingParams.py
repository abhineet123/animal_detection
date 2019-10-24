class PollingParams:
    """
    :param int allow_missing_detections: allow_missing_detections
    :param str class_names_path: Path to file containing class names
    :param str csv_paths: List of paths to csv annotations
    :param str csv_root_dir: Path to csv files
    :param float even_sampling: use evenly spaced sampling (< 1 would draw samples from only a fraction of the sequence; < 0 would invert the sampling)
    :param int fixed_ar: pad images to have fixed aspect ratio
    :param int inverted_sampling: invert samples defined by the remaining sampling parameters
    :param list load_samples: load_samples
    :param str load_samples_root: load_samples_root
    :param int min_size: min_size
    :param int n_frames: n_frames
    :param int only_sampling: only_sampling
    :param str output_path: Path to save the polled csv files
    :param int polling_type: polling_type: 0: single highest confidence detection in each frame;1: pool all detections and remove duplicates based on IOU;
    :param int random_sampling: enable random sampling
    :param str root_dir: Path to input files
    :param int samples_per_class: no. of samples to include per class; < 0 would sample from the end
    :param float sampling_ratio: proportion of images to include in the tfrecord file
    :param str seq_paths: List of paths to image sequences
    :param int shuffle_files: shuffle files
    """

    def __init__(self):
        self.cfg = ('',)
        self.allow_missing_detections = 1
        self.class_names_path = ''
        self.csv_paths = ''
        self.csv_root_dir = ''
        self.even_sampling = 0.0
        self.fixed_ar = 0
        self.inverted_sampling = 0
        self.load_samples = []
        self.load_samples_root = ''
        self.min_size = 1
        self.n_frames = 0
        self.only_sampling = 0
        self.output_path = ''
        self.polling_type = 0
        self.random_sampling = 0
        self.root_dir = ''
        self.samples_per_class = 0
        self.sampling_ratio = 1.0
        self.seq_paths = ''
        self.shuffle_files = 1
        self.conf_thresh = ''
        self.discard_below_thresh = 0
        self.allow_seq_skipping = 1
        self.help = {
            '__desc__': 'CSV to TFRecord Converter',
            'allow_missing_detections': 'allow_missing_detections',
            'class_names_path': 'Path to file containing class names',
            'csv_paths': 'List of paths to csv annotations',
            'csv_root_dir': 'Path to csv files',
            'even_sampling': 'use evenly spaced sampling (< 1 would draw samples from only a fraction of the sequence; < 0 would invert the sampling)',
            'fixed_ar': 'pad images to have fixed aspect ratio',
            'inverted_sampling': 'invert samples defined by the remaining sampling parameters',
            'load_samples': 'load_samples',
            'load_samples_root': 'load_samples_root',
            'min_size': 'min_size',
            'n_frames': 'n_frames',
            'only_sampling': 'only_sampling',
            'output_path': 'Path to save the polled csv files',
            'polling_type': 'polling_type: 0: single highest confidence detection in each frame;1: pool all detections and remove duplicates based on IOU;',
            'random_sampling': 'enable random sampling',
            'root_dir': 'Path to input files',
            'samples_per_class': 'no. of samples to include per class; < 0 would sample from the end',
            'sampling_ratio': 'proportion of images to include in the tfrecord file',
            'seq_paths': 'List of paths to image sequences',
            'shuffle_files': 'shuffle files',
        }
