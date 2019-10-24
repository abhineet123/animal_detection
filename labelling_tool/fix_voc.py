import os, sys, glob, re
from libs.pascal_voc_io import PascalVocReader

sys.path.append('..')
from tf_api.utilities import processArguments


def saveBoxesTXT(_type, voc_path, class_dict, out_dir=''):
    if _type == 0:
        _type_str = 'mAP'
    else:
        _type_str = 'yolo'

    if not voc_path or not os.path.isdir(voc_path):
        print('Folder containing the loaded boxes does not exist')
        return None

    files = glob.glob(os.path.join(voc_path, '*.xml'))
    n_files = len(files)
    if n_files == 0:
        print('No loaded boxes found')
        return None

    def convert_to_yolo(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def getint(fn):
        basename = os.path.basename(fn)
        num = re.sub("\D", "", basename)
        try:
            return int(num)
        except:
            return 0

    if len(files) > 0:
        files = sorted(files, key=getint)

    if not out_dir:
        out_dir = os.path.join(os.path.dirname(voc_path), _type_str)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    list_file = None
    if _type == 1:
        list_path = os.path.join(out_dir, 'list.txt')
        list_file = open(list_path, 'w')

    print('Loading VOC annotations from {:d} files at {:s}...'.format(n_files, voc_path))
    print('Writing {} annotations to {:s}...'.format(_type_str, out_dir))

    file_id = 0
    n_boxes = 0
    for file in files:
        file_no_ext = os.path.splitext(os.path.basename(file))[0]

        out_file_path = os.path.join(out_dir, '{}.txt'.format(file_no_ext))
        out_file = open(out_file_path, 'w')

        xml_reader = PascalVocReader(file)
        shapes = xml_reader.getShapes()

        img_width = xml_reader.width
        img_height = xml_reader.height

        for shape in shapes:
            label, points, _, _, difficult, bbox_source, id_number, score, _, _ = shape

            xmin, ymin = points[0]
            xmax, ymax = points[2]

            def clamp(x, min_value=0.0, max_value=1.0):
                return max(min(x, max_value), min_value)

            xmin = int(clamp(xmin, 0, img_width-1))
            xmax = int(clamp(xmax, 0, img_width-1))

            ymin = int(clamp(ymin, 0, img_height-1))
            ymax = int(clamp(ymax, 0, img_height-1))

            if _type == 0:
                out_file.write('{:s} {:d} {:d} {:d} {:d}\n'.format(label, xmin, ymin, xmax, ymax))
            else:
                class_id = class_dict[label] + 1
                bb = convert_to_yolo((xml_reader.width, xml_reader.height), [xmin, xmax, ymin, ymax])
                out_file.write('{:d} {:f} {:f} {:f} {:f}\n'.format(class_id, bb[0], bb[1], bb[2], bb[3]))
            if _type == 1:
                list_file.write('{:s}\n'.format(xml_reader.filename))
            n_boxes += 1

        file_id += 1
        sys.stdout.write('\rDone {:d}/{:d} files with {:d} boxes ({:d}x{:d})'.format(
            file_id, n_files, n_boxes, img_width, img_height))
        sys.stdout.flush()

        out_file.close()
    if _type == 1:
        list_file.close()

    sys.stdout.write('\n')
    sys.stdout.flush()

    return out_dir


if __name__ == '__main__':
    params = {
        'list_file': 'vis_list.txt',
        'class_names_path': '../labelling_tool/data//predefined_classes_orig.txt',
        'type': 0,
        # 'file_name': 'videos/grizzly_bear_video.mp4',
        'out_dir': '',
        'save_dir': '',
        'save_file_name': '',
        'csv_file_name': '',
        'map_folder': '',
        'load_path': '',
        'n_classes': 4,
        'img_ext': 'png',
        'batch_size': 1,
        'show_img': 0,
        'save_video': 1,
        'n_frames': 0,
        'codec': 'H264',
        'fps': 20,
    }
    processArguments(sys.argv[1:], params)
    list_file = params['list_file']
    class_names_path = params['class_names_path']
    _type = params['type']
    out_dir = params['out_dir']

    class_names = open(class_names_path, 'r').readlines()
    class_dict = {x.strip(): i for (i, x) in enumerate(class_names)}

    with open(list_file) as f:
        img_paths = f.readlines()
    img_paths = [x.strip() for x in img_paths]
    for img_path in img_paths:
        voc_path = os.path.join(img_path, 'annotations')
        seq_out_dir = out_dir
        if seq_out_dir:
            seq_name = os.path.basename(img_path)
            seq_out_dir = os.path.join(seq_out_dir, seq_name)
        saveBoxesTXT(_type, voc_path, class_dict, seq_out_dir)
