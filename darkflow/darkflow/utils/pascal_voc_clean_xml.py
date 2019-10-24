"""
parse PASCAL VOC xml annotations
"""

import os
import sys
import xml.etree.ElementTree as ET
import glob


def _pp(l):  # pretty printing
    for i in l: print('{}: {}'.format(i, l[i]))


def pascal_voc_clean_xml(ANN_list, IMG_list, pick, exclusive=False):
    print('Parsing for {} {}'.format(
        pick, 'exclusively' * int(exclusive)))

    dumps = list()

    for seq_idx, ANN in enumerate(ANN_list):
        cur_dir = os.getcwd()
        os.chdir(ANN)

        annotations = os.listdir('.')
        annotations = glob.glob(str(annotations) + '*.xml')
        size = len(annotations)

        print('\nReading annotations from {}'.format(ANN))
        IMG = IMG_list[seq_idx]

        for i, file in enumerate(annotations):
            # progress bar
            sys.stdout.write('\r')
            percentage = 1. * (i + 1) / size
            progress = int(percentage * 20)
            bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
            bar_arg += [file]
            sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
            sys.stdout.flush()

            # actual parsing
            in_file = open(file)
            tree = ET.parse(in_file)
            root = tree.getroot()
            jpg = str(root.find('filename').text)

            path = os.path.join(IMG, jpg)
            if not os.path.exists(path):
                # fixed_jpg = 'image{:06d}'.format(int(jpg.split('image')[1].split('.')[0]) + 1)
                file_name_no_ext = os.path.splitext(os.path.basename(file))[0]
                # if fixed_jpg != file_name_no_ext:
                #     text = 'Something weird happening:\n'
                #     text += 'file: {}\n'.format(file)
                #     text += 'fixed_jpg: {}\n'.format(fixed_jpg)
                #     text += 'file_name_no_ext: {}\n'.format(file_name_no_ext)
                #     raise IOError(text)

                fixed_jpg = '{}.jpg'.format(file_name_no_ext)
                fixed_path = os.path.join(IMG, fixed_jpg)
                if not os.path.exists(fixed_path):
                    continue
                jpg = fixed_jpg

            imsize = root.find('size')
            w = int(imsize.find('width').text)
            h = int(imsize.find('height').text)
            all = list()

            for obj in root.iter('object'):
                current = list()
                name = obj.find('name').text
                if name not in pick:
                    continue

                xmlbox = obj.find('bndbox')
                xn = int(float(xmlbox.find('xmin').text))
                xx = int(float(xmlbox.find('xmax').text))
                yn = int(float(xmlbox.find('ymin').text))
                yx = int(float(xmlbox.find('ymax').text))
                current = [name, xn, yn, xx, yx]
                all += [current]

            add = [[seq_idx, jpg, [w, h, all]]]
            dumps += add
            in_file.close()
        os.chdir(cur_dir)
    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]] += 1
                else:
                    stat[current[0]] = 1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))
    return dumps
