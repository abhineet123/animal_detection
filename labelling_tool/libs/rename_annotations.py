import os
import glob
import re

root_dir_name = '/home/abhineet/D/Datasets/GRAM/Images/DJI_0001'


# dirs = glob.glob("{:s}/*/".format(root_dir_name))
dirs = next(os.walk(root_dir_name))[1]

ann_dirs = [dir for dir in dirs if dir.startswith('annotations')]

for dir_name in ann_dirs:
    src_files = glob.glob(os.path.join(root_dir_name, dir_name, '*.xml'))
    src_files = [fn for fn in src_files if os.path.basename(fn).startswith('image')]

    print('Renaming {:d} files in {:s}'.format(len(src_files), dir_name))

    def getint(fn):
        basename = os.path.basename(fn)
        num = re.sub("\D", "", basename)
        try:
            return int(num)
        except:
            return -1


    dst_file_ids = [getint(fn) for fn in src_files]
    for src_fn, dst_id in zip(src_files, dst_file_ids):
        if dst_id < 0:
            continue
        dst_fn = os.path.join(root_dir_name, dir_name, '{:d}.xml'.format(dst_id - 1))
        os.rename(src_fn, dst_fn)