# Sample script
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

# Execute from object_detection folder
# Use: python csv_to_record.py --input_path="DIRECTORY TO CSV FILE" --outpath_path="DIRECTORY TO RECORD FILE"

import pandas as pd
import os, sys
# sys.path.append("..")

from utilities import processArguments

params = {
    'input_path': '../object_detection/data/train.csv',
    'img_path': '../object_detection/images/train',
    'output_path': 'train',
    'labels_path': '../labelling_tool/data/predefined_classes.txt',
}
processArguments(sys.argv[1:], params)
input_path = params['input_path']
img_path = params['img_path']
output_path = params['output_path']
labels_path = params['labels_path']

if not os.path.isdir(output_path):
    os.makedirs(output_path)

classes = [line.strip() for line in open(labels_path, 'r').readlines()]


def convert(size, box):
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
    return (x, y, w, h)


# Function to convert csv to YOLO labels
def csv_to_yolo(multiple_instance):
    # Get metadata
    filename = multiple_instance.iloc[0].loc['filename']
    width = float(multiple_instance.iloc[0].loc['width'])
    height = float(multiple_instance.iloc[0].loc['height'])

    file_no_ext = os.path.splitext(filename)[0]
    out_file_path = os.path.join(output_path, '{}.txt'.format(file_no_ext))
    out_file = open(out_file_path, 'w')

    for instance in range(0, len(multiple_instance.index)):
        xmin = multiple_instance.iloc[instance].loc['xmin']
        ymin = multiple_instance.iloc[instance].loc['ymin']
        xmax = multiple_instance.iloc[instance].loc['xmax']
        ymax = multiple_instance.iloc[instance].loc['ymax']
        class_name = multiple_instance.iloc[instance].loc['class']
        class_id = classes.index(class_name)

        bb = convert((width, height), [xmin, xmax, ymin, ymax])
        out_file.write('{:d} {:f} {:f} {:f} {:f}\n'.format(class_id, bb[0], bb[1], bb[2], bb[3]))

    out_file.close()
    return filename


if __name__ == '__main__':
    # Read .csv and store as dataframe
    df = pd.read_csv(input_path)
    df = df.sort_values(by='filename')

    list_path = os.path.join(output_path, 'list.txt')
    list_file = open(list_path, 'w')

    n_files = len(df.index)
    file_id = 0

    # Collect instances of objects and remove from df
    while not df.empty:
        try:
            # Look for objects with similar filenames, group them, send them to csv_to_record function and remove from df
            df_multiple_instance = df.loc[df['filename'] == df.iloc[0].loc['filename']]
        except IndexError:
            continue
        # Total # of object instances in a file
        no_instances = len(df_multiple_instance.index)

        if no_instances == 0:
            continue

        # Remove from df (avoids duplication)
        df = df.drop(df_multiple_instance.index[:no_instances])
        # Send all object instances of a filename to become yolo labels
        filename = csv_to_yolo(df_multiple_instance)
        list_file.write('{:s}\n'.format(os.path.join(img_path, filename)))

        file_id += 1
        sys.stdout.write('Done {:d} files. filename: {} no_instances: {}\n'.format(file_id, filename, no_instances))
        sys.stdout.flush()

    list_file.close()
    sys.stdout.write('\n')
    sys.stdout.flush()
