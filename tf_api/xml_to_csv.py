import xml.etree.ElementTree as ET
import pandas as pd
import os, sys
from utilities import processArguments


params = {
    'base_path': '/home/abhineet/N/Datasets/Acamp/',
    # List of the strings that is used to add correct label for each box.
    'seq_name': 'test',
    # 'file_name': 'videos/grizzly_bear_video.mp4',
    'save_dir': '',
    'save_file_name': '',
    'csv_file_name': '',
    'map_folder': '',
    'load_path': '',
    'n_classes': 4,
    'img_ext': 'png',
    'batch_size': 1,
    'show_img': 0,
    'n_frames': 0,
    'codec': 'H264',
    'fps': 20,
}

processArguments(sys.argv[1:], params)
base_path = params['base_path']
seq_name = params['seq_name']
# n_classes = params['n_classes']
# file_name = params['file_name']
# save_dir = params['save_dir']
# save_file_name = params['save_file_name']
# csv_file_name = params['csv_file_name']
# map_folder = params['map_folder']
# load_path = params['load_path']
# img_ext = params['img_ext']
# batch_size = params['batch_size']
# show_img = params['show_img']
# n_frames = params['n_frames']
# codec = params['codec']
# fps = params['fps']

# Column Names for CSV File
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']


# Scan folder for .xml files and convert to csv
def convert_xml_to_csv(path):
    xml_raw = []
    for file in os.listdir(path):
        # Validate .xml File & retrieve raw data
        file_no_ext = os.path.splitext(os.path.basename(file))[0]
        if file.endswith('.xml'):
            tree = ET.parse(path + '/' + file)
            root = tree.getroot()
            for member in root.findall('object'):
                _ext = os.path.splitext(root[1].text)[1]
                raw_data = {
                    # 'filename': root[1].text,
                    'filename': '{}{}'.format(file_no_ext, _ext),
                    'width': int(root[4][0].text),
                    'height': int(root[4][1].text),
                    'class': member[0].text,
                    'xmin': int(member[4][0].text),
                    'ymin': int(member[4][1].text),
                    'xmax': int(member[4][2].text),
                    'ymax': int(member[4][3].text)
                }
                xml_raw.append(raw_data)
    df = pd.DataFrame(xml_raw)
    return df


# Main function
def main():
    new_path = os.path.join(base_path, seq_name, 'annotations')
    df = convert_xml_to_csv(new_path)
    csv_file = os.path.join(base_path, seq_name, 'annotations.csv')
    # print csv_file
    df.to_csv(csv_file)


main()
