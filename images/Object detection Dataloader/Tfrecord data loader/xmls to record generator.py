# Convert VOC to CSV and then create record 
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import io
from PIL import Image
from utils import dataset_util
from collections import namedtuple, OrderedDict
import tensorflow.compat.v1 as tf
from tqdm import tqdm

######################### Replace this with label map ########################################### 
def class_text_to_int(row_label):
    if row_label == 'nine':
        return 1
    elif row_label == 'ten':
        return 2
    elif row_label == 'jack':
        return 3
    elif row_label == 'queen':
        return 4
    elif row_label == 'king':
        return 5
    elif row_label == 'ace':
        return 6
    else:
        None
################################################################################################

def xml_to_csv(path):
    """
    Function take path of folder where xmls are present and convert it into
    dataframe with columns 
    ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    Params:
        path - path to the folder which contains xml
    Return:
        returns a dataframe  
    """
    xml_list = []
    for xml_file in tqdm(glob.glob(path + '/*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    
    return xml_df


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

# Create df 
def create_tf_example(group, path):
    """
    Note - The csv must be in the following format 

    filename,width,height,class,xmin,ymin,xmax,ymax
    where 
        xmin - top left co-ordinate of x
        ymin - top left co-ordinate of y
        xmax - bottom right co-ordinate of x
        ymax = bottom right co-ordinate of y
         
        and image origin should be start from top left 
    """

    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

if __name__ == "__main__":
    ##########################################################################
    image_dir = r"D://github//Data-Converter-and-Data-Loader-//images//Object detection Dataloader//inputs//test"
    xml_path = r"D://github//Data-Converter-and-Data-Loader-//images//Object detection Dataloader//inputs//test"
    output_path_save = "test.record"
    ###########################################################################

    writer = tf.io.TFRecordWriter(output_path_save)
    path = os.path.join(os.getcwd(), image_dir)
    print("Creating dataframe .....")
    examples = xml_to_csv(xml_path)
    print("Dataframe creation Done .......", end="\n\n")
    print("Creating Tfrecord .....")
    grouped = split(examples, 'filename')
    for group in tqdm(grouped):
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    print("Tfrecord creation Done .......", end="\n\n")
    writer.close()
    output_path = os.path.join(os.getcwd(), output_path_save)
    print('Successfully created the TFRecords: {}'.format(output_path))
































