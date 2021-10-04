# First we need to create dataframe from coco json 
import pandas as pd
import json
import os
import io
import pandas as pd
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

label_map = {
                1 :"nine",
                2: 'ten',
                3:'jack', 
                4: 'queen', 
                5: 'king', 
                6:'ace' 
            }

def convert_coco_json_to_csv(filename):
    # COCO2017/annotations/instances_val2017.json
    s = json.load(open(filename, 'r'))
    out_file = filename[:-5] + '.csv'
    out = open(out_file, 'w')
    out.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')

    all_ids = []
    dicts = {}
    for im in s['images']:
        all_ids.append(im['id'])
        filename = im['file_name'].split("/")[-1]
        dicts[im["id"]] = [im['height'], im['width'], filename]

    all_ids_ann = []
    for ann in tqdm(s['annotations']):
        image_id = ann['image_id']
        label = label_map[ann["category_id"]]
        all_ids_ann.append(image_id)
        x1 = ann['bbox'][0]
        x2 = ann['bbox'][0] + ann['bbox'][2]
        y1 = ann['bbox'][1]
        y2 = ann['bbox'][1] + ann['bbox'][3]
        out.write('{},{},{},{},{},{},{},{}\n'.format(dicts[image_id][-1],dicts[image_id][1],dicts[image_id][0],label, x1, y1, x2, y2))

    all_ids = set(all_ids)
    all_ids_ann = set(all_ids_ann)
    no_annotations = list(all_ids - all_ids_ann)
    # Output images without any annotations
    for image_id in no_annotations:
        out.write('{},{},{},{},{},{},{},{}\n'.format(dicts[image_id][-1],dicts[image_id][1],dicts[image_id][0], -1, -1, -1, -1, -1))
    out.close()

    dataframe = pd.read_csv(out_file)
    return dataframe 

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
convert_coco_json_to_csv(r"C:\Users\chand\Desktop\Jobs Scrap\finalcoco\test.json")

if __name__ == "__main__":
    ##########################################################################
    image_dir = r"D://github//Data-Converter-and-Data-Loader-//images//Object detection Dataloader//inputs//test"
    json_path = r"D://github//Data-Converter-and-Data-Loader-//images//Object detection Dataloader//inputs//test_labels.json"
    output_path_save = "test.record"
    ###########################################################################

    writer = tf.io.TFRecordWriter(output_path_save)
    path = os.path.join(os.getcwd(), image_dir)
    print("Creating dataframe .....")
    examples = convert_coco_json_to_csv(json_path)
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




