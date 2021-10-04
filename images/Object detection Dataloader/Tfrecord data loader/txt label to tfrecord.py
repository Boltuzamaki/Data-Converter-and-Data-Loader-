import glob
import os
import cv2
from tqdm import tqdm
from pascal_voc_writer import Writer
def txt_to_voc(path_annotation, path_image):
    """
    This function helps to create pascal VOC xml files from labels
    present in txt format
    Args:
        path_annotation: Path to the annotation folder which contains txt format label
        path_image: Path to the image folder which contain images of the corresponding
                    annotation
    Return:
        Convert all txt format label one by one and save it in the txt folder
    
    Note:
        Format of txt file should be 
    """
    txt_files = glob.glob(path_annotation+"/*.txt")
    for tfiles in tqdm(txt_files):
        txt = tfiles
        image = txt.split(".txt")[0]+ ".jpg"
        image = image.split('\\')[-1]
        path = image
       # print(image)
        pm = os.path.join(path_image, image)
        im = cv2.imread(pm)
        height, width, _ = im.shape
        file = open(txt, "r")
        writer = Writer(path, width, height)
        for f in file:
            class_name = "people"
            x_min = int(float(f.split(" ")[0]))
            y_min = int(float(f.split(" ")[1]))
            x_max = int(float(f.split(" ")[2]))
            y_max = int(float(f.split(" ")[3].split("\\")[0]))
            writer.addObject(class_name, x_min, y_min, x_max, y_max)
        xml_p = txt.split(".txt")[0] + ".xml"
        writer.save(xml_p)


txt_to_voc(r"C:\Users\chand\Desktop\sep data\content\drive\MyDrive\Datasets\80\80m_10X\annotation", r"C:\Users\chand\Desktop\sep data\content\drive\MyDrive\Datasets\80\80m_10X\images")
