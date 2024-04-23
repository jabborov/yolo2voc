import os
from PIL import Image
import argparse
from multiprocessing import Pool

import xml.etree.ElementTree as ET # Import XML Parser
from pascal_voc_writer import Writer

import config

def voc2yolo(xml_file):
    """
    Convert VOC to YOLO
    Args:
        xml_file: str
    """
   
    with open(f"{config.XML_PATH}/{xml_file}") as input_file:
        root = ET.parse(input_file)
        size = root.getroot().find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        height, width = map(int, [height, width])

    class_exists = False
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name in config.names:
            class_exists = True
    
    if class_exists:
        with open(f"{config.LABEL_PATH}/{xml_file[:-4]}.txt", "w") as out_file:

            for obj in root.iter('object'):
                name = obj.find('name').text

                if name in config.names:
                    xml_box = obj.find('bndbox')
                    x_min = float(xml_box.find('xmin').text)
                    y_min = float(xml_box.find('ymin').text)

                    x_max = float(xml_box.find('xmax').text)
                    y_max = float(xml_box.find('ymax').text)

                    box_x = (x_min + x_max) / 2.0 - 1
                    box_y = (y_min + y_max) / 2.0 - 1
                    box_w = x_max - x_min
                    box_h = y_max - y_min
                    box_x = box_x * 1.0 / width
                    box_w = box_w * 1.0 / width
                    box_y = box_y * 1.0 / height
                    box_h = box_h * 1.0 / height

                    b = [box_x, box_y, box_w, box_h]
                    cls_id = config.names.index(obj.find('name').text)
                    out_file.write(str(cls_id) + " " + " ".join([str(f'{a:.6f}') for a in b]) + '\n') 

def yolo2voc(txt_file): 
    """Convert YOLO to VOC
    Args:
        txt_file: str
    """
    
    w, h = Image.open(os.path.join(config.IMAGE_PATH, f'{txt_file[:-4]}.jpg')).size
    writer = Writer(f'{txt_file[:-4]}.xml', w, h)
    with open(os.path.join(config.LABEL_PATH, txt_file)) as f:
        for line in f.readlines():
            label, x_center, y_center, width, height = line.rstrip().split(' ')
            x_min = int(w * max(float(x_center) - float(width) / 2, 0))
            x_max = int(w * min(float(x_center) + float(width) / 2, 1))
            y_min = int(h * max(float(y_center) - float(height) / 2, 0))
            y_max = int(h * min(float(y_center) + float(height) / 2, 1))
            writer.addObject(config.names[int(label)], x_min, y_min, x_max, y_max)
    writer.save(os.path.join(config.XML_PATH, f'{txt_file[:-4]}.xml'))  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--yolo2voc", action="store_true", help="YOLO to VOC")
    parser.add_argument("--voc2yolo", action="store_true", help="VOC to YOLO")
    
    args = parser.parse_args()

    if args.voc2yolo:
        print("VOC to YOLO")
        xml_files = [name for name in os.listdir(config.XML_PATH) if name.endswith(".xml")]     

        with Pool(os.cpu_count()) as pool:
            pool.map(voc2yolo, xml_files)
        pool.join()

    if args.yolo2voc:
        print("YOLO to VOC")
        txt_files = [name for name in os.listdir(config.LABEL_PATH) if name.endswith(".txt")]        

        with Pool(os.cpu_count()) as pool:
            pool.map(yolo2voc, txt_files)
        pool.join()



