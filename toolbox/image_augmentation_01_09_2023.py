from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageEnhance
import time
import argparse


##########

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_number', default = "Class_0", type = str, help = 'Define the folder name containing all image and label training data.')
    parser.add_argument('--your_pathway', type = str, help = 'Path to the folder that includes the folders "images" and "labels"')

    args = parser.parse_args()
    return args

###########

def yolo_to_converted(input_image, input_bbox):
    """
    This function takes as input, the input image, and it's bounding box (annotation) in YOLO format (class, x-coordinate, y-coordinate, width, heigth) and converts it into VOC Pascal format. It then converts 
    VOC Pascal format into the following format (x1-coordinate, y1-coodinate, x2-coordinate, y2-coordinate, class) and returns the bounding box in a numpy array. 
    """
    h, w, d = input_image.shape
    try:
        rows, columns = input_bbox.shape
    except:
        rows = 1
        columns = input_bbox.shape[0]

    #Convert YOLO to VOC and then to a converted format (x1-coordinate, y1-coodinate, x2-coordinate, y2-coordinate, class) 
    converted_bboxes = np.zeros((rows,columns))
    bboxes_voc = np.zeros(input_bbox.shape)
    
    for i in range(rows):
        if rows == 1:
            bbox_width = float(input_bbox[3]) * w
            bbox_height = float(input_bbox[4]) * h
            center_x = float(input_bbox[1]) * w
            center_y = float(input_bbox[2]) * h
            bboxes_voc[0] = input_bbox[0] # class
            bboxes_voc[1] = round(center_x - (bbox_width / 2)) # x-coordinate
            converted_bboxes[i][0] = bboxes_voc[1] # x1-coordinate
            bboxes_voc[2] = round(center_y - (bbox_height / 2)) # y-coordinate
            converted_bboxes[i][1] = bboxes_voc[2] # y1-coordinate
            bboxes_voc[3] = round(center_x + (bbox_width / 2)) # width
            converted_bboxes[i][2] = bboxes_voc[1] + bbox_width # x2-coordinate
            bboxes_voc[4] = round(center_y + (bbox_height / 2)) # height
            converted_bboxes[i][3] = bboxes_voc[2] + bbox_height # y2-coordinate
            converted_bboxes[i][4] = input_bbox[0] # class
        else:
            bbox_width = float(input_bbox[i][3]) * w
            bbox_height = float(input_bbox[i][4]) * h
            center_x = float(input_bbox[i][1]) * w
            center_y = float(input_bbox[i][2]) * h
            bboxes_voc[i][0] = input_bbox[i][0]
            bboxes_voc[i][1] = round(center_x - (bbox_width / 2)) # x-coordinate
            converted_bboxes[i][0] = bboxes_voc[i][1] # x1-coordinate
            bboxes_voc[i][2] = round(center_y - (bbox_height / 2)) # y-coordinate
            converted_bboxes[i][1] = bboxes_voc[i][2] # y1-coordinate
            bboxes_voc[i][3] = round(center_x + (bbox_width / 2)) # width
            converted_bboxes[i][2] = round(bboxes_voc[i][1] + bbox_width) # x2-coordinate
            bboxes_voc[i][4] = round(center_y + (bbox_height / 2)) # height
            converted_bboxes[i][3] = round(bboxes_voc[i][2] + bbox_height) # y2-coordinate
            converted_bboxes[i][4] = input_bbox[i][0] # class

    return converted_bboxes, bboxes_voc

def converted_to_yolo(img_aug, bboxes_aug_converted):
    """
    This function takes as input, the augmented image, and it's converted augmented bounding box (annotation) in the converted format (x1-coordinate, y1-coodinate, x2-coordinate, y2-coordinate, class) and converts it into 
    YOLO format (class, x-coordinate, y-coordinate, width, heigth). It returns the bounding box in a numpy array. 
    """
    
    h, w, d = img_aug.shape
    try:
        rows, columns = bboxes_aug_converted.shape
    except:
        rows = 1
        columns = bboxes_aug_converted.shape[0]
    
    #Convert converted format to YOLO (class, x, y, width, height)
    bboxes_aug_yolo = np.zeros((rows,columns))
    bboxes_aug_voc = np.zeros(bboxes_aug_converted.shape)
    for i in range(rows):
        if rows == 1:
            bboxes_aug_voc[i][0] = bboxes_aug_converted[i][4]  # class
            bboxes_aug_voc[i][1] = bboxes_aug_converted[i][0]  # xmin-coordinate
            bboxes_aug_voc[i][2] = bboxes_aug_converted[i][1]  # ymin-coordinate
            bboxes_aug_voc[i][3] = bboxes_aug_converted[i][2]  # xmax-coordinate
            bboxes_aug_voc[i][4] = bboxes_aug_converted[i][3]  # ymax-coordinate

            bboxes_aug_yolo[i][0] = int(bboxes_aug_voc[i][0]) # class
            bboxes_aug_yolo[i][1] = round(((bboxes_aug_voc[i][1] + bboxes_aug_voc[i][3])/2.0 -1)*(1./w), 6) # x-coordinate
            bboxes_aug_yolo[i][2] = round(((bboxes_aug_voc[i][2] + bboxes_aug_voc[i][4])/2.0 -1)*(1./h), 6) # y-coordinate
            bboxes_aug_yolo[i][3] = round((bboxes_aug_voc[i][3] - bboxes_aug_voc[i][1])*(1./w), 6)          # width
            bboxes_aug_yolo[i][4] = round((bboxes_aug_voc[i][4] - bboxes_aug_voc[i][2])*(1./h), 6)          # height

        else:
            bboxes_aug_voc[i][0] = bboxes_aug_converted[i][4] # class
            bboxes_aug_voc[i][1] = bboxes_aug_converted[i][0]  # x-coordinate
            bboxes_aug_voc[i][2] = bboxes_aug_converted[i][1]  # y-coordinate
            bboxes_aug_voc[i][3] = bboxes_aug_converted[i][2] # xmax-coordinate
            bboxes_aug_voc[i][4] = bboxes_aug_converted[i][3] # ymax-coordinate
    
            bboxes_aug_yolo[i][0] = bboxes_aug_voc[i][0] # class
            bboxes_aug_yolo[i][1] = round(((bboxes_aug_voc[i][1] + bboxes_aug_voc[i][3])/2.0 -1)*(1./w), 6) # x-coordinate
            bboxes_aug_yolo[i][2] = round(((bboxes_aug_voc[i][2] + bboxes_aug_voc[i][4])/2.0 -1)*(1./h), 6) # y-coordinate
            bboxes_aug_yolo[i][3] = round((bboxes_aug_voc[i][3] - bboxes_aug_voc[i][1])*(1./w), 6)          # width
            bboxes_aug_yolo[i][4] = round((bboxes_aug_voc[i][4] - bboxes_aug_voc[i][2])*(1./h), 6)          # height


    return bboxes_aug_voc, bboxes_aug_yolo

def random_aug_generator(img, bboxes):
    """
    This function takes as input, the input image, and it's bounding box (annotation) after converting its format to: (x1-coordinate, y1-coodinate, x2-coordinate, y2-coordinate, class). It then chooses a random image augmentation
    and applies it to the input image and generates the bounding box after the image augmentation.
    """
    number = random.randint(0, 7)
    enhancer = ImageEnhance.Brightness(Image.fromarray(img)) # Brightening and Darkening of image

    if number == 0: # Scale Augmentation
        img_aug, bboxes_aug = RandomScale(0.3, diff = True)(img.copy(), bboxes.copy())
    elif number == 1: # Horizontal Flip Augmentation
        img_aug, bboxes_aug = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
    elif number == 2: # Random Translate
        img_aug, bboxes_aug = RandomTranslate(0.3, diff = True)(img.copy(), bboxes.copy())
    elif number == 3: # Rotate Augmentation
        img_aug, bboxes_aug = RandomRotate(20)(img.copy(), bboxes.copy())
    elif number == 4: #Shear Augmentation
        img_aug, bboxes_aug = RandomShear(0.2)(img.copy(), bboxes.copy())
    elif number == 5: # Resize Augmentation
        img_aug, bboxes_aug = Resize(608)(img.copy(), bboxes.copy())
    elif number == 6: # Darkening Augmentation
        darken_factor = random.uniform(0.3, 0.8)
        img_aug = enhancer.enhance(darken_factor)
        img_aug = np.array(img_aug)
        bboxes_aug = bboxes
    elif number == 7: # Brightning Augmentation
        brighten_factor =  random.uniform(1.5, 2.5)
        img_aug = enhancer.enhance(brighten_factor)
        img_aug = np.array(img_aug)
        bboxes_aug = bboxes

    return img_aug, bboxes_aug

def to_save(train_image_file_list, img_aug, bboxes_aug):
    """
    This function takes as input, the augmented image, and it's augmented bounding box (annotation) and saves them locally to a specified directory.
    """
    directory_images = f"{args.your_pathway}/images/{args.class_number}_augmented"
    directory_labels = f"{args.your_pathway}/labels/{args.class_number}_augmented"

    if not os.path.exists(directory_images):
        os.makedirs(directory_images)
    if not os.path.exists(directory_labels):
        os.makedirs(directory_labels)

    # print("Before saving image:")  
    # print(os.listdir(directory_images))

    # Filename
    filename = f"{os.path.splitext(train_image_file_list)[0]}_augmented.jpg"
    bbox_filename = f"{os.path.splitext(train_image_file_list)[0]}_augmented.txt"
    #filename = "augmented_image_%d.jpg"%(img_)
    
    if len(bboxes_aug) != 0:
        # Change the current directory to specified directory 
        os.chdir(directory_images)
        # Converting the image to RGB to BGR as it is the input type for cv2.imwrite()
        img_aug = cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)
        # Saving the image
        cv2.imwrite(filename, img_aug)

        # Change the current directory to specified directory
        os.chdir(directory_labels)
        fmt_str = ['%.0f' if i == 0 else '%.5f' for i in range(len(bboxes_aug[0]))]
        # Saving the bounding box
        np.savetxt(bbox_filename, bboxes_aug, delimiter = ' ', fmt = fmt_str)

        print(f'{filename} and {bbox_filename} successfully saved')
    else:
        # np.savetxt(bbox_filename, bboxes_aug, delimiter = ' ')
        print(f"No bees found in image {train_image_file_list}, skipping save...")

    # # List files and directories  
    # print("After saving image:")  
    # print(os.listdir(directory_images))

    # # List files and directories  
    # print("After saving labels:")  
    # print(os.listdir(directory_labels))
    
if __name__ == "__main__":
    args = parse_args()
    
    #main(args)

    # for debugging purposes
    args.your_pathway = '/home/katharina/test_bee-finder/bee-finder/data' 
    args.class_number = 'Class_0'

    images_directory = f"{args.your_pathway}/images/{args.class_number}"
    labels_directory = f"{args.your_pathway}/labels/{args.class_number}"

    train_image_file_list = sorted(os.listdir(images_directory)) # Directory containing original training images
    train_annotations_file_list = sorted(os.listdir(labels_directory)) # Directory containing original YOLO annotation txt files

    for i in range(len(train_annotations_file_list)):
        input_image = cv2.imread(images_directory + "//" + train_image_file_list[i])[:,:,::-1]  # opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
        input_bboxes = np.genfromtxt(labels_directory + "//" + train_annotations_file_list[i], dtype=str, encoding=None, delimiter=" ") # Add YOLO annotation text files into a numpy array
        
        if input_bboxes.size != 0: # Skip empty YOLO annotation text files with images that do not contain any objects
            try:
                rows, columns = input_bboxes.shape
            except:
                rows = 1
                columns = input_bboxes.shape[0]
            
            if rows == 1: # Handles cases with only one object in an image

                #if int(float(input_bboxes[0])) == 0: # Skips performing Image Augmentation on images with class 0 (Female O.cornuta)

                # print("input_bboxes:\n {}".format(input_bboxes))

                bbox_converted, bboxes_voc = yolo_to_converted(input_image, input_bboxes)

                # print("bboxes_voc:\n {}".format(bboxes_voc))
                # print("bbox_converted:\n {}".format(bbox_converted))

                img_aug, bboxes_aug_converted = random_aug_generator(input_image, bbox_converted)

                # print("bboxes_aug_converted:\n {}".format(bboxes_aug_converted))

                bboxes_aug_voc, bboxes_aug_yolo  = converted_to_yolo(img_aug, bboxes_aug_converted)

                # print("bboxes_aug_voc:\n {}".format(bboxes_aug_voc))
                # print("bboxes_aug_yolo:\n {}".format(bboxes_aug_yolo))

                to_save(train_image_file_list[i], img_aug, bboxes_aug_yolo)

                    #plotted_img = draw_rect(img_aug, bboxes_aug_converted)
                    #plt.imshow(plotted_img)
                    #plt.show()
                #else:
                    #pass
            else: # Handles cases with several objects in an image

                #if 0 not in input_bboxes[:0].astype(int):

                # print("input_bboxes:\n {}".format(input_bboxes))

                bbox_converted, bboxes_voc = yolo_to_converted(input_image, input_bboxes)

                # print("bboxes_voc:\n {}".format(bboxes_voc))
                # print("bbox_converted:\n {}".format(bbox_converted))

                img_aug, bboxes_aug_converted = random_aug_generator(input_image, bbox_converted)

                # print("bboxes_aug_converted:\n {}".format(bboxes_aug_converted))

                bboxes_aug_voc, bboxes_aug_yolo  = converted_to_yolo(img_aug, bboxes_aug_converted)

                # print("bboxes_aug_voc:\n {}".format(bboxes_aug_voc))
                # print("bboxes_aug_yolo:\n {}".format(bboxes_aug_yolo))

                to_save(train_image_file_list[i], img_aug, bboxes_aug_yolo)

                    #plotted_img = draw_rect(img_aug, bboxes_aug_converted)
                    #plt.imshow(plotted_img)
                    #plt.show()
                #else:
                #    pass

  
