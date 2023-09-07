import os
import numpy as np
import random
import shutil
import argparse

### 
#
#If - only if! - you have more than one classes, conduct the code for each class individually and remove the sorted images out 
# of the folder structure for not overwriting. If not, it is not required to change anything.


def parse_args():
    parser = parser = argparse.ArgumentParser()
    parser.add_argument('--class_number', default = "Class_0", type = str, help = 'Define the folder name containing all image and label training data.')
    parser.add_argument('--your_pathway', type = str, help = 'Path to the folder "toolbox" that includes the folders "images" and "labels"')

    args = parser.parse_args()
    return args


def move_random_imgs_to_train_test_val(your_pathway, class_number):
    """
    Move images and their annotations to the train/validation/test folders based on the 70%, 20%, and 10% rule
    """
    train_images_directory = f"{your_pathway}/images/train"
    val_images_directory = f"{your_pathway}/images/val"
    test_images_directory = f"{your_pathway}/images/test"

    train_labels_directory = f"{your_pathway}/labels/train"
    val_labels_directory = f"{your_pathway}/labels/val"
    test_labels_directory = f"{your_pathway}/labels/test"

    class_0_images_directory = f"{your_pathway}/images//{class_number}"

    class_0_images_file_list = os.listdir(class_0_images_directory)

    class_0_labels_directory = f"{your_pathway}/labels//{class_number}"

    class_0_labels_file_list = os.listdir(class_0_labels_directory)

    class_0_train_labels = random.sample(class_0_labels_file_list, k=round(len(class_0_labels_file_list)*0.7))
    class_0_labels_file_list = list(set(class_0_labels_file_list) - set(class_0_train_labels))

    class_0_val_labels = random.sample(class_0_labels_file_list, k=round(len(class_0_labels_file_list)*(2/3)))
    class_0_labels_file_list = list(set(class_0_labels_file_list) - set(class_0_val_labels))

    class_0_test_labels = random.sample(class_0_labels_file_list, k=round(len(class_0_labels_file_list)))
    class_0_labels_file_list = list(set(class_0_labels_file_list) - set(class_0_test_labels))

    # Moving Training Images and Labels to images/train and labels/train folder 
    for i in range(len(class_0_train_labels)):
        class_0_train_image = os.path.splitext(class_0_train_labels[i])[0] + '.jpg'
        print(class_0_train_image)
        print(class_0_train_labels[i])
        print(".../n...")
        shutil.copy2(class_0_labels_directory + "//" + class_0_train_labels[i], train_labels_directory + "//" + class_0_train_labels[i])
        shutil.copy2(class_0_images_directory + "//" + class_0_train_image, train_images_directory + "//" + class_0_train_image) 
   
    # Moving Validation Images and Labels to images/val and labels/val folder  
    for m in range(len(class_0_val_labels)):
        class_0_val_image = os.path.splitext(class_0_val_labels[m])[0] + '.jpg'
        shutil.copy2(class_0_labels_directory + "//" + class_0_val_labels[m], val_labels_directory + "//" + class_0_val_labels[m])
        shutil.copy2(class_0_images_directory + "//" + class_0_val_image, val_images_directory + "//" + class_0_val_image)        

    # Moving Test Images and Labels to images/test and labels/test folder 
    for q in range(len(class_0_test_labels)):
        class_0_test_image = os.path.splitext(class_0_test_labels[q])[0] + '.jpg'
        shutil.copy2(class_0_labels_directory + "//" + class_0_test_labels[q], test_labels_directory + "//" + class_0_test_labels[q])
        shutil.copy2(class_0_images_directory + "//" + class_0_test_image, test_images_directory + "//" + class_0_test_image)   


def move_imgs_to_corresponding_class(your_pathway, class_number):
    """
    Moves images and annotations to their corresponding class folder 
    Please change:
    - train_images_directory
    - train_labels_directory

    
    """
    train_images_directory = f"{your_pathway}/images/train"
    train_labels_directory = f"{your_pathway}/labels/train"

    #train_image_file_list = os.listdir(train_images_directory) # Directory containing original training images
    train_annotations_file_list = os.listdir(train_labels_directory) # Directory containing YOLO annotation txt files

    for i in range(len(train_annotations_file_list)):
        input_bboxes = np.genfromtxt(f"{train_labels_directory}/{train_annotations_file_list[i]}", dtype=str, encoding=None, delimiter=" ") # Add YOLO annotation text files into a numpy array
        image_name = f"{os.path.splitext(train_annotations_file_list[i])[0]}.jpg"

        if input_bboxes.size != 0: # Skip empty YOLO annotation text files with images that do not contain any objects
            try:
                rows, columns = input_bboxes.shape
            except:
                rows = 1
                columns = input_bboxes.shape[0] 
            if rows == 1:
                if int(float(input_bboxes[0])) == 0:
                    os.rename(train_labels_directory + "//" + train_annotations_file_list[i],f"{your_pathway}/labels//{class_number}" + "//" + train_annotations_file_list[i])
                    os.rename(train_images_directory + "//" + image_name, f"{your_pathway}/images//{class_number}" + "//" + image_name)


def main(args):

    move_imgs_to_corresponding_class(args.your_pathway, args.class_number)
    move_random_imgs_to_train_test_val(args.your_pathway, args.class_number)


if __name__ == "__main__":
    args = parse_args()
    main(args)


