import cv2
import os
import torch
import argparse
import glob
import numpy as np
import pandas as pd
import datetime, time
import copy
#from paddleocr import PaddleOCR
import logging
import math
#import ffmpeg #hashtagged this as ffmpeg is already installed in yolo_env, otherwise error KW

def convert_to_mp4(path_to_video):
    """
    This method converts an input video to mp4 format.
    """
    fps = 60
    full_path = os.path.split(path_to_video) 
    file_name = os.path.splitext(full_path[1])[0]
    save_to_path = f"{full_path[0]}/{file_name}.mp4"

    logging.info(f"---------------------------------------------------------------------------------")
    logging.info(f"Converting video format to mp4")
    logging.info(f"---------------------------------------------------------------------------------")

    time.sleep(1)
    os.system(f'ffmpeg -framerate {fps} -i "{path_to_video}" -c copy "{save_to_path}"')

    return save_to_path

def vid_to_frames(path_to_video, fps):
    """
    This method converts an input video into frames with a set frame per second value using ffmpeg. 
    It also creates a folder with the same name of the video containing all the frames.
    """
    full_path = os.path.split(path_to_video) 
    file_name = os.path.splitext(full_path[1])
    save_to_dir = f"{full_path[0]}/{file_name[0]}"
    
    logging.info(f"---------------------------------------------------------------------------------")
    logging.info(f"Converting video to frames with {fps} frames per second")
    logging.info(f"---------------------------------------------------------------------------------")

    os.system(f'ffmpeg -i "{path_to_video}" -q:v 2 -r {fps} -start_number 0 "{save_to_dir}"/image_%08d.jpg')
    os.remove(path_to_video)
   
def frames_to_vid(save_to_dir, fps):
    """
    This method generates an mp4 video from frames with a set frame per second value using ffmpeg.
    """
    image_list = []
    os.chdir(f"{save_to_dir}/bees_detected")

    if len(os.listdir(os.getcwd())) == 0:
       logging.info(f"No bees were found in video '{os.path.split(save_to_dir)[1]}.h264'") 
    else:
        for file_name in sorted(glob.glob("*.jpg")):
            image_list.append(file_name)
        with open('image_list.txt', mode='wt', encoding='utf-8') as myfile:
            myfile.write('\n'.join(['file ' + sub for sub in image_list]))

        os.chdir(f"{save_to_dir}/outputs")
    
        os.system(f'ffmpeg -r {fps} -f concat -i "{save_to_dir}/bees_detected/image_list.txt" {os.path.split(save_to_dir)[1]}_bees_reconstructed.mp4')

def import_frames_to_list(save_to_dir):
    """
    This method retrieves images in a specified directory and puts them in a sorted list (based on frame sequence).
    """
    image_list = []
    for filename in glob.glob(f"{save_to_dir}/*.jpg"):

        image_list.append(filename)

    return sorted(image_list)

def yolo_detect(model_weights, batch_size, image_list, save_to_dir, extract_timestamp, cut, window_size):
    """
    This method applies the trained YOLO bee detector and outputs the result into a DataFrame.
    """
    # First time initialization 
    no_occ_prev = {'ocornuta'   : 10000,
                   'blue_tag'   : 10000,
                   'red_tag'    : 10000,
                   'yellow_tag' : 10000,
                   'white_tag'  : 10000
    }

    # First time initialization for counter of count_ocornuta
    bee_counter = {'ocornuta'   : 0,
                   'blue_tag'   : 0,
                   'red_tag'    : 0,
                   'yellow_tag' : 0,
                   'white_tag'  : 0
    }

    no_occ_accum = {'ocornuta'   : np.zeros(window_size),
                    'blue_tag'   : np.zeros(window_size),
                    'red_tag'    : np.zeros(window_size),
                    'yellow_tag' : np.zeros(window_size),
                    'white_tag'  : np.zeros(window_size)
    }

    count = 0
    # Model
    model_dir = f"{os.getcwd()}/bee-finder"

    model = torch.hub.load(model_dir, 'custom', path = model_weights, source = 'local') #changed to directory with yolov5 folder KW
    # model = torch.hub.load('/home/katharina/yolov5_modified', 'custom', path = model_weights, source = 'local') #changed to directory with yolov5 folder KW
    # model = torch.hub.load('/home/katharina/bee-finder', 'custom', path = model_weights, source = 'local') #tryout bee_finder KW
    model.to(cuda_device)

    imgs = []
    imgs_names = []
    i = 1
    if extract_timestamp == 'True':
        ocr = PaddleOCR(use_angle_cls = True, lang='en', show_log = False, use_gpu = True)
        bees_df = pd.DataFrame(columns = ['Image_Name', 'Timestamp', 'X1_Coordinate', 'Y1_Coordinate','X2_Coordinate', 'Y2_Coordinate', 'Confidence', 'Class_Number', 'Class_Name'])

        for image in image_list:
            imgs_names.append(os.path.split(image)[1])
            img = cv2.imread(image)[..., ::-1]
            imgs.append(img)
            if i % batch_size == 0 or i == len(image_list):
                torch.cuda.empty_cache()
                batch_pred = model(imgs)
                z = 0
                for z in range(len(batch_pred)): # Updating file names after passing through YOLO model from image0 to image_0000 syntax
                    batch_pred.files[z] = imgs_names[z]

                if len(imgs) == batch_size:
                    logging.info(f"\n")
                    logging.info(f"---------------------------------------------------------------------------------")
                    logging.info(f"Batch Number: {int(i / batch_size)}")
                    logging.info(f"Processing Images: {i - batch_size} to {i - 1}")
                    logging.info(f"Total No. of Images: {len(image_list)}")
                    
                    bee_count(batch_pred, window_size, no_occ_prev, bee_counter, no_occ_accum, count)

                    bees_detected_dir, bees_df = move_detected_objects_to_dir(batch_pred, save_to_dir, extract_timestamp, cut, bees_df=bees_df, ocr=ocr)
                elif len(imgs) < batch_size: # This means we are in the last batch (but will be less than 100 images to process)
                    logging.info(f"---------------------------------------------------------------------------------")
                    logging.info(f"Batch Number: {math.ceil(i / batch_size)}")
                    logging.info(f"Processing Images: {i - batch_size} to {i - 1}")
                    logging.info(f"Total No. of Images: {len(image_list)}")
                    
                    bee_count(batch_pred, window_size, no_occ_prev, bee_counter, no_occ_accum, count)

                    bees_detected_dir, bees_df = move_detected_objects_to_dir(batch_pred, save_to_dir, extract_timestamp, cut, bees_df=bees_df, ocr=ocr)
                imgs = []
                imgs_names = []
            i+=1
        bees_df.to_csv(f"{save_to_dir}/outputs/{os.path.split(save_to_dir)[1]}_model_results.csv")
        return bees_detected_dir, bees_df         
    elif extract_timestamp == 'False':
        for image in image_list:
            imgs_names.append(os.path.split(image)[1])
            img = cv2.imread(image)[..., ::-1]
            imgs.append(img)
            if i % batch_size == 0 or i == len(image_list):
                torch.cuda.empty_cache()
                batch_pred = model(imgs)
                z = 0
                for z in range(len(batch_pred)): # Updating file names after passing through YOLO model from image0 to image_0000 syntax
                    batch_pred.files[z] = imgs_names[z]

                if len(imgs) == batch_size:
                    logging.info(f"\n")
                    logging.info(f"---------------------------------------------------------------------------------")
                    logging.info(f"Batch Number: {int(i / batch_size)}")
                    logging.info(f"Processing Images: {i - batch_size} to {i - 1}")
                    logging.info(f"Total No. of Images: {len(image_list)}")
                    
                    bee_count(batch_pred, window_size, no_occ_prev, bee_counter, no_occ_accum, count)

                    bees_detected_dir = move_detected_objects_to_dir(batch_pred, save_to_dir, extract_timestamp, cut)
                elif len(imgs) < batch_size: # This means we are in the last batch (but will be less than 100 images to process)
                    logging.info(f"---------------------------------------------------------------------------------")
                    logging.info(f"Batch Number: {math.ceil(i / batch_size)}")
                    logging.info(f"Processing Images: {i - batch_size} to {i - 1}")
                    logging.info(f"Total No. of Images: {len(image_list)}")

                    bee_count(batch_pred, window_size, no_occ_prev, bee_counter, no_occ_accum, count)
                        
                    bees_detected_dir = move_detected_objects_to_dir(batch_pred, save_to_dir, extract_timestamp, cut)
                imgs = []
                imgs_names = []
            i+=1
        return bees_detected_dir 

def bee_count(batch_pred, window_size, no_occ_prev, bee_counter, no_occ_accum, count):
    """
    This method is responsible for counting the bees.
    """
    batch_pred = batch_pred.pandas().xyxy # DF with size of batch_size

    for i in range(len(batch_pred)):
        no_occ_accum['ocornuta'][count] = np.count_nonzero(batch_pred[i]['class'].values == 0)
        if count == window_size - 1:
            count = 0
            no_occ_ocornuta_now = round(np.median(no_occ_accum['ocornuta']))
            if no_occ_ocornuta_now > no_occ_prev['ocornuta']:
                bee_counter['ocornuta'] = bee_counter['ocornuta'] + (no_occ_ocornuta_now - no_occ_prev['ocornuta'])
            no_occ_prev['ocornuta'] = no_occ_ocornuta_now
        
        count += 1
    logging.info(f"count_ocornuta: {bee_counter['ocornuta']}")
    logging.info(f"---------------------------------------------------------------------------------")
    logging.info(f"\n")    


def filter_list_elements(list_of_elements, list_empty_df):
    """
    This helper method removes empty dataframes from the a list.
    """   
    result = []
    for i, t in enumerate(list_of_elements):
        if i not in list_empty_df:
            result.append(t)
    
    return result

def move_detected_objects_to_dir(pred, save_to_dir, extract_timestamp, cut, **kwargs):
    """
    This method checks for detected objects from images/frames and moves them to a dedicated folder named "bees_detected". The YOLO annotation file is also saved.
    """
    if kwargs:  
        bees_df = kwargs['bees_df']
        ocr = kwargs['ocr']

    bees_detected_dir = f"{save_to_dir}/bees_detected/"

    if cut == 'True':

        pred_with_objects = copy.deepcopy(pred)

        list_empty_df = [i for i , df in enumerate(pred.pandas().xyxy) if df.empty]

        pred_with_objects.files = filter_list_elements(pred_with_objects.files, list_empty_df)
        pred_with_objects.ims = filter_list_elements(pred_with_objects.ims, list_empty_df)
        pred_with_objects.pred = filter_list_elements(pred_with_objects.pred, list_empty_df)
        pred_with_objects.xywh = filter_list_elements(pred_with_objects.xywh, list_empty_df)
        pred_with_objects.xywhn = filter_list_elements(pred_with_objects.xywhn, list_empty_df)
        pred_with_objects.xyxy = filter_list_elements(pred_with_objects.xyxy, list_empty_df)
        pred_with_objects.n = len(pred_with_objects.files)

        pred_with_objects.save(save_dir = bees_detected_dir) # Saves frames with bees detected into images
            
        if extract_timestamp == 'True':

            for index, df in enumerate(pred_with_objects.pandas().xyxy):
                #Column Names: xmin    ymin    xmax    ymax    confidence  class   name    
                np.savetxt(f'{save_to_dir}/bees_detected/{os.path.splitext(os.path.split(pred_with_objects.files[index])[1])[0]}.txt', df.values, fmt='%s')
                bees_df = bees_dataframe(df, f'{save_to_dir}/bees_detected/{pred_with_objects.files[index]}', bees_df, ocr)

            return bees_detected_dir, bees_df
        
        elif extract_timestamp == 'False':

            for index, df in enumerate(pred_with_objects.pandas().xyxy):
                #Column Names: xmin    ymin    xmax    ymax    confidence  class   name    
                np.savetxt(f'{bees_detected_dir}/{os.path.splitext(os.path.split(pred_with_objects.files[index])[1])[0]}.txt', df.values, fmt='%s')

            return bees_detected_dir

    elif cut == 'False':
        pred.save(save_dir = bees_detected_dir) # Saves frames with bees detected into images 

    if extract_timestamp == 'True':

        for index, df in enumerate(pred.pandas().xyxy):
            #Column Names: xmin    ymin    xmax    ymax    confidence  class   name    
            np.savetxt(f'{bees_detected_dir}/{os.path.splitext(os.path.split(pred.files[index])[1])[0]}.txt', df.values, fmt='%s')
            bees_df = bees_dataframe(df, f'{bees_detected_dir}/{pred.files[index]}')

        return bees_detected_dir, bees_df
    
    elif extract_timestamp == 'False':

        for index, df in enumerate(pred.pandas().xyxy):
            #Column Names: xmin    ymin    xmax    ymax    confidence  class   name    
            np.savetxt(f'{bees_detected_dir}/{os.path.splitext(os.path.split(pred.files[index])[1])[0]}.txt', df.values, fmt='%s')

        return bees_detected_dir

def extract_timestamp(image, ocr):
    """
    This method extracts the timestamp from input frames and updates the bees DF with extracted timestamps. It is then exported and saved 
    in csv format.
    """ 
    img = cv2.imread(image)
    timestamp_section = img[15:55, 360:660] # Pixels containing the timestamp
    
    #PaddleOCR Method
    timestamp = ocr.ocr(timestamp_section, cls=True)

    if not timestamp:
        timestamp = ''
        return timestamp
    else:
        return timestamp[0][0][1][0]

def bees_dataframe(df, image, bees_df, ocr):
    """
    This methods saves the results from the yolo detector into a DataFrame.
    """
    timestamp = extract_timestamp(image, ocr)

    current_bee_row = pd.DataFrame(data = {'Image_Name' : [f"{os.path.splitext(os.path.split(image)[1])[0]}.jpg"] * df.shape[0],
                                            'Timestamp' : [f"{timestamp}"] * df.shape[0],
                                            'X1_Coordinate' : list(df.values[:,0]),
                                            'Y1_Coordinate' : list(df.values[:,1]),
                                            'X2_Coordinate' : list(df.values[:,2]),
                                            'Y2_Coordinate' : list(df.values[:,3]),
                                            'Confidence'    : list(df.values[:,4]),
                                            'Class_Number'  : list(df.values[:,5]),
                                            'Class_Name'    : list(df.values[:,6])})
    bees_df = pd.concat([current_bee_row, bees_df.loc[:]]).reset_index(drop=True)   

    return bees_df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_device', default = '0', type = str, help = 'CUDA device to run script on') 
    parser.add_argument('--path_to_video', type = str, help = 'Path of Input Video to be processed')   
    parser.add_argument('--model_weights', default = '/home/katharina/Pipeline_files/best.pt', type = str, help = 'Path to model weights (the "best.pt" file)')
    parser.add_argument('--fps', default = 1, type = int, help = 'Frames Per Second for converting videos to frames')
    parser.add_argument('--batch_size', default = 128, type = int, help = 'How many images are processed per time')
    parser.add_argument('--extract_timestamp', default = 'False', choices = ('True', 'False'), help = 'When set to True, runs images into OCR timestamp extractor and saves results in a csv file')
    parser.add_argument('--cut', default = 'True', choices = ('True', 'False'), help = 'When set to True, removes images that do not contain any bees')
    args = parser.parse_args()
    return args

def process_video(video_path):
    start_time = datetime.datetime.now()

    save_to_dir = os.path.splitext(video_path)[0]

    if not os.path.exists(f"{save_to_dir}/outputs"):
        os.makedirs(f"{save_to_dir}/outputs")
    
    if not os.path.exists(f"{save_to_dir}/bees_detected/"):
        os.makedirs(f"{save_to_dir}/bees_detected/") 

    logging.basicConfig(filename = f"{save_to_dir}/outputs/run.log", format = '%(asctime)s | %(message)s', filemode = "w", level=logging.INFO)

    logging.info(f"Start Time : {start_time}")
    logging.info(f"\n")
    logging.info(f"---------------------------------------------------------------------------------")
    logging.info(f"Set Parameters : ")
    logging.info(f"cuda_device : {cuda_device}")
    logging.info(f"path_to_video : {args.path_to_video}")
    logging.info(f"model_weights : {args.model_weights}")
    logging.info(f"fps : {args.fps}")
    logging.info(f"batch_size : {args.batch_size}")
    logging.info(f"extract_timestamp : {args.extract_timestamp}")
    logging.info(f"cut : {args.cut}")

    if '.h264' in video_path:
        path_to_video_mp4 = convert_to_mp4(video_path) #unhashtagged this KW
    elif '.mp4' in video_path:
        path_to_video_mp4 = video_path
        logging.info(f"---------------------------------------------------------------------------------")
        logging.info(f"Skipping format conversion, video already in mp4 format...")

    vid_to_frames(path_to_video_mp4, args.fps) 

    image_list = import_frames_to_list(save_to_dir)

    yolo_detect(args.model_weights, args.batch_size, image_list, save_to_dir, args.extract_timestamp, args.cut, window_size = 60)
    # yolo_detect(args.model_weights, args.detect_tagged_bees, args.batch_size, image_list, save_to_dir, args.extract_timestamp, args.cut, window_size = args.fps)

    frames_to_vid(save_to_dir, args.fps)

    # Remove all extracted images in directory after processing
    os.chdir(f"{save_to_dir}")
    jpg_files = glob.glob('*.jpg')
    for file in jpg_files:
        os.remove(file)
    
    end_time = datetime.datetime.now()
    delta = end_time - start_time

    logging.info(f"\n")
    logging.info(f"---------------------------------------------------------------------------------")
    logging.info(f"Start Time : {start_time}")
    logging.info(f"End Time   : {end_time}")
    logging.info(f"Total Running Time : {(delta.total_seconds()/60):.2f} minutes, {(delta.total_seconds()/3600):.2f} hours")
    logging.info(f"Results saved to {save_to_dir}")
    logging.info(f"---------------------------------------------------------------------------------")

def main(args):
    
    global cuda_device
    cuda_device = torch.device(f'cuda:{args.cuda_device}')
    # cuda_device = torch.device(f'cuda:{2}') #hashtagged this Mohamed

    # args.path_to_video = '/home/katharina/Bee_videos/Plot01/TOP/Plot01_top_2021_04_10_10_00_01.h264' #hashtagged this Mohamed
    # args.path_to_video = '/home/katharina/Bee_videos/Plot01/TOP/Plot01_top_2021_04_12_15_00_01.h264' #hashtagged this Mohamed
    # args.path_to_video = '/home/katharina/Bee_videos/Plot01/TOP/Plot01_top_2021_04_10_10_00_01.h264'
    # args.path_to_video = '/home/katharina/Bee_videos/Plot01/TOP/Plot01_top_2021_04_12_15_00_01.mp4' #hashtagged this Mohamed
    # args.path_to_video = '/home/katharina/Bee_videos/Plot01/TOP/videos_test'

    # Handling multiple videos in a directory
    if os.path.isdir(args.path_to_video):
        videos = os.listdir(args.path_to_video)
        for video in videos:
            video_path = f"{args.path_to_video}/{video}"
            logging.info(f"Processing video {video_path}")
            process_video(video_path)
    elif os.path.isfile(args.path_to_video):
        video_path = args.path_to_video
        logging.info(f"Processing video {video_path}")
        process_video(video_path)
    else:
        print(f"Video {args.path_to_video} not found")

if __name__ == "__main__":
  args = parse_args()
  main(args)

