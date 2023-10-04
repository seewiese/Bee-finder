# Bee-finder

The bee-finder aims to simplify the evaluation process of videos. One can input video files in ".264" and ".mp4" format, which the bee-finder will filter for target species (according to the weights provided) and reconstruct a ".mp4" video file out of the frames which contain detected target species. The target group of the bee-finder are users that are not (yet) familiar with tools like command line, programming etc. The provided weights are to detect horned mason bees (<em>O. cornuta</em>) in front of a nesting aid. This manual provides a step-by-step guide how to train one's own YOLO network, as well. 

## Installation and set-up

### Requirements
- Windows Operating System or
- Linux Operating System. <br>

A decent NVIDIA GPU is necessary to use the CNN YOLOv5 on which the bee-finder is based.
<br>

### Installation of neccessary software for training a neural network and the the bee-finder toolbox
<em>Anaconda</em>
<br>Programming tool, neccessary to run YOLO. [Download](https://docs.anaconda.com/anaconda/install/index.html) the Anaconda version which is appropriate for your system. 

  - On Windows: Install Anaconda in directory "C:/anaconda3". A detailed guide with images can be found [here](https://docs.anaconda.com/anaconda/install/windows/).
  - On Linux: Type and confirm
    ```bash
    Anaconda-latest-Linux-x86_64.sh
    ```
    in your Terminal window. A detailed guide can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

<br>

<em>ffmpeg</em>
<br> Neccessary for video processing, e.g. to convert the videos in .h264 format into .mp4 format.

  - On Windows: <br>
        1. [Download](https://www.gyan.dev/ffmpeg/builds/) the ffmpeg version which is appropriate for your system. <br>
        2. Open <em> Control Panel > System > Advanced system settings </em>and click on "Advanced".<br>
        3. Click on "Environment variables and add in "PATH" the pathway with the labelImg-folder, in this case: "C:\labelImg". <br>

  - On Linux: <br>
        1. Launch <em> Anaconda Prompt (Anaconda3) </em> from Start Menu.<br>
        2. Type and confirm: <br>
    ```bash
    conda install pip
    ```
     <br>

    ```bash
    pip install ffmpeg
    ```
    
    <br>

<em>Windows Build Tools C++ 2022</em> 
<br>Some libraries in python are written natively in C++, for which we need this software.
  - On Windows: <br>
        1. [Download](https://visualstudio.microsoft.com/de/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&cid=2030&passive=false) Visual Studio Community 2022. <br>
        2. Install "VisualStudioSetup.exe". After the installer has started, select "Desktop development with C++" and click on "Install" with following parameters:

![Installer](https://github.com/seewiese/bee-finder/assets/141718841/2dd2ae04-9c2d-43b8-8be5-4f033fa8d18a)

  - On Linux: No installation neccesary, required for Windows only.<br><br>

<em>CUDA and cuDNN</em>
<br>
Developed by NVIDIA, CUDA focuses on general computing on GPUs and speeds up various computations. It is required to run deep learning frameworks such as PyTorch (required for YOLOv5). Please note, that a NVIDIA account is neccessary to download files.
- On Windows:<br>
      - Download and install CUDA version 11.7.1 from [CUDA Toolkit 11.7 Update 1](https://developer.NVIDIA.com/cuda-11-7-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) (default settings).<br>
      - Download cuDNN version 8.5.0 for CUDA 11.x from [CUDA Deep Neural Network (cuDNN)](https://developer.NVIDIA.com/cudnn).
        To install CuDNN, one needs to copy following files in the respective CUDA directory (found in the "NVIDIA GPU computing Toolkit" folder): <br><br>
          - bin/cudnn64/8.dll <br>
          - include/cudnn.h <br>
          - lib/cudnn.lib <br> <br>
  :bulb: <strong>TIPP</strong>: If the version listed here cannot be found on the website: all archived versions of cuDNN are found [here](https://developer.NVIDIA.com/rdp/cudnn-archive). 
<br>

- On Linux: No installation neccesary, required for Windows only.<br> <br>

<br>
<em>Visual Studio Code</em> 

<br>

Code editor that will be used to view, change and develop code. The software can be downloaded [here](https://code.visualstudio.com/download). 

  - On Windows: Doubleclick on the downloaded file and follow the provided installer instructions.
  - On Linux: Doubleclick on the downloaded file and click "Install".

<br>

🠊 Restart your computer after those installations have been completed successfully.
<br>
<br>

# Setup of a virtual environment in Anaconda, installation of neccessary packages and the bee-finder
Some libraries and python require a specific version which we need to install in a virtual environment. 

1.   Launch <em> Anaconda Prompt (Anaconda3) </em> from Start Menu.
2.   Create a virtual environment (here called "yolo_env") by typing and running: <br>
      ```bash
      conda create --name yolo_env python=3.8.0
      ```
3.   Activate the created Anaconda environment with: <br>
      ```bash
      conda activate yolo_env
      ```
4.    Install the necessary packages to handle download and installation via "git" and "pip" by typing and confirming:
      ```bash
      conda install git
      ```
      ```bash
      conda install pip
      ```
5.   Navigate into the folder you want to have the bee-finder in, e.g. ```cd 'C:/Users/Max Mustermann/Desktop/'```
6.   Install the bee-finder in this folder by typing and confirming:<br>
     ```bash
     git clone https://github.com/seewiese/bee-finder.git
     ```

7.   Now install all necessary packages for the bee-finder:
      ```bash
      pip install -r requirements_bee_finder.txt
      ```

🠊 The bee-finder has been set up successfully.
 
<br>

### Data organisation for YOLOv5 training
To ease the training process, the following folder structure is recommended to conduct the YOLOv5 training with the provided scripts. Copy all ".jpg" files of the annotated images in the the folder <em> images > Class_0 </em> and all ".txt" files of the annotated images in the folder <em> labels > Class_0 </em>. 
<br><br>
:bulb: <strong> TIPP </strong>: If you have more than one class, add a folder in the "images" and "labels" folder, following the same structure (e.g. Class_1).

<br>

```
├── data
  └── images
      ├── Class_0
          ├── image01.jpg
          ├── image02.jpg
          └── ...
      ├── test
      ├── val
      └── train
   └── labels
      ├── Class_0
          ├── image01.txt
          ├── image02.txt
          └── ...
      ├── test
      ├── val
      └── train
```

<br>

### [Optional] Image augmentation to increase training dataset size
In case you want to increase training dataset size (i.e. create more images for training), you can conduct image augmentation. Recommendation is to have at least 1,500 annotated images per class. The provided code doubles the annotated images each run and the additional images will be randomly rotated, flipped and changed in brightness or size of the original files. The annotation is automatically adapted. Please be aware that only images with annotations will be augmented, background-only images will be skipped. <br> <br>
<strong> Example </strong>: If Class_0 has only 730 annotated images, run image augmentation once on Class_0 to generate additional 730 images.  <br>

1.    Launch <em>Anaconda Prompt (Anaconda3)</em> from Start Menu and enter your virtual environment with <br>

```bash
conda activate yolo_env
```

<br>

2.    Navigate to the directory in which the file "image_augmentation.py" and the folder "data_aug" is found, by copy-pasting the pathway to the file and using the "cd" command, e.g.```cd 'C:/Users/Max Mustermann/Desktop/bee-finder/toolbox/'```

<br>

3.   The function to conduct image augmentation requires following prompts: <br>
<strong>--class_number</strong>: The name of the folder which includes the images and the .txt files (see "Data Organisation for YOLOv5 training"). <br>
<strong>--your_pathway</strong>: The pathway to the folder "toolbox". Please copy and paste your pathway to the file.

<br>

Now you can conduct image augmentation by typing and confirming:<br>
```bash
python image_augmentation.py --class_number="Class_0"  --your_pathway='[system_path_to_folder]/data/'
```

<br>

Your augmented images and annotations will be found in a folder named after your Class number and the suffix "_augmented" (i.e. "Class_0_augmented"). If you are satisfied with the results, move all ".jpg" files and ".txt" files in the <em>Class_0</em> folder and delete the folder <em>Class_0_augmented</em>. <br> 

:bulb: <strong>TIPP</strong>: If you have more than one class, add e.g. "Class_1" folder in each the "images" and "labels" folder and adapt the prompt accordingly (e.g.```--class_number="Class_1"```) to separately augment this class, or combine all classes to one single folder if the dataset is already well-balanced. Please note that images without annotation (i.e. background only images) will not be augmented. 

<br>

🠊 Image augmentation has been completed successfully.

<br>

### Sorting data randomly in the train / validation / test folders

1.    Launch <em>Anaconda Prompt (Anaconda3)</em> from Start Menu and enter your virtual environment with

<br>

```bash
conda activate yolo_env
```

<br>

2.    Navigate to the directory in which the file "split_dataset.py" is found, by copy-pasting the pathway to the function, e.g.```cd 'C:/Users/Max Mustermann/Desktop/bee-finder/toolbox/'```

<br>

3.    Now, you need to copy all images from the folder (in our example "Class_0") to the train (70%), val (20%) and test (10%) folders. For this, the "split_dataset.py" function was created. The original files will stay in the "Class_0" folder and only copies will be created in the train, val and test folders. Following arguments can be adapted depending on your setup: <br>
<strong>--class_number</strong>: The name of the folder which includes the images and the ".txt" files (see "Data Organisation for YOLOv5 training"). If you have more than one class (e.g. "Class_1"), adapt the code by replacing "Class_0" with "Class_1". 
    <br>
<strong>--your_pathway</strong>: The pathway to the folder "toolbox". Please copy and paste your pathway to the file.

<br>
You can apply the function by typing and confirming:

<br>

```bash
python split_dataset.py --class_number="Class_0"  --your_pathway='[system_path_to_folder]/data/'
```

<br>

🠊 Your training dataset for YOLOv5 has been successfully split in training / validation / test images.

<br>

### Train yolov5 with custom data
This step is only needed if you want to train your own custom CNN. In case you only want to use the bee-finder or you already have other "weights", i.e. a ".pt" file you want to use, you can directly jump to the "Using the bee-finder" section.

<br>


To train YOLOv5, you need to adapt the file “training_config.yaml” first.  <br>
1.    Launch <em>Anaconda Prompt (Anaconda3)</em> from Start Menu and enter your virtual environment with

<br>

```bash
conda activate yolo_env
```
<br>

2.    Navigate to the directory in which the file "train.py" is found, by copy-pasting the pathway to the function, e.g. ```cd 'C:/Users/Max Mustermann/Desktop/bee-finder/'```
  
<br>

3.   Open the file "training_config.yaml", which you can find in the directory <em>bee-finder/data</em>, with Visual Studio Code. In there, replace the class names with the classes predefined in the annotation process, e.g. "O.cornuta", "bee", "Class_0" etc.. Save the file to finalize configuration for the YOLOv5 bee-finder. <br> <br>
:bulb: <strong>TIPP</strong>: A quick adaptation in the Anaconda prompt is also possible with a standard software called nano (already installed in base anaconda), so you can adapt the code by navigating to the folder which contains the "training_config.yaml" file via Anaconda prompt and type ```nano training_config.yaml```. Change the pathway, press Ctrl + X and confirm the change with Y. <br>

![grafik](https://github.com/seewiese/bee-finder/assets/141718841/d3a8760a-93a1-41a6-b044-4af5e33db382)


<br>

4.    Before you train your own YOLO network with

<br>

```bash
python train.py --img 1280 --batch 16 --epochs 3 --data “[system_path_to_folder]/training_config.yaml” --weights yolov5x6.pt
```

<br>

adapt following arguments according to your setup:

<br>
<strong>--img</strong>: defines the size of the annotated images. Please make sure that the images correspond to the selected network (see "--weights").
<br>
<strong>--batch</strong>: "batch" specifies how many images are processed at once and depends on the available GPU´s memory capabilities. Processing 16 images simultaneously is relatively high for a single GPU and may produce an error (“out of memory”). If this is the case, lower the number. If one uses many GPUs simultaneously, one can increase the number to e.g. 64 or even higher.
<br>
<strong>--epochs</strong>: "epochs" specifies the total number of iterations of all training data in one training cycle. To find the desired best performance requires a bit trial and error. The higher the number, the longer the training time is. As such, try to increase the epochs e.g. in steps of 100 and find a well-performing model with the lowest number of epochs. <br>
<strong>--weights</strong>: Choose a pretrained network from the ultralytics website (https://github.com/ultralytics/yolov5#pretrained-checkpoints) and specify it here. Those weights will be automatically downloaded in the bee-finder. Please note the specifications to image size. Those should correspond with "--img" and the actual annotated images size. 

<br>
<br>

For every epoch, the box-loss, object loss and class loss will be shown. Also, the mAP@0.5:0.95 will be depicted, which will increase each epoch while YOLO improves its detecting skills by adjusting weights until no improvement is possible any more. If this plateau cannot be found with the epochs you specified within the “train.py” function, repeat the training by increasing the number of epochs. All results of the run will be saved in a results.csv document within <em>bee-finder/runs/train/exp</em>.
<br><br>
The best weights will be saved in the weights folder.
<br>

🠊 YOLOv5 has been trained successfully.

<br>

### Using the bee-finder
The bee-finder (namely the “yolov5_v7.0_modified.py”-file) has several parameters which can be adjusted according to the specific project. These are: <br>
  <strong>  --path_to_video:</strong> type in the path to the input video <br>
<strong>    --model_weights:</strong> type in the path to the trained YOLOv5 model weights <br>
<strong>    --cut:</strong> The bee-finder only saves images in which the target organism is present and merges it to a video. If you want to copy the whole video with target organism highlighted, you can set this parameter to "False". (default = "True") <br>
 <strong>   --fps: </strong> frames per second value used to convert input video to frames (default value=1) <br>
 <strong>   --batch_size:</strong> number of images processed per time (default value = 128). The higher this number, the more memory is required. <br>
<strong>    --cuda_device:</strong> When working with several GPUs, you can switch the working GPU here. Otherwise, you do not need this prompt. <br>

To use the bee-finder, please follow those steps: 

 1.    Launch <em>Anaconda Prompt (Anaconda3)</em> from Start Menu and enter your virtual environment with

<br>

```bash
conda activate yolo_env
```

<br>

2. Navigate to the directory in which the file "yolov5_7.0_modified.py" is found, by copy-pasting the pathway to the prompt, e.g.```cd 'C:/Users/Max Mustermann/Desktop/bee-finder/toolbox/'```

3. Use the bee-finder by typing and confirming the following commannd, while adding necessary parameters from above:
  
 <br>

```bash
python yolov5_7.0_modified.py --path_to_video="<Path_to_videos>/" --model_weights="[system_path_to_folder]/best.pt" --fps=30
```

<br>   
:bulb: <strong>TIPP</strong>:  If you need to extract a timestamp in the generated .csv file, you can utilize the image names (i.e. frame number) in the log-file, as it consists of consecutive numbers by calculating (frame number)/(fps of video)=seconds in the video you can replicate a time stamp for each detection. <br>

:bulb: <strong>TIPP</strong>: If you want to use the bee-finder on videos with 60 fps, an estimate of independent bee flights is calculated by calculating the detections at image margin. By generating the median of how many bees were present on the margin per second, an estimate of independent bee flights in and out of the nesting aid is calculated. Please note that this bee counter only works with 60 fps.
<br>

   
## Troubleshoot
We collected some mistakes that a person can run into. 

 <br>
1. General errors <br>
<br>

<strong>Assertion error: File not found</strong> <br>

![AssertionError](https://github.com/seewiese/bee-finder/assets/141718841/99ad7be9-c956-4dec-b5d8-d272426f5b4a)

<br>

In this case, the wrong separators were used to set the correct path. Windows can operate with both "/" and "\\", but sometimes (and in Linux) it can only operate with "/", otherwise you run into the depicted error. Please also note, that Windows Linux requires a "/" before starting a path whereas Windows does not. e.g. Linux: ```'/C:/Max Mustermann'```; Windows: ```'C:/Max Mustermann'```

<br>

<strong>Set-Location </strong> <br>

![Set-Location](https://github.com/seewiese/bee-finder/assets/141718841/3aadfb63-45f2-488c-9812-db80839010b2)

<br>

This error shows that it cannot recognise the pathway prompted in the menu. The error is caused by the folder which contains a " "[space] and can be easily fixed by wrapping the whole pathway in ' '. The fixed command looks like this: 
```cd 'C:/Users/Max Mustermann/Documents/Random_folder'```

 <br>
2. While training YOLOv5: <br>

<br> 
<strong>torch.cuda.OutOfMemoryError: CUDA is out of memory </strong>

<br>

![grafik](https://github.com/seewiese/bee-finder/assets/141718841/391988d2-1301-4edc-96af-35adcbcafb07)

<br>

This error shows, when your GPU cannot process as many images simultaneously as you specified (i.e. with "--batch"). In this case, decrease the number (e.g. when ```--batch=10```and you receive this error, lower the number with ```--batch=5```), until your GPU can handle it. Please be aware, that the amount of images a GPU can process is not necessarily always the same. E.g. if there is no sufficient cooler for the GPU, it can handle less images simultaneously as a GPU whose temperature is cooled better.

<br>

<strong>IndexError: index 2 is out of bounds for axis 0 with size 1</strong>
<br>

![Index](https://github.com/seewiese/bee-finder/assets/141718841/08d89ba0-1cf8-40dc-b2ee-f429b48f15eb)

<br>
This error shows when there is a problem with the annotated classes. In this case, there were annotations of a class "2", which was not specified in the "training_config.yaml" file. One can easily fix this by either correcting the "training_config.yaml" file when there are more than one classes (see <strong>Train yolov5 with custom data</strong>). The bee-finder only has one class, so that a correction of the file containing the class "2" to class "0" (as we only have "bee" as a class) solved the error.
Before running the training again, make sure to delete the all cache files in the "labels" folder (e.g. train.cache), as otherwise YOLOv5 can show the same error again even though the problem has been solved.

3. While operating the bee-finder

<strong>UnboundLocalError: local variable 'path_to_video_mp4' referenced before assignment </strong>
![grafik](https://github.com/seewiese/bee-finder/assets/141718841/2d3d64b2-22aa-4536-a2db-ffd59165245b)

<br>
This error is produced by having run the bee-finder once already and there are still files in the created directories. Please delete all previous output files of the bee-finder for the videos you want to use the bee-finder on again (i.e. the directory with the same name as the video file) run the bee-finder again. <br> 


## References 
Jocher, G. (2020). YOLOv5 by Ultralytics (Version 7.0) [Computer software]. https://doi.org/10.5281/zenodo.3908559 

<br>

Paperspace (2020). Data Augmentation For Object Detection [Computer software].  https://github.com/Paperspace/DataAugmentationForObjectDetection?ref=blog.paperspace.com


## License

The detector is released under the [GNU General Public License v3.0 license](LICENSE).
