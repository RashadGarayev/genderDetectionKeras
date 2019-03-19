# Gender detection using deep learning with keras.
The keras model is created by training  on 1300 face images (~650 for each class). Face region is cropped by applying `face detection` using `Opencv` on the images gathered from Google Images. It acheived around 96% training accuracy and ~90% validation accuracy. (20% of the dataset is used for validation)



## Python packages
* numpy
* tensorflow_gpu
* OpenCv
* PyQt5
* Matplotlib


Install the required packages by executing the following command.

`$ pip install -r requirements.txt`

**Note: Python 2.x is not supported** 



## Usage

### If you want before training model!
`$ python train.py`

### Real-time gender detection(webcam)

`$ python deep.py`
**Note:
Depending on the hardware configuration of your system, the execution time will varry. On CPU, training will be slow. Please use only tensorflow_gpu .
-(https://developer.nvidia.com/rdp/cudnn-archive#a-collapse705-8) CudNN 7.0.5
-(https://developer.nvidia.com/cuda-90-download-archive) Cuda Toolkit 9.0

- Minimal:
- 8gb RAM
- Core i5
- Nvidia

If you have an Nvidia GPU, then you can install `tensorflow-gpu` package. It will make things run a lot faster.

## Training
`$ python train.py`

## Help
If you are facing any difficulty, feel free to create a new [issue](https://github.com/RashadGarayev/genderDetectionKeras/issues) or reach out on Facebook [Rashad Garayev](https://www.facebook.com/fly.trion) .
