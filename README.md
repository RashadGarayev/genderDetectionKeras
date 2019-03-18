# Gender detection using deep learning with keras.
The keras model is created by training  on 1300 face images (~6500 for each class). Face region is cropped by applying `face detection` using `Opencv` on the images gathered from Google Images. It acheived around 96% training accuracy and ~90% validation accuracy. (20% of the dataset is used for validation)



## Main Python packages
* numpy
* opencv-python
* tensorflow_gpu
* OpenCv
* PyQt5
* Matplotlib


Install the required packages by executing the following command.

`$ pip install -r requirements.txt`

**Note: Python 2.x is not supported** 

Make sure `pip` is linked to Python 3.x  (`pip -V` will display this info).

If `pip` is linked to Python 2.7. Use `pip3` instead. 



## Usage

### If you want before training model!
`$ python train.py`

### Real-time gender detection(webcam)

`$ python deep.py`



## Training
`$ python train.py`




Depending on the hardware configuration of your system, the execution time will vary. On CPU, training will be slow. Please use onlu tensorflow_gpu  GPU .

- Minimal:
- 8gb RAM
- Core i5
- Nvidia

If you have an Nvidia GPU, then you can install `tensorflow-gpu` package. It will make things run a lot faster.

## Help
If you are facing any difficulty, feel free to create a new [issue](https://github.com/RashadGarayev/genderDetectionKeras/issues) or reach out on Facebook [Rashad Garayev](https://www.facebook.com/fly.trion) .
