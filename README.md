

# Gender detection using deep learning with keras.
The keras model is created by training  on 1300 face images (~6500 for each class). Face region is cropped by applying `face detection` using `Opencv` on the images gathered from Google Images.Used 6 gender(male,female) after generated with ImageDataGenerator.

**Note: Python 2.x is not supported**
<img src="https://camo.githubusercontent.com/ba2171fe9ab58bba2f169b740c35c26bd3cb4241/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f70796261646765732e737667" alt="versions" data-canonical-src="https://img.shields.io/pypi/pyversions/pybadges.svg" style="max-width:100%;">

## Python packages
* numpy
* tensorflow_gpu
* OpenCv
* PyQt5
* Matplotlib


Install the required packages by executing the following command.

`$ pip install -r requirements.txt`

 





## Usage

### If you want before training model!
`$ python train.py`

### Real-time gender detection(webcam)

`$ python deep.py`


- If you have an Nvidia GPU, then you can install `tensorflow-gpu` package. It will make things run a lot faster.
Depending on the hardware configuration of your system, the execution time will vary. On CPU, training will be slow. Please use onlu tensorflow_gpu  GPU .

<img src="/google/pybadges/raw/master/tests/golden-images/build-failure.svg?sanitize=true" alt="pip installation" style="max-width:100%;">

- Minimal:
- 8gb RAM
- Core i5
- Nvidia


## Help
If you are facing any difficulty, feel free to create a new [issue](https://github.com/RashadGarayev/genderDetectionKeras/issues) or reach out on Facebook [Rashad Garayev](https://www.facebook.com/fly.trion) .
