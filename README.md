[//]: # (Image References)

[image1]: ./images/IM-0115-0001_sample_input.jpeg "Sample input"
[image2]: ./images/AP_PA_orientation.PNG "Orientation"

# Thorax-Pneumonia Classifier Project

## Project Overview
Welcome to this **Deep Learning** project: Chest X-ray images are classified being normal or pneumonia ones by using Convolutional Neural Networks. Means, given a converted .jpeg compressed image of a chest X-ray [DICOM](https://www.dicomstandard.org/) image, the algorithm will identify an estimate of the image status showing a pneumonia or not. 

![Sample input][image1]

The international _Digital Imaging and Communications in Medicine_ standard, DICOM standard for short, delivers the processes and interfaces to transmit, store, retrieve, print, process, and display medical imaging information between relevant modality components of an hospital information system.

The used [Kaggle dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/version/2) delivers already labelled images as training, testing and validation samples. As mentioned, these images are already converted to the .jpeg image format, so, private individual data information sets don't exist.

After viewing such images it has been identified, that only posterior-anterior image orientation is available and that mostly children images are selected. No anterior-posterior or lateral orientation has been found. This could only be analysed more properly by reconverting the images to the .dcm DICOM format having the associated DICOM tags available. Doing this, regulatory data protection aspects have to be taken into account (e.g. Health Insurance Portability and Accountability Act, [HIPAA](https://hipaa.com/)), therefore this has not been done. It would be a HIPAA compliance breach.

![Orientation][image2]


## Project Instructions

### Instructions

1. Download the [chest image dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/version/2). Unzip the folder and place the delivered 'chest_xray' directory in your repository, at location `path/to/chest-classifier-project/data`.

2. **If you are running the project on your local machine (and not using AWS)** create and activate a new environment.
  - __Windows__
  ```
	conda create --name chest-class-project python=3.6
	activate chest-class-project
	pip install -r requirements/requirements.txt
  ```
  
3. **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `chest-class-project` environment. 
```
python -m ipykernel install --user --name chest-class-project --display-name "chest-class-project"
```

4. Open the notebook.
```
jupyter notebook chest-class_app.ipynb
```

## License
This project coding is released under the [MIT](https://github.com/IloBe/CNN_Thorax-Pneumonia_Classifier/blob/master/LICENSE).










