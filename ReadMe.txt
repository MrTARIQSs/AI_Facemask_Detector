Face Mask Detector

By Team #11
-Hussain AlYousef
-Yasser Jaber
-Tariq Alkhamis

2 ways to run the system:
	1-through the included '.exe' file
	2-using the IDE to run the program

Using the IDE to Run the Program:

Recommended Python IDE: PyCharm

Python Version Required: Python Miniconda 3.7

Libraries needed:
-Markdown
-Pillow
-PyYAML
-Werkzeug
-absl-py
-adabelief-tf
-astunparse
-cached-property
-cachetools
-certifi
-chardet
-colorama
-future
-gast
-google-auth
-google-auth-oauthlib
-google-pasta
-grpcio
-h5py
-idna
-importlib-metadata
-imutils
-keras-facenet
-mtcnn
-numpy
-oauthlib
-opencv-python
-opt-einsum
-pip
-protobuf
-pyasn1
-pyasn1-modules
-requests
-requests-oauthlib
-rsa
-scipy
-setuptools
-six
-tabulate
-tensorboard
-tensorboard-plugin-wit
-tensorflow
-tensorflow-addons
-tensorflow-estimator
-termcolor
-typeguard
-urllib3
-wheel
-wincertstore
-wrapt
-zipp

How to install the libraries with PyCharm:
File > Settings > (Expand) Project: ICS381Project > Python Interpreter > '+' icon

Or by terminal
pip install (library name)

How to use the system:
1-shift + F10 (to run the program)
2-The UI will show up with 3 options at the top section
	2.1- Left button will start a live video
		2.1.1- press 'esc' or 'q' to exit the live feed and show the results in the bottom text
	2.2- middle drop-down menu let's you choose between Model 1 (default) and model 2
		2.2.1- Model 1 is the FaceNet pre-trained model (selected by default)
		2.2.2- Model 2 is the model we created and trained from scratch
	2.3- Right button will open the file explorer, from there you can choose the picture to process and search for faces with/without mask
3-The middle section of the UI shows the chosen picture either from the live video or imported using the file explorer
	3.1- on the left is the original image selected
	3.2- on the right is the processed image with a rectangle surrounding the detected faces
		3.2.1- a green rectangle indicates that the detected face is wearing a mask
		3.2.2- a red rectangle indicates that the detected face is not wearing a mask
4-The bottom section displays the number of people in the image above it who are wearing a mask, and the number of people who are not wearing a mask
5-pressing the 'X' exit icon terminates the program