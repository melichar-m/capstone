# capstone
capstone project from gcu undergrad


In order to use this optical character recognition system, understand a few things first: 

1) This is a work in progress. I have not finished it yet, and am actively developing toward meeting the end of training and GUI elements to show the user what the OCR is seeing in real-time.
2) All of what has been uploaded is my current, live version. It may not be working at the exact moment you're looking at it if I've broken something.
3) Since everything I have is being shared, all images are my live training set of data. Feel free to download it, play with it, and share with me if you find something interesting or know how to fix whatever I'm pulling my hair out over.

That being said, how can you use my OCR? Great question.

To use:

If on a colab/jupyter notebook:
Copy and paste the code from dataset.py, then crnn.py, and finally training.py into separate sequential code blocks. 
Upload the CSV file to the root directory you're running the notebook from.
Create a folder and name it "training". Upload all of the 800 images inside of the images.zip folder into this training folder.
Run dataset.py, crnn.py, then training.py. You can adjust how many epochs of training the model undergoes with the variable toward the top of training.py aptly named "epochs".

If you want to run them on your own PC or a virtual machine:
Your system must have python, as well as all required libraries installed.
Download all files, and unzip the folder labeled "images" into the directory that has all of the other python files that will be run. 
The extracted folder should be renamed to "training". 
With all of the python files, the CSV, and the training folder full of 800 images of text, you should be ready to start running the python files.
Run them in this order: dataset.py -> crnn.py -> training.py.
