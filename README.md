This project is an assignment for Deep Learning course. 
The topic of the project is multimodel problems- specifically, visual question answering (VQA). 

In the repo you can find the instructions for the assignment, and a report of what we have done in the project. 


Our proposed model is an ensemble of 3 models: 
1. no pretrained model with 8 CNN layers 
2. pretrained autoEncoder with 4 CNN layers 
3. pretrained autoEncoder with 8 CNN layers 

main.py is reproducing the train of all 3 models. 

evaluate_hw2.py initializes all 3 models, loads the trained model_dicts, creates the dataset and calculates the soft accuracy of the ensemble. 


Note 1: main.py and evaluate_hw2.py are running the entire preprocess on creating and preprocessing the images and texts- it takes some time..
Note 2: for convenience, the saved models are inside the folder 'saved models'. evaluate_hw2.py loads the model from this folder 
