

# How To Use
To run the code, you just need to perform the following steps:
1. Install the requirements:
    ```
    pip install -r requirements.txt
    ```
2. Install this torch version if you are using A100 Cuda Gpus
    ```
    pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```
3 Install sentencepiece library
    ```
    pip install sentencepiece

4. To train or test the system, There are steps modify train.py file and test.py file
<br>
    -> To train and test models in multiple languages, modify test.py and train.py as follows
    <br>
    TO Train
    <br>
    -> In Train.py, in Line 75 "language = "English"", Change the language to either English, Spanish, German,Italian, Dutch, Portuguese or French and run the file to create language model for that particular language.<br>
    -> It is stored in the same directory as the train.py file. The models will be saves as english.pt, spanish.pt, german.pt, italian.pt, french.pt, dutch.pt or portguese.ts <br>
    -> These models can be used to predict<br>
<br>
    To Test<br>
    -> To make predictions on test set using the model that is already saved, modify test.py file<br>
    -> In test.py, Modify lines 89,90,91 "language = "English"<br>
                                          checkpoint = "english.pt"<br>
                                          dataset_type = "all"". Change the language to either English, Spanish, German,Italian, Dutch, Portuguese or French<br>
                                                                 Checkpoint to english.pt, spanish.pt, german.pt, italian.pt, french.pt, dutch.pt or portguese.ts<br>
                                                                 dataset_type to either all, seen, unseen.<br>
    -> Run the test.py file and It will show you the precision, recall and F1-scores of the model on the test-set<br>

    
# You can run final Project.ipynb after making the changes in test.py and train.py.
# By default it will train English Model and makes prediction on all Test-set(Robustness collected Dataset)

# Hyper parameters can be tuned in train.py file.(lr, random seed, batchsize, epochs, patience)
<br>
