# TTO

Head trajectory prediction on 360 degree images based on Transformer network. This project is based on a paper which is under review. Anyone cannot use it in any application including paper submission.


# HTRO dataset for training/testing the TTO 

The pre-processed training/testing dataset has been uploaded onto the cloud storage. The link is: https://bhpan.buaa.edu.cn:443/link/7F9B5536B2B20458089D21D920C9FBF6.

Note:

You should ask for the PASSWORD to download it by sending emails to 13021041@buaa.edu.cn. The emial should contain your university/company, your grade (undergraduate, master or doctor) and the purpose of pursuing this dataset. The act of spreading our dataset freely on the Internet is forbidden.

You can download it and put it in a directory in your server and set your own "config.py" for training/testing configuration, according to the "yl360Dataset.py".

The "data_4" is the ground-truth head trajectories of 1,080 ODIs, and you can take it for testing the performance.  

# Implementation
   
1. To train the Transformer network

   python train1.py

2. To test the TTO approach

   python test.py
   
The training dataset is based on our HTRO dataset, you can contact me 13021041@buaa.edu.cn for it; If you have any questions of the code, please contact me.
