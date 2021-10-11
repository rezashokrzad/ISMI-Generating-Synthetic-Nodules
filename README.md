# ISMI-Generating-Synthetic-Noduls
Examining how an out-of-thebox detection algorithm can beneﬁt from generated lung nodule data. 

Deep learning is becoming more common in medical imaging systems. With the use of deep learning models, new challenges arise, such as the issue of data scarcity, for the selection of training data. In this research, we examine how an out-of-thebox detection algorithm can beneﬁt from generated lung nodule data. A pipeline was created that produces realistic looking, simulated nodules and the accompanying metadata. The level of similarity to real nodules was tested with the use of detection algorithms. These tests show however that the generated nodules still need to be improved as they, in their current state, harm the performance of detection of real data. Index Terms—Deep learning, CT, Chest X-ray, data augmentation, simulation, nodules



Trunk branch is now the master branch.
The newest version can also be found in the lennart_project branch, but i suggest chosing a more formal name.
Reza will add his newest superimpose function sunday 15 august evening.

Created a convenient jupyter notebook called pipeline.ipynb containing only 3 cells.
This file contains everything needed to generate test code for superimposed nodules in xray generation.
In the 3rd cell of the pipeline function there is a forloop containing all this code.
The samples_count variable decides on how many samples are shown.

All functions are moved to the utils file. When adding or changing functionalities, do it in this file.
This makes it easier to find and edit functions. 
Whenever you make a change in the utils file, you have to restart you kernel and re-run the first cell of the pipeline notebook.
