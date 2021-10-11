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