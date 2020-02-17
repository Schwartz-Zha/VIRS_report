## This is only a small demo about how to use MPII dataset.
[MPII](http://human-pose.mpi-inf.mpg.de/#overview) is a professional huge dataset, used by almost all of the moion tracking models for evauation. The datset (actually only the images, no videos contained) is in .mat format itself, and provides no official python api, which creates a little bit inconvenience.

A full pytorch implementation of customized MPII dataset coube be found [here](https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/dataset/mpii.py), written by Microsoft.

The provided dataset is a, well actually very stupid, implementation. It only does a very basic job, decode the .mat file. 

If you do look into the dataset in detail, you will find that there are various problems about this huge dataset, like missing value, different size images, 0 value in .mat format is not recognized as zero in scipy...

My dummy dataset only takes those figures from the dataset with only one human figure (well, this is not exactly true, because some small figures in the image don't have corresponding annotations.)

I really appreciate any feedback on this.