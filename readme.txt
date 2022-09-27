This is my reimplementation of DeepLung: Wentao Zhu, Chaochun Liu,Wei Fan, Xiaohui Xie, DeepLung: 3D Deep Convolutional Nets for Automated Pulmonary Nodule Detection and Classification, 
arXiv:1709.05538v1

The reimplementation accepts training data in more friendly format that the original DeepLung: CT data and reference segmentations of lung nodules are saved as nifti files (the latter as binary files).
I do not distingish between malignancy scores - I just want to detect all suspected regions as this is what radiologists expect from me.
Besides that, the processing is based on automated lung segmentations.

The preprocessing was implemented specifically for the data format described above.
The first step of processing (detector) is mostly unchanged compared to DeepLung.
The second step of processing is new - I use Densenet to learn classification of detections made in the first step true positive detections/false positive detections
Moreover, the processing is done in 5-fold cross validation scheme with sharp separation of testing and training/validation datasets.

