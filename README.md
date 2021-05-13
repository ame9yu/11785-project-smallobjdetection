# 11785-project-smallobjdetection
Course project for 11785. Improving small object detection via data augmentation
The report can be found on [overleaf](https://www.overleaf.com/project/6046af9fbc7f7343cacd6a38).

## Transformed dataset
All the data and trained weights can be found [here](https://www.dropbox.com/sh/qk5om60vh0wj3et/AADT1mx0QqOIn9JaaAv4aULaa?dl=0).

#### Padded frames
JHMDB dataset images are padded such that the bounding boxes are shrinked to 32 x 32.

#### Motion History Information (MHI)
MHI images are produced from JHMDB dataset. 

#### Ground truths for "small" video

## Data folder structure
The data folder should be organized as follows:
```
merged_jhmdb
|--- labels
|     |--- category
|          |--- video_name
|                |--- frame_id.txt
|--- rgb-images
|     |--- category
|          |--- video_name
|                |--- frame_id.jpg
|--- trainlist.txt
|--- testlist.txt
|--- testlist_video.txt
|--- merged_test_list_video.txt
|--- merged_train_list.txt
|--- merged_test_list.txt
```
`merged_*.txt` contains video folders for all videos, whereas `*txt` without `merged` prefix only contains video folders from the original jhmdb dataset.

## Running various data augmentation methods
### General command
All tasks listed below can be run with three commands. However, to carry out different data augmentation methods, different configuration files (`*.yaml`) should be used.
- Training: 
  - `python main.py --cfg cfg/*.yaml` 
  - Produces detections of each epoch in jhmdb_detections
- Validation: 
  - `python main.py --cfg cfg/*yaml` 
  - Need to change Evaluation term to True in yaml file
- Frame mAP: 
  - `python evaluation_ucf24_jhmdb/pascalvoc.py --gtfolder /path/to/groundtruth_jhmdb_* --detfolder /path/to/detections`
- Video mAP: 
  - `python video_mAP.py --cfg cfg/*.yaml`

### Various tasks
#### 1. Padded frames
- **[[IMPORTANT]]** need to uncomment line 167 of and comment out line 166 in [optimization.py](https://github.com/ame9yu/11785-project-smallobjdetection/blob/main/YOWO/core/optimization.py). Otherwise enlarged frames is also included.
- Configuration file used: `jhmdb_merged.yaml`
#### 2. Enlarged frames for testing
- Configuration file used: `jhmdb_orig.yaml` 
#### 3. MHI
- See instructions [here](https://github.com/ame9yu/11785-project-smallobjdetection/tree/main/YOWO_MHI).

## Acknowledgement
The code base for YOWO is obatained from https://github.com/wei-tim/YOWO
