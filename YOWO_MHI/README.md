# YOWO -MHI

Add those files into YOWO model for MHI branch

(Subfolder - MHI_Resnet_training: code for training a resnet model on MHI of JHMDB video)

* set the pretrained model and dataset path in the correct yaml file (/YOWO_MHI/cfg/jhmdb.yaml)



## Running the Code

```bash
python main.py --cfg cfg/jhmdb.yaml
```

## Validating the model

```bash
python video_mAP.py --cfg cfg/ucf24.yaml
```








