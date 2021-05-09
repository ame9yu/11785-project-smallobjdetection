from shutil import copyfile
import os

gt_dir = "/home/ubuntu/project/YOWO/evaluation_ucf24_jhmdb/groundtruths_jhmdb"
detect_dir_1 = "/home/ubuntu/project/YOWO/jhmdb_merged_detections/detections_7"
detect_dir_2 = "/home/ubuntu/project/YOWO/jhmdb_orig_detections/detections_0"
detect_dir_3 = "/home/ubuntu/project/YOWO/jhmdb_merged_detections/detections_ensemble"

target_dir = "/home/ubuntu/project/YOWO/detections/"

if not os.path.isdir(target_dir):
    os.mkdir(target_dir)
if not os.path.isdir(target_dir+"/pretrain_small_50"):
    os.mkdir(target_dir+"/pretrain_small_50")
    os.mkdir(target_dir+"/retrain_small_50")
if not os.path.isdir(target_dir+"/pretrain_small_80"):
    os.mkdir(target_dir+"/pretrain_small_80")
    os.mkdir(target_dir+"/retrain_small_80")
if not os.path.isdir(target_dir+"/pretrain_small_portion"):
    os.mkdir(target_dir+"/pretrain_small_portion")
    os.mkdir(target_dir+"/retrain_small_portion")

try:
    os.mkdir(target_dir+"/ensemble_small_portion")
    os.mkdir(target_dir+"/ensemble_small_50")
    os.mkdir(target_dir+"/ensemble_small_80")
except:
    pass


for ty in ["50","80","portion"]:
    for f in os.listdir(gt_dir+"_"+ty+"_small"):
        # copyfile(detect_dir_1+"/"+f, target_dir+"/retrain_small"+"_"+ty+"/"+f)
        # copyfile(detect_dir_2+"/"+f, target_dir+"/pretrain_small"+"_"+ty+"/"+f)
        copyfile(detect_dir_3+"/"+f, target_dir+"/ensemble_small"+"_"+ty+"/"+f)


    