import os
import sys
import numpy as np

# orig_detection_dir = "jhmdb_orig_detections/detections_0"
orig_detection_dir = "jhmdb_merged_detections/detections_7"
rep_detection_dir = "jhmdb_merged_detections/detections_rep_7"
output_dir = "jhmdb_merged_detections/detections_ensemble_retrain"
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


threshold =  170*170

print(orig_detection_dir, rep_detection_dir)
print("Threshold {}".format(threshold))

orig_files = sorted([f for f in os.listdir(orig_detection_dir) if "rep" not in f])
rep_files = sorted([f for f in os.listdir(rep_detection_dir) if "rep" in f])

def read_f(f):
    cat = []
    conf = []
    boxes = []
    boxes_size = []

    for line in open(f):
        l = line.rstrip("\n").split(" ")
        cat.append(int(l[0]))
        conf.append(float(l[1]))
        boxes.append(l[2:])
        boxes_size.append(bbox_size(l[2:]))
    return cat, conf, boxes, boxes_size

def bbox_size(bbox):
    '''
    bbox: x1, y1, x2, y2
    '''
    return (int(bbox[3]) - int(bbox[1])) * (int(bbox[2]) - int(bbox[0]))

def write_file(cat_list, conf_list, boxes_list, fname):
    
    with open(output_dir+"/"+fname,"w") as f:
        for cat,conf,boxes in zip(cat_list, conf_list, boxes_list):
            f.write(str(cat))
            f.write(" "+str(conf))
            for b in boxes:
                f.write(" "+str(b))
            f.write("\n")

counter = 0
for f_orig, f_rep in zip(orig_files, rep_files):
    cat_orig, conf_orig, boxes_orig, boxes_size_orig = read_f(orig_detection_dir+"/"+f_orig)
    cat_rep, conf_rep, boxes_rep, boxes_size_rep = read_f(rep_detection_dir+"/"+f_rep)

    new_cat = []
    new_conf = []
    new_boxes = []

    if np.mean(boxes_size_orig) > threshold:
        for i in range(min(len(conf_orig), len(conf_rep))):
            if conf_orig[i] > conf_rep[i]:
                new_cat.append(cat_orig[i])
                new_conf.append(conf_orig[i])
                new_boxes.append(boxes_orig[i])
            else:
                new_cat.append(cat_rep[i])
                new_conf.append(conf_rep[i])
                new_boxes.append(boxes_orig[i])

        if len(conf_orig) > len(conf_rep):
            for i in range(len(conf_rep), len(conf_orig)):
                new_cat.append(cat_orig[i])
                new_conf.append(conf_orig[i])
                new_boxes.append(boxes_orig[i])

        write_file(new_cat, new_conf, new_boxes, f_orig)
        counter += 1
    else:
        write_file(cat_orig, conf_orig, boxes_orig, f_orig)

print("Changed {} files".format(counter))