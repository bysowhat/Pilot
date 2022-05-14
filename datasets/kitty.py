import os
import csv
import logging
import random
import numpy as np

from .base import Dataset

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
}

class Kitti(Dataset):

  def __init__(self, cfg, phase):
    self.root = cfg.DATASETS.ROOT
    self.phase = phase
    self.classes = cfg.DATASETS.DETECT_CLASSES

    self.get_file_list()

    self.input_width = cfg.INPUT.WIDTH_TRAIN
    self.input_height = cfg.INPUT.HEIGHT_TRAIN
    self.output_width = self.input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
    self.output_height = self.input_height // cfg.MODEL.BACKBONE.DOWN_RATIO
    self.max_objs = cfg.DATASETS.MAX_OBJECTS

    self.logger = logging.getLogger(__name__)
    self.logger.info("Initializing KITTI {} set with {} files loaded".format(self.phase, self.num_samples))

    print(1)

  def get_file_list(self):
    if self.phase == "train":
      imageset_txt = os.path.join(self.root, "ImageSets", "train.txt")
    elif self.phase == "val":
      imageset_txt = os.path.join(self.root, "ImageSets", "val.txt")
    elif self.phase == "trainval":
      imageset_txt = os.path.join(self.root, "ImageSets", "trainval.txt")
    elif self.phase == "test":
      imageset_txt = os.path.join(self.root, "ImageSets", "test.txt")
    else:
      raise ValueError("Invalid split!")

    if self.phase in ("train", "val", "trainval"):
      self.image_dir = os.path.join(self.root, "training", "image_2")
      self.val_dir = os.path.join(self.root, "training", "velodyne")
      self.label_dir = os.path.join(self.root, "training", "label_2")
      self.calib_dir = os.path.join(self.root, "training", "calib")
    elif self.phase == 'test':
      self.image_dir = os.path.join(self.root, "testing", "image_2")
      self.val_dir = os.path.join(self.root, "testing", "velodyne")
      self.calib_dir = os.path.join(self.root, "testing", "calib")
    else:
      raise ValueError('Unavaliable dataset phase')

    image_files = []
    for line in open(imageset_txt, "r"):
      base_name = line.replace("\n", "")
      image_name = base_name + ".png"
      image_files.append(image_name)
    self.image_files = image_files
    self.label_files = [i.replace(".png", ".txt") for i in self.image_files]
    self.lidar_files = [i.replace(".png", ".bin") for i in self.image_files]
    self.num_samples = len(self.image_files)

  def load_annotations(self, idx):
    annotations = []
    file_name = self.label_files[idx]
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                  'dl', 'lx', 'ly', 'lz', 'ry']

    if self.phase in ("train", "val", "trainval"):
      with open(os.path.join(self.label_dir, file_name), 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

        for line, row in enumerate(reader):
          if row["type"] in self.classes:
            annotations.append({
              "class": row["type"],
              "label": TYPE_ID_CONVERSION[row["type"]],
              "truncation": float(row["truncated"]),
              "occlusion": float(row["occluded"]),
              "alpha": float(row["alpha"]),
              "dimensions": [float(row['dl']), float(row['dh']), float(row['dw'])],
              "locations": [float(row['lx']), float(row['ly']), float(row['lz'])],
              "rot_y": float(row["ry"])
            })

    # get camera intrinsic matrix K
    with open(os.path.join(self.calib_dir, file_name), 'r') as csv_file:
      reader = csv.reader(csv_file, delimiter=' ')
      for line, row in enumerate(reader):
        if row[0] == 'P2:':
          K = row[1:]
          K = [float(i) for i in K]
          K = np.array(K, dtype=np.float32).reshape(3, 4)
          K = K[:3, :3]
          break

    return annotations, K

  def __iter__(self):
    pass

  def __len__(self):
    return self.num_samples