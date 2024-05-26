# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging
from detectron2.data.datasets import register_coco_instances

logger = logging.getLogger(__name__)

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )

def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


_root = os.getenv("DETECTRON2_DATASETS", "/data/js_acdc")
register_coco_unlabel(_root)
register_coco_instances("cityscapes_train", {}, os.path.join(_root, "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_val", {}, os.path.join(_root, "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"), os.path.join(_root, "cityscapes/leftImg8bit/val"))

register_coco_instances("acdc_fog_train", {}, os.path.join(_root, "ACDC/annotations/fog/instancesonly_fog_train_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_night_train", {}, os.path.join(_root, "ACDC/annotations/night/instancesonly_night_train_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_rain_train", {}, os.path.join(_root, "ACDC/annotations/rain/instancesonly_rain_train_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_snow_train", {}, os.path.join(_root, "ACDC/annotations/snow/instancesonly_snow_train_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))

register_coco_instances("acdc_fog_val", {}, os.path.join(_root, "ACDC/annotations/fog/instancesonly_fog_val_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_night_val", {}, os.path.join(_root, "ACDC/annotations/night/instancesonly_night_val_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_rain_val", {}, os.path.join(_root, "ACDC/annotations/rain/instancesonly_rain_val_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_snow_val", {}, os.path.join(_root, "ACDC/annotations/snow/instancesonly_snow_val_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))

register_coco_instances("acdc_fog", {}, os.path.join(_root, "ACDC/annotations/instancesonly_fog_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_night", {}, os.path.join(_root, "ACDC/annotations/instancesonly_night_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_rain", {}, os.path.join(_root, "ACDC/annotations/instancesonly_rain_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_snow", {}, os.path.join(_root, "ACDC/annotations/instancesonly_snow_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))

register_coco_instances("acdc", {}, os.path.join(_root, "ACDC/annotations/instancesonly_all_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))

register_coco_instances("acdc_whole", {}, os.path.join(_root, "ACDC/annotations/instancesonly_whole_gt_detection.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_whole_noaug", {}, os.path.join(_root, "ACDC/annotations/instancesonly_whole_gt_detection_noaug.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_whole_noaug_irregular", {}, os.path.join(_root, "ACDC/annotations/instancesonly_whole_gt_detection_noaug_irregular.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test", {}, os.path.join(_root, "ACDC/annotations/test.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_nobackprop", {}, os.path.join(_root, "ACDC/annotations/test_nobackprop.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_nobackprop_filter", {}, os.path.join(_root, "ACDC/annotations/test_nobackprop_filter.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_batch128_bn", {}, os.path.join(_root, "ACDC/annotations/test_batch128_bn.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_1", {}, os.path.join(_root, "ACDC/annotations/test_1.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_2", {}, os.path.join(_root, "ACDC/annotations/test_2.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_3", {}, os.path.join(_root, "ACDC/annotations/test_3.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_4", {}, os.path.join(_root, "ACDC/annotations/test_4.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_5", {}, os.path.join(_root, "ACDC/annotations/test_5.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_6", {}, os.path.join(_root, "ACDC/annotations/test_6.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_7", {}, os.path.join(_root, "ACDC/annotations/test_7.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_8", {}, os.path.join(_root, "ACDC/annotations/test_8.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_9", {}, os.path.join(_root, "ACDC/annotations/test_9.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_10", {}, os.path.join(_root, "ACDC/annotations/test_10.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_11", {}, os.path.join(_root, "ACDC/annotations/test_11.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_12", {}, os.path.join(_root, "ACDC/annotations/test_12.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_13", {}, os.path.join(_root, "ACDC/annotations/test_13.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_14", {}, os.path.join(_root, "ACDC/annotations/test_14.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_15", {}, os.path.join(_root, "ACDC/annotations/test_15.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_16", {}, os.path.join(_root, "ACDC/annotations/test_16.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_17", {}, os.path.join(_root, "ACDC/annotations/test_17.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_18", {}, os.path.join(_root, "ACDC/annotations/test_18.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_19", {}, os.path.join(_root, "ACDC/annotations/test_19.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_20", {}, os.path.join(_root, "ACDC/annotations/test_20.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_21", {}, os.path.join(_root, "ACDC/annotations/test_21.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_22", {}, os.path.join(_root, "ACDC/annotations/test_22.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_23", {}, os.path.join(_root, "ACDC/annotations/test_23.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_24", {}, os.path.join(_root, "ACDC/annotations/test_24.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_25", {}, os.path.join(_root, "ACDC/annotations/test_25.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_26", {}, os.path.join(_root, "ACDC/annotations/test_26.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_27", {}, os.path.join(_root, "ACDC/annotations/test_27.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_28", {}, os.path.join(_root, "ACDC/annotations/test_28.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_29", {}, os.path.join(_root, "ACDC/annotations/test_29.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_30", {}, os.path.join(_root, "ACDC/annotations/test_30.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_31", {}, os.path.join(_root, "ACDC/annotations/test_31.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_32", {}, os.path.join(_root, "ACDC/annotations/test_32.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_33", {}, os.path.join(_root, "ACDC/annotations/test_33.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_34", {}, os.path.join(_root, "ACDC/annotations/test_34.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_35", {}, os.path.join(_root, "ACDC/annotations/test_35.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_36", {}, os.path.join(_root, "ACDC/annotations/test_36.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_test_37", {}, os.path.join(_root, "ACDC/annotations/test_37.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))

register_coco_instances("acdc_noise_80", {}, os.path.join(_root, "ACDC/annotations/noise_injected_80.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_noise_20", {}, os.path.join(_root, "ACDC/annotations/noise_injected_20.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))
register_coco_instances("acdc_noise_test", {}, os.path.join(_root, "ACDC/annotations/origin_sequence4.json"), os.path.join(_root, "ACDC/rgb_anon_trainvaltest/rgb_anon"))


register_coco_instances("cityscapes_fog", {}, os.path.join(_root, "cityscapes/fog_train_added.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_clear", {}, os.path.join(_root, "cityscapes/clear_train_added.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_rain", {}, os.path.join(_root, "cityscapes/rain_train_added.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_snow", {}, os.path.join(_root, "cityscapes/snow_train_added.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))

register_coco_instances("cityscapes_whole_added", {}, os.path.join(_root, "cityscapes/whole_train.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_1", {}, os.path.join(_root, "cityscapes/test_1.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_2", {}, os.path.join(_root, "cityscapes/test_2.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_3", {}, os.path.join(_root, "cityscapes/test_3.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_4", {}, os.path.join(_root, "cityscapes/test_4.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_5", {}, os.path.join(_root, "cityscapes/test_5.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_6", {}, os.path.join(_root, "cityscapes/test_6.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_7", {}, os.path.join(_root, "cityscapes/test_7.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_8", {}, os.path.join(_root, "cityscapes/test_8.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_9", {}, os.path.join(_root, "cityscapes/test_9.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_10", {}, os.path.join(_root, "cityscapes/test_10.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_11", {}, os.path.join(_root, "cityscapes/test_11.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_12", {}, os.path.join(_root, "cityscapes/test_12.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_13", {}, os.path.join(_root, "cityscapes/test_13.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_14", {}, os.path.join(_root, "cityscapes/test_14.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_15", {}, os.path.join(_root, "cityscapes/test_15.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_16", {}, os.path.join(_root, "cityscapes/test_16.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))
register_coco_instances("cityscapes_test_17", {}, os.path.join(_root, "cityscapes/test_17.json"), os.path.join(_root, "cityscapes/leftImg8bit/train"))


register_coco_instances("kitti_fog", {}, os.path.join(_root, "kitti/fog_train_cutted.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_clear", {}, os.path.join(_root, "kitti/clear_train_cutted.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_rain", {}, os.path.join(_root, "kitti/rain_train_cutted.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_snow", {}, os.path.join(_root, "kitti/snow_train_cutted.json"), os.path.join(_root, "kitti/data_object/training"))

register_coco_instances("kitti_whole_added", {}, os.path.join(_root, "kitti/whole_train.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_test_1", {}, os.path.join(_root, "kitti/test_1.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_test_2", {}, os.path.join(_root, "kitti/test_2.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_test_3", {}, os.path.join(_root, "kitti/test_3.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_test_4", {}, os.path.join(_root, "kitti/test_4.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_test_5", {}, os.path.join(_root, "kitti/test_5.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_test_6", {}, os.path.join(_root, "kitti/test_6.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_test_7", {}, os.path.join(_root, "kitti/test_7.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_test_8", {}, os.path.join(_root, "kitti/test_8.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_test_9", {}, os.path.join(_root, "kitti/test_9.json"), os.path.join(_root, "kitti/data_object/training"))
register_coco_instances("kitti_test_10", {}, os.path.join(_root, "kitti/test_10.json"), os.path.join(_root, "kitti/data_object/training"))
