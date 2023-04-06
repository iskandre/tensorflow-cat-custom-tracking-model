r"""Creates and runs `Estimator` for object detection model on TPUs.
There are two examples here: processing XML and JSON files to annotate images

Check official documentation https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

"""

import json
import os
import re
import time
import warnings

import cv2
import numpy as np
import xmltodict
import tensorflow as tf

import hashlib
import io
import sys
import json
import logging
import os
import contextlib2
import numpy as np
import PIL.Image
import sys

import sys
sys.path.append('models/research')
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util

np.random.seed(0)
ROOT_DIR = 'SOME_PATH'

def list_all_files(root_dir, ext='.xml'):
    """List all files in the root directory and all its sub directories.

    :param root_dir: root directory
    :param ext: filename extension
    :return: list of files
    """
    files = []
    file_list = os.listdir(root_dir)
    for i in range(0, len(file_list)):
        path = os.path.join(root_dir, file_list[i])
        if os.path.isdir(path):
            files.extend(list_all_files(path))
        if os.path.isfile(path):
            if path.lower().endswith(ext):
                files.append(path)
    return files

name2id = {
    'l_eye': 0,
    'r_eye': 1,
    'l_ear': 2,
    'r_ear': 3,
    'nose': 4,
    'tail': 5,
    'l_f_elbow': 6,
    'r_f_elbow': 7,
    'l_b_elbow': 8,
    'r_b_elbow': 9,
    'l_f_knee': 10,
    'r_f_knee': 11,
    'l_b_knee': 12,
    'r_b_knee': 13,
    'l_f_paw': 14,
    'r_f_paw': 15,
    'l_b_paw': 16,
    'r_b_paw': 17,
    'withers': 18
}
name2id_replacement = {
    'l_earbase':'l_ear',
    'r_earbase':'r_ear',
    'tailbase':'tail',
}

def process_keypointname(x):
    x = x.lower()
    if x in name2id_replacement.keys():
        x = name2id_replacement[x]
    if x not in name2id.keys() and x not in ('throat','withers'):
        print('naming error')
    return x

def xml2tfrecords(file_list, img_root, tfrec_output_path, haro_kp_json_list, haro_bbox_json_list, haro_file_list, start_ann_id=0):
    """Save annotations in coco-format.

    :param file_list: list of data annotation files.
    :param img_root: the root dir to load images.
    :param tfrec_output_path: the path to save transformed annotation file.
    :param start_ann_id: the starting point to count the annotation id.
    :param val_num: the number of annotated objects for validation.
    """
    images = []
    annotations = []
    img_ids = []
    ann_ids = []

    ann_id = start_ann_id

    num_shards=1
    cats_count = 0
    with contextlib2.ExitStack() as tf_record_close_stack:
            
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, tfrec_output_path, num_shards)
        filtered_cats_pics = [x.split('.')[0].lower() for x in os.listdir(F'{ROOT_DIR}/mmpose_data/images/VOC2012/JPEGImages_cats')]

        ######### XML example ########
        for idx, file in enumerate(file_list):
            data_anno = xmltodict.parse(open(file).read())['annotation']

            img_id = data_anno['image']
            print('img_id %s'%img_id)
            if re.match('.*\.(jpg|jpeg)$',img_id):
                img_id = img_id.split('.')[0]

            if img_id.lower() in filtered_cats_pics:
                if img_id not in img_ids:
                    image_name = 'VOC2012/JPEGImages/' + img_id + '.jpg'
                    try:
                        _img = cv2.imread(os.path.join(img_root, image_name))
                        old_height = _img.shape[0]
                        old_width = _img.shape[1]
                        img = cv2.resize(_img,(512,512))
                        cv2.imwrite(os.path.join(img_root, 'VOC2012/JPEGImages_512x512/' + img_id + '.jpg'), img)
                        with tf.gfile.GFile(os.path.join(img_root, 'VOC2012/JPEGImages_512x512/' + img_id + '.jpg'), 'rb') as fid:
                            encoded_jpg = fid.read()
                        encoded_jpg_io = io.BytesIO(encoded_jpg)
                        image = PIL.Image.open(encoded_jpg_io)
                        key = hashlib.sha256(encoded_jpg).hexdigest()

                        image = {}
                        image['id'] = img_id
                        image['file_name'] = image_name
                        image['height'] = img.shape[0]
                        image['width'] = img.shape[1]

                        images.append(image)
                        img_ids.append(img_id)
                        keypoint_anno = data_anno['keypoints']['keypoint']
                        num_keypoints = 19
                        assert len(keypoint_anno) == 20
                        keypoints = np.zeros([num_keypoints, 3], dtype=np.float32)
                        visibility_l = [0]*num_keypoints
                        keypoint_names = [b'']*num_keypoints

                        for kpt_anno in keypoint_anno:
                            keypoint_name = kpt_anno['@name']
                            keypoint_name = process_keypointname(keypoint_name)
                            if keypoint_name.lower() in name2id:
                                keypoint_id = name2id[keypoint_name.lower()]
                                keypoint_names[keypoint_id] = bytes(keypoint_name.lower(), encoding='utf-8')

                                visibility = int(kpt_anno['@visible'])
                                if visibility > 0:
                                    visibility = 1
                                visibility_l[keypoint_id] = visibility

                                if visibility > 0:
                                    x = float(kpt_anno['@x'])/old_width
                                    y = float(kpt_anno['@y'])/old_height
                                    if x > 1.0 or y > 1.0:
                                        visibility_l[keypoint_id] = 0
                                    else:
                                        keypoints[keypoint_id, 0] = x
                                        keypoints[keypoint_id, 1] = y
                                        keypoints[keypoint_id, 2] = 2
                
                        anno = {}
                        anno['keypoints'] = keypoints.reshape(-1).tolist()
                        anno['image_id'] = img_id
                        anno['id'] = ann_id
                        anno['num_keypoints'] = int(sum(keypoints[:, 2] > 0))
                        visibility_l = visibility_l[:num_keypoints]

                        visible_bounds = data_anno['visible_bounds']
                        anno['bbox'] = [
                            float(visible_bounds['@xmin']),
                            float(visible_bounds['@ymin']),
                            float(visible_bounds['@width']),
                            float(visible_bounds['@height'])
                        ]
                        anno['iscrowd'] = 0
                        anno['area'] = float(anno['bbox'][2] * anno['bbox'][3])
                        anno['category_id'] = 1

                        annotations.append(anno)
                        ann_ids.append(ann_id)
                        ann_id += 1

                        assert float(visible_bounds['@width']) >= 0
                        assert float(visible_bounds['@height']) >= 0

                        x = float(visible_bounds['@xmin'])
                        y = float(visible_bounds['@ymin'])
                        x1 = x + float(visible_bounds['@width'])
                        y1 = y + float(visible_bounds['@height'])
                        print('xml %s'%file)
                        assert x1 > x
                        assert y1 > y

                        print('img width %s'%image['width'])
                        print('x1 %s'%x1)
                        assert x1 <= image['width']
                        if x1 == image['width'] and x1 > 0:
                            x1 = x1 - 1

                        assert y1 <= image['height']
                        if y1 == image['height'] and y1 > 0:
                            y1 = y1 - 1

                        assert min(x, y, x1, y1) >= 0
                        xmin = float(x) / old_width
                        ymin = float(y) / old_height
                        xmax = float(x1) / old_width
                        ymax = float(y1) / old_height

                        feature_dict = {
                            'image/height':
                                dataset_util.int64_feature(image['height'] ),
                            'image/width':
                                dataset_util.int64_feature(image['width'] ),
                            'image/filename':
                                dataset_util.bytes_feature(file.encode('utf8')),
                            'image/source_id':
                                dataset_util.bytes_feature(str(image_name).encode('utf8')),
                            'image/key/sha256':
                                dataset_util.bytes_feature(key.encode('utf8')),
                            'image/encoded':
                                dataset_util.bytes_feature(encoded_jpg),
                            'image/format':
                                dataset_util.bytes_feature('jpeg'.encode('utf8')),
                            'image/object/bbox/xmin':
                                dataset_util.float_list_feature([xmin]),
                            'image/object/bbox/xmax':
                                dataset_util.float_list_feature([xmax]),
                            'image/object/bbox/ymin':
                                dataset_util.float_list_feature([ymin]),
                            'image/object/bbox/ymax':
                                dataset_util.float_list_feature([ymax]),
                            'image/object/class/text':
                                dataset_util.bytes_list_feature([b'cat']),
                            'image/object/class/label':
                                dataset_util.int64_list_feature([1]),
                        }
                        x_kps = anno['keypoints'][::3]
                        y_kps = anno['keypoints'][1::3]
                        feature_dict['image/object/keypoint/y'] = (
                            dataset_util.float_list_feature( y_kps ))
                        feature_dict['image/object/keypoint/x'] = (
                            dataset_util.float_list_feature( x_kps ))
                        feature_dict['image/object/keypoint/num'] = (
                            dataset_util.int64_list_feature([num_keypoints]))
                        feature_dict['image/object/keypoint/visibility'] = (
                            dataset_util.int64_list_feature(visibility_l))
                        feature_dict['image/object/keypoint/text'] = (
                            dataset_util.bytes_list_feature(keypoint_names))
                        tfrec = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                        shard_idx = idx % num_shards
                        print('idx=%s writing output_tfrecords'%idx)
                        output_tfrecords[shard_idx].write(tfrec.SerializeToString())
                        cats_count += 1

                    except Exception as e:
                        print(e)
                        print('img_id %s error %s'%(img_id,e))
                else:
                    pass
            else:
                pass
        ######### XML example ########

        ######### JSON example ########
        for haro_kp_json, haro_bbox_json in zip(haro_kp_json_list, haro_bbox_json_list):
            with open(haro_kp_json, 'r') as f:
                anno_arr = json.load(f)

            kp_dict = {}
            for idx, anno_d in enumerate(anno_arr):
                img_id = anno_d['data']['img']
                img_id = img_id.split('-')[-1][:-4]
                print('img_id %s'%img_id)

                num_keypoints = 19
                annos = anno_d['annotations'][0]['result']
                keypoints = np.zeros([num_keypoints, 3], dtype=np.float32)
                visibility_l = [0]*19
                keypoint_names = [b'']*19
                for anno in annos:
                    keypoint_name = anno['value']['keypointlabels'][0].lower()
                    keypoint_name = process_keypointname(keypoint_name)
                    if keypoint_name.lower() in name2id:
                        keypoint_id = name2id[keypoint_name.lower()]
                        keypoint_names[keypoint_id] = bytes(keypoint_name.lower(), encoding='utf-8')
                        x = anno['value']['x']
                        y = anno['value']['y']
                        original_width = anno['original_width']
                        original_height = anno['original_height']

                        visibility_l[keypoint_id] = 1
                        x = float(x/100)
                        y = float(y/100)
                        if x > 1.0 or y > 1.0:
                            visibility_l[keypoint_id] = 0
                            keypoints[keypoint_id, 0] = 0.00
                            keypoints[keypoint_id, 1] = 0.00
                            keypoints[keypoint_id, 2] = 0
                        else:
                            keypoints[keypoint_id, 0] = x
                            keypoints[keypoint_id, 1] = y
                            keypoints[keypoint_id, 2] = 1

                kp_dict[img_id] = {'keypoints':keypoints.reshape(-1).tolist(),
                                    'visibility_l':visibility_l,
                                    'keypoint_names':keypoint_names,
                                    'num_keypoints':19}

            with open(haro_bbox_json, 'r') as f:
                bbox_arr = json.load(f)

            for bbox_rec in bbox_arr:
                img_id = bbox_rec['data']['image']
                img_id = img_id.split('-')[-1][:-4]
                if img_id in kp_dict:

                    anno = kp_dict[img_id]
                    try:
                        x = float(bbox_rec['annotations'][0]['result'][0]['value']['x'])/100
                        y = float(bbox_rec['annotations'][0]['result'][0]['value']['y'])/100
                        width = bbox_rec['annotations'][0]['result'][0]['value']['width']
                        height = bbox_rec['annotations'][0]['result'][0]['value']['height']
                    except Exception as err:
                        print(err)

                    img_original_width = bbox_rec['annotations'][0]['result'][0]['original_width']
                    img_original_height = bbox_rec['annotations'][0]['result'][0]['original_height']

                    x1 = x + float(width/100)
                    y1 = y + float(height/100)

                    assert x1 > x
                    assert y1 > y

                    try:
                        assert x1 <= 1
                    except:
                        x1 = 1.00
                    if x1 == 1 and x1 > 0:
                        x1 = x1 - 0.01

                    try:
                        assert y1 <= 1
                    except:
                        y1 = 1.00
                    if y1 == 1 and y1 > 0:
                        y1 = y1 - 0.01

                    assert min(x, y, x1, y1) >= 0

                    xmin = float(x)
                    ymin = float(y)
                    xmax = float(x1)
                    ymax = float(y1)

                    image_name = img_id + '.jpg'

                    _img = cv2.imread(os.path.join(img_root, 'haro', img_id + '.jpg'))
                    old_height = _img.shape[0]
                    old_width = _img.shape[1]
                    img = cv2.resize(_img,(512,512))
                    cv2.imwrite(os.path.join(img_root, 'haro_512x512/' + img_id + '.jpg'), img)

                    with tf.gfile.GFile(os.path.join(img_root, 'haro_512x512/' + img_id + '.jpg'), 'rb') as fid:
                        encoded_jpg = fid.read()
                    encoded_jpg_io = io.BytesIO(encoded_jpg)
                    image = PIL.Image.open(encoded_jpg_io)
                    key = hashlib.sha256(encoded_jpg).hexdigest()

                    feature_dict = {
                            'image/height':
                                dataset_util.int64_feature(512 ),
                            'image/width':
                                dataset_util.int64_feature(512 ),
                            'image/filename':
                                dataset_util.bytes_feature(''.encode('utf8')),
                            'image/source_id':
                                dataset_util.bytes_feature(str(image_name).encode('utf8')),
                            'image/key/sha256':
                                dataset_util.bytes_feature(key.encode('utf8')),
                            'image/encoded':
                                dataset_util.bytes_feature(encoded_jpg),
                            'image/format':
                                dataset_util.bytes_feature('jpeg'.encode('utf8')),
                            'image/object/bbox/xmin':
                                dataset_util.float_list_feature([xmin]),
                            'image/object/bbox/xmax':
                                dataset_util.float_list_feature([xmax]),
                            'image/object/bbox/ymin':
                                dataset_util.float_list_feature([ymin]),
                            'image/object/bbox/ymax':
                                dataset_util.float_list_feature([ymax]),
                            'image/object/class/text':
                                dataset_util.bytes_list_feature([b'cat']),
                            'image/object/class/label':
                                dataset_util.int64_list_feature([1]),
                    }
                    if np.max(anno['keypoints'][::3]) > 1.0:
                        print('keypoints max > 1.0')   
                    if np.max(anno['keypoints'][1::3]) > 1.0:
                        print('keypoints max > 1.0')   
                    x_kps = anno['keypoints'][::3]
                    y_kps = anno['keypoints'][1::3]
                    feature_dict['image/object/keypoint/y'] = (
                        dataset_util.float_list_feature( y_kps ))
                    feature_dict['image/object/keypoint/x'] = (
                        dataset_util.float_list_feature( x_kps))
                    feature_dict['image/object/keypoint/num'] = (
                        dataset_util.int64_list_feature([num_keypoints]))
                    feature_dict['image/object/keypoint/visibility'] = (
                        dataset_util.int64_list_feature(anno['visibility_l']))
                    feature_dict['image/object/keypoint/text'] = (
                        dataset_util.bytes_list_feature(anno['keypoint_names']))
                    tfrec = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                    shard_idx = idx % num_shards
                    print('idx=%s writing output_tfrecords'%idx)
                    output_tfrecords[shard_idx].write(tfrec.SerializeToString())
                    cats_count += 1
            ######### JSON example ########

    print(F'Images processed {cats_count}')


file_list = list_all_files(F'{ROOT_DIR}/mmpose_data/xml')
img_root = F'{ROOT_DIR}/mmpose_data/images'
tfrec_output_path = F'{ROOT_DIR}/mmpose_data/tfrecords'
haro_file_list = list_all_files(F'{ROOT_DIR}/mmpose_data/xml/haro')
haro_kp_json_list = [F'{ROOT_DIR}/mmpose_data/json/haro_keypoints_full.json']
haro_bbox_json_list = [F'{ROOT_DIR}/mmpose_data/json/haro_bbox_full.json']
xml2tfrecords(file_list, img_root, tfrec_output_path, haro_kp_json_list, haro_bbox_json_list, haro_file_list)