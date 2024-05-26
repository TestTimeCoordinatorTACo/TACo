import os
from ..bounding_box import BoundingBox
from .enumerators import BBFormat, BBType, CoordinatesType

def get_file_name_only(file_path):
    if file_path is None:
        return ''
    return os.path.splitext(os.path.basename(file_path))[0]

def coco2bb(path, bb_type=BBType.GROUND_TRUTH):
    ret = []

    json_object = path
   
    classes = {}
    if 'categories' in json_object:
        classes = json_object['categories']
        # into dictionary
        classes = {c['id']: c['name'] for c in classes}
    images = {}
    # into dictionary
    for i in json_object['images']:
        images[i['id']] = {
            'file_name': i['file_name'],
            'img_size': (int(i['width']), int(i['height']))
        }
    annotations = []
    if 'annotations' in json_object:
        annotations = json_object['annotations']

    for annotation in annotations:
        img_id = annotation['image_id']
        x1, y1, bb_width, bb_height = annotation['bbox']
        if bb_type == BBType.DETECTED and 'score' not in annotation.keys():
            print('Warning: Confidence not found in the JSON file!')
            return ret
        confidence = annotation['score'] if bb_type == BBType.DETECTED else None
        # Make image name only the filename, without extension
        # img_name = images[img_id]['file_name']
        if bb_type == BBType.DETECTED:
            img_name = images[img_id]['file_name']
        else: 
            img_name = images[img_id]['file_name']
        img_name = get_file_name_only(img_name)
        # create BoundingBox object
        if bb_type == BBType.DETECTED:
            bb = BoundingBox(image_name=img_name,
                                class_id=classes[annotation['category_id']],
                                coordinates=(x1, y1, bb_width, bb_height),
                                type_coordinates=CoordinatesType.ABSOLUTE,
                                img_size=images[img_id]['img_size'],
                                confidence=confidence,
                                bb_type=bb_type,
                                format=BBFormat.XYWH)

        else:
            bb = BoundingBox(image_name=img_name,
                                class_id=classes[annotation['category_id']],
                                coordinates=(x1, y1, bb_width, bb_height),
                                type_coordinates=CoordinatesType.ABSOLUTE,
                                img_size=images[img_id]['img_size'],
                                confidence=confidence,
                                bb_type=bb_type,
                                format=BBFormat.XYWH)



        ret.append(bb)
    return ret
