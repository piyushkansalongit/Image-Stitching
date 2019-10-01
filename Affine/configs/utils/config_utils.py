import argparse
import json
import os
import shutil
from bunch import Bunch


def mkdir_if_not_exist(dir_name, is_delete=False):
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(u'[INFO] Dir "%s" exists, deleting.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(u'[INFO] Dir "%s" not exists, creating.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False

def get_config_from_json(json_file):

    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file) 

    config = Bunch(config_dict) 

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.image_path = os.path.join(os.getcwd(), os.path.join("Data", config.filename))
    config.keypoint_path = os.path.join(os.getcwd(), os.path.join(os.path.join("Visuals", config.keypoint_detector), config.filename))
    config.match_path = os.path.join(os.getcwd(), os.path.join("Visuals", os.path.join("correspondances",config.filename)))
    config.affine_path = os.path.join(os.getcwd(), os.path.join(os.path.join("Visuals", "affine"), config.filename))
    # config.keypoint_path = os.path.join(os.getcwd(), os.path.join(os.path.join("Visuals", "keypoint"), config.filename))
    # config.descriptor_path = os.path.join(os.getcwd(), os.path.join(os.path.join("Visuals", "descriptor"), config.filename))
    config.num_matches = os.path.join(os.getcwd(), os.path.join(os.path.join("Visuals", "num_matches"), config.filename))
    config.stitched_path = os.path.join(os.getcwd(), os.path.join(os.path.join("Visuals", "stitched"), config.filename))
    config.blend_path = os.path.join(os.getcwd(), os.path.join(os.path.join("Visuals", "blended"), config.filename))
    config.balanced_path = os.path.join(os.getcwd(), os.path.join(os.path.join("Visuals", "balanced"), config.filename))

    mkdir_if_not_exist(config.keypoint_path)
    mkdir_if_not_exist(config.match_path)
    mkdir_if_not_exist(config.affine_path)
    mkdir_if_not_exist(config.stitched_path)
    mkdir_if_not_exist(config.blend_path)
    mkdir_if_not_exist(config.num_matches)
    mkdir_if_not_exist(config.balanced_path)
    # mkdir_if_not_exist(config.keypoint_path)
    # mkdir_if_not_exist(config.descriptor_path)
    return config




