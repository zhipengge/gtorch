# -*- coding: utf-8 -*-
"""
@author: gehipeng @ 20230614
@file: utils.py
@brief: utils
"""
import time
import os
import json


def get_current_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def get_current_timestamp_ms():
    return int(round(time.time() * 1000))

def change_folder_to_file_tree(dir_map, file_tree, root):
    root_name = os.path.basename(root)
    file_tree[root_name] = dir_map[root]
    for i, dir_ in enumerate(dir_map[root]["dirs"]):
        dir_path = os.path.join(root, dir_)
        if dir_path in dir_map:
            file_tree[root_name]["dirs"][i] = {dir_: {}}
            change_folder_to_file_tree(dir_map, file_tree[root_name]["dirs"][i], dir_path)

def get_file_tree(root_dir):
    """
    :param root_dir: root dir, str, absolute path
    :return: file tree, dict
    """
    base_dir = os.path.dirname(root_dir)
    tmp_map = {}
    for root, dirs, files in os.walk(root_dir):
        files = [file for file in files if file != ".DS_Store"]
        rel_root = os.path.relpath(root, base_dir)
        tmp_map[rel_root] = {"dirs": sorted(dirs), "files": sorted(files)}
    file_tree = {}
    root = os.path.relpath(root_dir, base_dir)
    change_folder_to_file_tree(tmp_map, file_tree, root)
    return file_tree

def print_dict(dct, indent=4):
    print(json.dumps(dct, sort_keys=False, indent=indent))

if __name__ == "__main__":
    # print(get_current_time_str())
    # root = "../"
    # file_tree = get_file_tree(root)
    # print_dict(file_tree)
    lst = [1, 2, 3, 4, 5]
    lst = set(lst)
    print_dict(lst)