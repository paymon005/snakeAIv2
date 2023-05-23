import os
import math


def find_file_in_folder(folder, filename):  # can input a filename with or without an extension
    file_split = filename.split('.')
    ext = None
    if len(file_split) > 1:
        ext = file_split[-1]
    for (dir_path, dir_names, file_names) in os.walk(folder):
        for file in file_names:
            if ext is None:
                if filename == file.split('.')[0]:
                    return dir_path + '\\' + filename
            else:
                if filename == file:
                    return dir_path + '\\' + filename
    return None


def create_save_dir(top_folder, bot_folder):
    save_dir = os.path.join(os.getcwd(), top_folder, bot_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def calc_distance(pt1, pt2):
    x2 = (pt1[0] - pt2[0]) ** 2
    y2 = (pt1[1] - pt2[1]) ** 2
    return math.sqrt(x2 + y2)


def get_normalized_angle(x1, y1, x2, y2):
    angle = 0
    if y1 < y2:
        if x1 < x2:
            angle = abs(math.degrees(math.atan((y2 - y1) / (x2 - x1))))
        elif x1 > x2:
            angle = abs(math.degrees(math.atan((y2 - y1) / (x2 - x1)))) + 90
        else:
            angle = 90
    elif y1 > y2:
        if x1 < x2:
            angle = -abs(math.degrees(math.atan((y2 - y1) / (x2 - x1))))
        elif x1 > x2:
            angle = -abs(math.degrees(math.atan((y2 - y1) / (x2 - x1)))) - 90
        else:
            angle = -90
    else:
        if x1 <= x2:
            angle = 0
        elif x1 > x2:
            angle = -180
    return angle / 180


def get_newest_file_in_folder_w_ext(folder, ext):
    files = []
    for (dir_path, dir_names, file_names) in os.walk(folder):
        for file in file_names:
            if file.endswith(ext):
                files.append(dir_path + '\\' + file)
    files = sorted(files, key=os.path.getctime, reverse=True)
    for file in files:
        if file.endswith('.' + ext):
            return file
    return None


def is_not_zero(value):
    if value == 0:
        return False
    return True
