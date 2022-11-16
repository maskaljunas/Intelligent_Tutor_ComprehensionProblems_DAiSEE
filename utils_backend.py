"""
Extracting facial features with OpenFace via Feature Extraction.exe.
For Windows, OpenFace must be installed in the python.exe directory as instructed in the wiki.
https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation
"""
import subprocess
import os
import sys
import pandas as pd
import numpy as np


def extract(data_path, output_path):
    python_path = os.path.dirname(os.path.abspath(sys.executable))
    openface_path = os.path.abspath(os.path.join(python_path, 'openface'))

    data_path = data_path[:-1]
    output_path = output_path[:-1]
    # extract only pose, AUs, gaze related information
    args = 'FeatureExtraction.exe -fdir "{data_path}" -pose -aus -gaze -out_dir "{output}"'.format(
        data_path=data_path, output=output_path)

    # call command line to start openface - FeatureExtraction.exe and supress printed output
    subprocess.run(args, shell=True,
                   cwd=openface_path, stdout=subprocess.DEVNULL)

'----------------------------------------------------------------------------------------------------------------------'
"""
Calculating selected features for the Intelligent Tutoring System based on the FeatureExtraction data.
"""

# ACTION UNITS only the ones needed for the model
# calculate average AU presence in 10 sec video for feature selection
def calc_action_units(dataframe):

    # action_units_name = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c',
    # 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
    action_units_name = ['AU01_c', 'AU02_c', 'AU04_c', 'AU12_c', 'AU20_c', 'AU25_c', 'AU28_c']

    action_units_orig = dataframe[action_units_name]
    action_units_mean = pd.DataFrame(action_units_orig.mean(axis=0)).transpose()
    action_units_mean = action_units_mean.add_suffix('_mean')

    action_units = action_units_mean

    return action_units

'----------------------------------------------------------------------------------------------------------------------'

# GAZE ANGLE in radians for left-right and up-down movements
def calc_gaze_angle(dataframe):

    gaze_name = ['gaze_angle_x', 'gaze_angle_y']
    gaze_angle_xy = dataframe[gaze_name]

    gaze_angle_x_diff = [list(gaze_angle_xy.iloc[:, 0])[n] - list(gaze_angle_xy.iloc[:, 0])[n - 1] for n in
                         range(1, len(list(gaze_angle_xy.iloc[:, 0])))]
    gaze_angle_y_diff = [list(gaze_angle_xy.iloc[:, 1])[n] - list(gaze_angle_xy.iloc[:, 1])[n - 1] for n in
                         range(1, len(list(gaze_angle_xy.iloc[:, 1])))]

    gaze_angle_x_diff_std = pd.DataFrame([np.std(gaze_angle_x_diff)], columns=['gaze_angle_x_diff_std'])
    gaze_angle_y_diff_std = pd.DataFrame([np.std(gaze_angle_y_diff)], columns=['gaze_angle_y_diff_std'])

    gaze_fixation_count_x = gaze_angle_x_diff.count(0)
    gaze_fixation_count_y = gaze_angle_y_diff.count(0)
    gaze_fixation_count = pd.DataFrame([(gaze_fixation_count_x + gaze_fixation_count_y) / 2],
                                       columns=['gaze_fixation_count'])

    gaze = pd.concat([gaze_angle_x_diff_std, gaze_angle_y_diff_std, gaze_fixation_count], axis=1)

    return gaze


'----------------------------------------------------------------------------------------------------------------------'


# HEAD DISTANCE with respect to the camera in mm
def calc_head_distance(dataframe):

    # distance between camera and face (+z away from camera)
    head_distance_z = dataframe['pose_Tz']

    # max head distance from avg away from camera
    max_away_head_distance = pd.DataFrame([head_distance_z.max() - head_distance_z.mean()], columns=['pose_Tz_max_away'])

    head_distance = max_away_head_distance

    return head_distance


'----------------------------------------------------------------------------------------------------------------------'


# HEAD POSE
def calc_head_pose(dataframe):

    # left(-)/right(+) head pose
    yaw_rot = dataframe['pose_Ry']

    # check if left and right orientation exist
    # the higher the mean of left or right rotation values the longer their head was rotation into a specific direction
    left_rot = [x for x in yaw_rot if x <= 0]  # negative values
    right_rot = [x for x in yaw_rot if x >= 0]  # positive values

    if not left_rot:
        left_rot_avg = 0
        left_rot_max = 0
    else:
        left_rot_avg = np.mean(left_rot)
        left_rot_max = min(left_rot)

    if not right_rot:
        right_rot_mean = 0
    else:
        right_rot_mean = np.mean(right_rot)

    head_pose_list = [[left_rot_avg, left_rot_max, right_rot_mean]]
    head_pose = pd.DataFrame(head_pose_list, columns=['left_rot_avg', 'left_rot_max', 'right_rot_mean'])

    return head_pose


'----------------------------------------------------------------------------------------------------------------------'


# AU45 - blinks
# frame length of blinks=1 and frame interval between blinks
def get_frame_interval_len(frame_diff):
    inter_interval = []
    frame_len = []
    length = 0
    for item in frame_diff:
        if item == 1:
            length += 1
        else:
            frame_len.append(length)
            inter_interval.append(item)
            # blink_len = 0
    frame_len.append(length)
    return frame_len, inter_interval


# count how many blinks: frame blink length >=3 frames and frame interval between blinks >=5
# a blink last for at least 100 ms --> 3 frames// eyes open between blinks at around 150 ms --> 5 frames
def get_counts(frame_len, inter_interval):
    i = 0
    j = 0
    counts = 0
    for i in range(0, len(frame_len)):
        if i == len(frame_len) - 1:
            if frame_len[i] >= 5:
                counts += 1
        else:
            if frame_len[i] >= 3 and inter_interval[j] >= 5:
                counts += 1
                i += 1
                j += 1
            else:
                continue
    return counts


def get_blinks(dataframe):

    # only frame number of blinks==1
    all_blinks_frame = dataframe.loc[dataframe['AU45_c'] == 1, 'frame']

    # for every frame a blink is stated as 0 or 1, all frame numbers where a blink=1 are saved in order to calculate how
    # long and how many blinks exist (consecutive frame numbers = ongoing blink, long difference between frames = open)
    all_blinks_frame_diff = [list(all_blinks_frame)[n] - list(all_blinks_frame)[n - 1] for n in
                             range(1, len(list(all_blinks_frame)))]

    blink_frame_len, inter_blink_interval = get_frame_interval_len(all_blinks_frame_diff)

    if not blink_frame_len:
        blink_frame_len = [0]
    if not inter_blink_interval:
        inter_blink_interval = [0]

    blinks = pd.DataFrame([get_counts(blink_frame_len, inter_blink_interval)], columns=['blinks'])

    blink_stuff = blinks

    return blink_stuff


'----------------------------------------------------------------------------------------------------------------------'


def calc_features(dataframe):
    action_units = calc_action_units(dataframe)
    gaze = calc_gaze_angle(dataframe)
    blink_stuff = get_blinks(dataframe)  # exclude since too many outliers

    head_distance = calc_head_distance(dataframe)
    head_pose = calc_head_pose(dataframe)

    feature_frame = pd.concat([action_units, gaze, blink_stuff, head_distance, head_pose], sort=False, axis=1)

    return feature_frame

'----------------------------------------------------------------------------------------------------------------------'


def preprocess(dataframe):

    # Merge AU1 and AU2
    dataframe['AU01+2_c_mean'] = dataframe['AU01_c_mean'] + dataframe['AU02_c_mean']
    dataframe['AU01+2_c_mean'] = (dataframe['AU01+2_c_mean'] / 2)
    dataframe = dataframe.drop(columns=['AU01_c_mean', 'AU02_c_mean'])

    # reorder
    cols = list(dataframe.columns)
    cols = [cols[-1]] + cols[0:-1]
    dataframe = dataframe[cols]

    # Gaze stuff
    gaze = ['gaze_angle_x_diff_std', 'gaze_angle_y_diff_std']
    for fcol in gaze:
        # log transform
        dataframe[fcol] = (dataframe[fcol] + 1).transform(np.log)

    # Blinks
    blinks = ['gaze_fixation_count']
    for fcol in blinks:

        # log transform
        dataframe[fcol] = (dataframe[fcol] + 1).transform(np.log)

    # Head pose
    head = ['pose_Tz_max_away', 'left_rot_avg', 'left_rot_max', 'right_rot_mean']
    for fcol in head:

        if fcol in ['left_rot_avg', 'left_rot_max']:
            dataframe[fcol] = (dataframe[fcol] * (-1))

        # log transform
        dataframe[fcol] = (dataframe[fcol] + 1).transform(np.log)

    return dataframe
