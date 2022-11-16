"""
Low cost Intelligent Tutoring System
"""

import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import pyglet
import cv2
import joblib
from time import time, sleep
import utils_frontend as utf
import utils_get_test_results as uttr
import utils_backend as utb


# process of ongoing cam stream, every 300 frames they are saved in four different folders
def stream(cam_stream_path_1, cam_stream_path_2, cam_stream_path_3, cam_stream_path_4):
    #sleep(20) # wait such that user information is at least half done
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not (cap.isOpened()):
        print("Could not open video device")

    segment = pd.DataFrame([1], columns=['segment'])
    while cap.isOpened():
        # save always the same frame number for each directory to pass for FeatureExtraction.exe
        for i in range(1, 1201):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # face detection and visualisation slows down the performance
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # v2.imshow('frame', frame)
            # if len(faces) > 0: # not used, since very slow with my notebook
            if 1 <= i <= 300:
                cv2.imwrite(cam_stream_path_1 + 'frame' + str(i) + '.jpg', frame)
                if i == 300:
                    segment.to_csv(cam_stream_path_1 + 'segment.csv', index=False)
                    print('1 segment done')
            elif 301 <= i <= 600:
                cv2.imwrite(cam_stream_path_2 + 'frame' + str(i) + '.jpg', frame)
                if i == 600:
                    segment.to_csv(cam_stream_path_2 + 'segment.csv', index=False)
                    print('2 segment done')
            elif 601 <= i <= 900:
                cv2.imwrite(cam_stream_path_3 + 'frame' + str(i) + '.jpg', frame)
                if i == 900:
                    segment.to_csv(cam_stream_path_3 + 'segment.csv', index=False)
                    print('3 segment done')
            elif 901 <= i <= 1200:
                cv2.imwrite(cam_stream_path_4 + 'frame' + str(i) + '.jpg', frame)
                if i == 1200:
                    segment.to_csv(cam_stream_path_4 + 'segment.csv', index=False)
                    print('4 segment done')

        print('------------------------------------------------------------------------------        ', segment.iloc[0][0])
        segment.iloc[0][0] += 1

    cap.release()
    cv2.destroyAllWindows()

########################################################################################################################


# Display the explanation video, check for comprehension problem detections and if needed display hint, otherwise test
def play_vid(features1, features2, features3, features4, segment1, segment2, segment3, segment4):

    # load RandomForest model from trained DAISEE dataset
    filename = 'rf_best_model.sav'
    model = joblib.load(filename)

    # general information about subject
    info, ready = utf.get_user_info()
    user_info = pd.DataFrame([info], columns=['subject', 'gender', 'age', 'education'])
    print('-----------------------------------Now Video started ------------------------------------------------------')
    if ready == 'j':
        # play video (110 sec) and close window after 15 seconds
        vid_path = 'video.mp4'
        window = pyglet.window.Window(fullscreen=True)
        player = pyglet.media.Player()
        # source = pyglet.media.StreamingSource()
        mediaload = pyglet.media.load(vid_path)
        player.queue(mediaload)
        player.play()

        @window.event
        def on_draw():
            window.clear()
            if player.source and player.source.video_format:
                player.get_texture().blit(250, 150)

        def close(event):
            window.close()

        pyglet.clock.schedule_once(close, 125)
        pyglet.app.run()
    print('-----------------------------------------Now Video End ----------------------------------------------------')
    sleep(20)    # wait 20 sec for processing last frames

    print(segment1)
    print(segment2)
    print(segment3)
    print(segment4)

    features = [features1, features2, features3, features4]
    segments = [segment1, segment2, segment3, segment4]
    segments = np.vstack(segments)
    features = np.vstack(features)
    print('Process: playvid() --->   Segments ', segments)
    predictions = model.predict(np.vstack(features))
    print('Predicted Comprehension Problems: -----------------------------------------> ', predictions)

    if 1 in predictions:    # if at least one 10-sec segment entails CP predictions show hint for 30 sec
        # ask user if more help is needed, because of detected "comprehension problems"
        help_request = utf.help_request()
        if help_request == 'j':
            # display hint for 30 seconds
            pic = pyglet.image.load('hint.jpg')
            window = pyglet.window.Window(fullscreen=True)
            sprite = pyglet.sprite.Sprite(pic, x=50, y=30)

            @window.event
            def on_draw():
                window.clear()
                sprite.draw()

            def close(event):
                window.close()

            pyglet.clock.schedule_once(close, 30)
            pyglet.app.run()

    # display test, save data to csv
    predictions = pd.DataFrame(predictions, columns=["predictions"])
    predictions.to_csv(user_info.iloc[0][0] + '_predictions.csv', index=False)
    start_test = time()
    results = uttr.main()
    end_test = time()
    test_results = pd.DataFrame([results], columns=['talo', 'tie', 'kirja', 'katu', 'jalka', 'sopu', 'sota', 'tikka'])
    subject_data = pd.concat([user_info, test_results], axis=1)
    subject_data['time_test'] = [end_test - start_test]
    print(subject_data)
    subject_data.to_csv(subject_data.iloc[0][0] + '_subject.csv', index=False)

########################################################################################################################

def data_processing_1(cam_stream_path_1, cam_stream_path_3, features1, features3, segment1, segment3):
    sleep(13)#31)  # wait 30 seconds such that stream and first 300 frames are saved

    # while video is presented extract features
    while True:
        # frame 1 to 300 // check if first segment has all 300 frames and one segment number, otherwise wait
        while len(os.listdir(cam_stream_path_1)) != 301:
            print('1 : Number of files, when FeatEx -------------> ', len(os.listdir(cam_stream_path_1)))
            sleep(0.5)
        print('1 : Number of files, when FeatEx -------------> ', len(os.listdir(cam_stream_path_1)))
        utb.extract(cam_stream_path_1, cam_stream_path_1)
        data = pd.read_csv(cam_stream_path_1 + 'cam_stream_1.csv')
        segment = pd.read_csv(cam_stream_path_1 + 'segment.csv')
        # delete all files
        for f in os.scandir(cam_stream_path_1):
            os.remove(f)
        data.columns = [col.replace(" ", "") for col in data.columns]
        feature1 = utb.calc_features(data)
        feature1 = utb.preprocess(feature1)  # merge AU1, AU2 and preprocess data (normalisation, log transformation)
        segment1.append(segment.iloc[0][0])
        features1.append(list(feature1.iloc[0].values))
        print('1 segment processing done')

        # frame 601 to 900
        while len(os.listdir(cam_stream_path_3)) != 301:
            print('3 : Number of files, when FeatEx -------------> ', len(os.listdir(cam_stream_path_3)))
            sleep(0.5)
        print('3 : Number of files, when FeatEx -------------> ', len(os.listdir(cam_stream_path_3)))
        utb.extract(cam_stream_path_3, cam_stream_path_3)
        data = pd.read_csv(cam_stream_path_3 + 'cam_stream_3.csv')
        segment = pd.read_csv(cam_stream_path_3 + 'segment.csv')
        # delete all files
        for f in os.scandir(cam_stream_path_3):
            os.remove(f)
        data.columns = [col.replace(" ", "") for col in data.columns]
        feature3 = utb.calc_features(data)   # merge AU1, AU2 and preprocess data (normalisation, log transformation)
        feature3 = utb.preprocess(feature3)
        segment3.append(segment.iloc[0][0])
        features3.append(list(feature3.iloc[0].values))
        print('3 segment processing done')


def data_processing_2(cam_stream_path_2, cam_stream_path_4, features2, features4,segment2, segment4):
    sleep(20)#40)  # wait 40 seconds such that stream and first 300 frames of segment 2 are saved

    # while video is presented extract features
    while True:
        # frame 301 to 600
        while len(os.listdir(cam_stream_path_2)) != 301:
            print('2 : Number of files, when FeatEx -------------> ', len(os.listdir(cam_stream_path_2)))
            sleep(0.5)
        print('2 : Number of files, when FeatEx -------------> ', len(os.listdir(cam_stream_path_2)))
        utb.extract(cam_stream_path_2, cam_stream_path_2)
        data = pd.read_csv(cam_stream_path_2 + 'cam_stream_2.csv')
        segment = pd.read_csv(cam_stream_path_2 + 'segment.csv')
        # delete all files
        for f in os.scandir(cam_stream_path_2):
            os.remove(f)
        data.columns = [col.replace(" ", "") for col in data.columns]
        feature2 = utb.calc_features(data)
        feature2 = utb.preprocess(feature2)
        segment2.append(segment.iloc[0][0])
        features2.append(list(feature2.iloc[0].values))
        print('2 segment processing done')

        # frame 901 to 1200
        while len(os.listdir(cam_stream_path_4)) != 301:
            print('4 : Number of files, when FeatEx -------------> ', len(os.listdir(cam_stream_path_4)))
            sleep(0.5)
        print('4 : Number of files, when FeatEx -------------> ', len(os.listdir(cam_stream_path_4)))
        utb.extract(cam_stream_path_4, cam_stream_path_4)
        data = pd.read_csv(cam_stream_path_4 + 'cam_stream_4.csv')
        segment = pd.read_csv(cam_stream_path_4 + 'segment.csv')
        # delete all files
        for f in os.scandir(cam_stream_path_4):
            os.remove(f)
        data.columns = [col.replace(" ", "") for col in data.columns]
        feature4 = utb.calc_features(data)
        feature4 = utb.preprocess(feature4)
        segment4.append(segment.iloc[0][0])
        features4.append(list(feature4.iloc[0].values))
        print('4 segment processing done')


if __name__ == '__main__':

    # ongoing 300 frames are saved in 4 different folders in order to extract features from them
    # give full path of project
    path = os.path.dirname(os.path.abspath(__file__))

    cam_stream_path_1 = path + '\\cam_stream_1\\'
    cam_stream_path_2 = path + '\\cam_stream_2\\'
    cam_stream_path_3 = path + '\\cam_stream_3\\'
    cam_stream_path_4 = path + '\\cam_stream_4\\'

    if not os.path.isdir(cam_stream_path_1):
        os.mkdir(cam_stream_path_1)
    if not os.path.isdir(cam_stream_path_2):
        os.mkdir(cam_stream_path_2)
    if not os.path.isdir(cam_stream_path_3):
        os.mkdir(cam_stream_path_3)
    if not os.path.isdir(cam_stream_path_4):
        os.mkdir(cam_stream_path_4)

    manager = mp.Manager()

    # shared features variable to have access in play_vid() from data_processing()
    features1 = manager.list()
    features2 = manager.list()
    features3 = manager.list()
    features4 = manager.list()
    segment1 = manager.list()
    segment2 = manager.list()
    segment3 = manager.list()
    segment4 = manager.list()

    # define 3 processes running in parallel
    p1 = mp.Process(target=play_vid,
                    args=[features1, features2, features3, features4, segment1, segment2, segment3, segment4])
    p2 = mp.Process(target=stream,
                    args=[cam_stream_path_1, cam_stream_path_2, cam_stream_path_3, cam_stream_path_4])
    p3 = mp.Process(target=data_processing_1,
                    args=[cam_stream_path_1, cam_stream_path_3, features1, features3, segment1, segment3])
    p4 = mp.Process(target=data_processing_2,
                    args=[cam_stream_path_2, cam_stream_path_4, features2, features4, segment2, segment4])

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
