import os
import shutil

from process import process, detectors, trackers

if __name__ == "__main__":
    for video in os.listdir("videos"):
        for detector in detectors.keys():
            for tracker in trackers.keys():
                process("videos/" + video, detector, tracker, True)
                for file in os.listdir("videos"):
                    if file.endswith("3min.mp4"):
                        continue
                    shutil.move("videos/" + file, "/content/drive/MyDrive/mgr/colab-output/")
