import os

from process import process, detectors, trackers

if __name__ == "__main__":
    for video in os.listdir("videos"):
        for detector in detectors.keys():
            for tracker in trackers.keys():
                process(video, detector, tracker, True)
