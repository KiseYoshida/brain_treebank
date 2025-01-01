import sys
from PIL import Image
import numpy as np
from pathlib import Path
from glob import glob as glob
import json
import cv2
import os

ROOT="../braintreebank_data"

def check_frames(movie_name, movie_path):
    total_frame_count = 100000
    cap = cv2.VideoCapture(movie_path)
    k = 60
    for frame_number in range(1000, total_frame_count, 10000):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            break  # End of video
        assert frame.dtype == np.uint8
        stimulus_frame = frame[:, :, ::-1]

        stored_frame_path = os.path.join(ROOT, 'movie_frames', movie_name, f"frame_{frame_number:04d}.png")
        stored_frame = np.array(Image.open(stored_frame_path))
        if np.all(stimulus_frame==stored_frame):
            print(f'frame {frame_number} is aligned')
        else:
            print(f'frame {frame_number} is NOT aligned')
    cap.release()

movie_path = sys.argv[2]
movie_name = sys.argv[1]
check_frames(movie_name, movie_path)
