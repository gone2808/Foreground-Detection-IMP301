from calendar import c
import pafy
import cv2
import os
import math
import time

DOWNLOAD_FOLDER = 'Data'    # download folder
SAVE_VIDEO = True           # set to False to save single frames
FPS = 30
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')
VIDEO_CHUNKS_FRAMES = 1*(FPS*60)   # save video at chunks of frames (30 min @ 25 fps)
MAX_CHUNKS_TO_SAVE = -1            # total number of chunks to save, -1 for no limit

# url = '1EiC9bvVGnk'
# url = "https://www.youtube.com/watch?v=1EiC9bvVGnk" #jackson hole wyoming

print('\nInsert youtube video URL:')
url = input()
video = pafy.new(url)
print('> Info:')
print(video)

# best = video.getbest(preftype="mp4")
print('> Streams:')
streams = [ s for s in video.streams if s.extension == 'mp4' ]
index = 0
for s in streams:
    print( f'[{index}] {s.resolution} {s.extension} {s.get_filesize()}')
    index +=1

print('Choose video resolution: ', end = '' )
try:
    choice = int(input())
except:
    print('Invalid choice')
    exit(1)

stream = streams[choice]
video_url = stream.url

filename_max_length = min(20, len(video.title))
video_folder = f'{DOWNLOAD_FOLDER}/{video.title[:filename_max_length]}'
if not os.path.exists(video_folder):
    os.makedirs(video_folder)
    
def get_frame(capture):
    ret, frame = capture.read()
    if ret:
        cv2.imshow('yt', frame)
    return ret, frame

def print_progress(current, tot, prefix =''):
    scale = 60/tot
    left = math.floor( (tot - current) * scale )
    curr = math.ceil ( current * scale )
    print('{}: |={}{}|'.format(prefix, '='*curr, ' '*left), end='\r')
    
def print_minutage(frames_count):
    tot_min = frames_count / (FPS * 60) 
    print(f'Frame {frames_count} - Tot diration: {tot_min} min')


shape = stream.dimensions[::-1]

if( SAVE_VIDEO ):
    cap = None
else:
    cap = cv2.VideoCapture()
    cap.open(video_url)

ret = True
frames_count = 0
video_chunk = 0

while True:

    if SAVE_VIDEO:
        current_frame = frames_count % VIDEO_CHUNKS_FRAMES
        # check if a new chunk must be created
        if current_frame == 0:
            if cap is None:
                cap = cv2.VideoCapture()
                cap.open(video_url)
            else:
                out.release()
            out = cv2.VideoWriter('{}/out{}.mov'.format(video_folder, video_chunk), VIDEO_CODEC, FPS, stream.dimensions)
            video_chunk += 1
            
        ret, frame = get_frame(cap)
        if frame.shape[:2] == shape:
            out.write(frame)
        
        print_progress(current_frame, VIDEO_CHUNKS_FRAMES, video_chunk)
        if video_chunk - 1  == MAX_CHUNKS_TO_SAVE:
            break
        
    else:
        ret, frame = get_frame(cap)
        framename = '{}/frame{:05d}.jpg'.format(video_folder, frames_count)
        cv2.imwrite(framename, frame)

        print_minutage(frames_count)               

    frames_count += 1
    if  ( cv2.waitKey(10) & 0xFF ) == ord('q'):
        break
    if not cap.isOpened() or ret == False:
        break

if SAVE_VIDEO: 
    out.release()
cap.release()
cv2.destroyAllWindows()