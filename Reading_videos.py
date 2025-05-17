import cv2
import os

# setting the video_path
video_path = "traffic_video.mp4"
if not os.path.exists(video_path):
    print ('Error: Vidoe file not found!')
#else: print('it is working!')

def read_video():
    #connecting the video    
    cap = cv2.VideoCapture('traffic_video.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        exit()
    else: print('cap is opened')

    ''' viewing the videos '''
    #reading frame by frame
    # to skip or limit the number of frame reading , 
    # add frame_count and frame_count%frame_skip==0 before imshow()
    while cap.isOpened():
        #cap.read() return whether there is frame and the frame
        ret,frame = cap.read()

        #imshow() to show the frame
        cv2.imshow('name_of_window',frame)

        delay = int(1000/fps)
        if cv2.waitKey(delay) == ord('q'):
            break

def write_video_to_file():

    cap = cv2.VideoCapture('traffic_video.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #cap.set(3,1280)  #setting width
    #cap.set(4, 720)  #setting height

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    root = os.getcwd()
    out_path = os.path.join(root, 'test.avi') 
    
    #initialize a VideoWriter object for video writing 
    out = cv2.VideoWriter(out_path, fourcc, fps, (width,height) ,isColor=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray_frame = cv2. cvtColor(frame,cv2.COLOR_BGR2GRAY)
            out.write(gray_frame)
            cv2.imshow('Video_Window',gray_frame)
        else:
            break
        if cv2.waitKey(int(1000/fps)) == ord('q'):
            break

    # free up the memory and finalizing the video file
    cap.release()
    out.release()
    cv2.destroyAllWindows()

write_video_to_file()





