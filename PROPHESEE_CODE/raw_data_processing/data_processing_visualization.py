import matplotlib.pyplot as plt
import tqdm
import cv2
from enum import Enum
import numpy as np
import Input_Dialog as ID
import tkinter as tk
from tkinter import filedialog

def formEvents(EventArray, resolution,number_of_frames, integration_window, visualize = False, fps=33):
    '''
    Function that receives the array which contains events and creates an output array which accumulates events based on integration window. 

    Arguments
    ---------

    - EventArray (np.ndarray):
        Array that contains events in shape `(X, Y , T, P)`
    - resolution (Tuple(int,int)) 
        Tuple that contains the resolution of the camera
    - number_of_frames (int):
        Output number of frames
    - visualize (bool):
        Flag that denotes if the events are going to be visualized or not
    
    Returns
    -------

    - events_frames (np.ndarray)
        Output array of shape `(W,H,frames)` that contains the events accumulated based on `integration_window`

        
    Notes
    -----

        - It does NOT integrate the events but rather just accumulates them.
        
        - It ignores the polarity, keeping both negative and positive events as a discrete value of 1.

    '''
    start_index = EventArray[0][3]
    events_frames = np.zeros((resolution[0],resolution[1],number_of_frames), dtype = np.uint8)
    event_index = 0
    
    # for each frame 
    for frame in tqdm.tqdm(range(number_of_frames)):
        end_index = start_index + integration_window
        # calculate the end time as the start_time + integration window (in us)

        # Hence for each event in the array whose timestamp is between start, end index
        while( EventArray[event_index][3] < end_index):
            # retrieve X,Y coordinates
            x_idx = EventArray[event_index][0]
            y_idx = EventArray[event_index][1]
            
            # Asign value of 1, without taking polarity into account
            events_frames[y_idx][x_idx][frame] = 1
            # increment the event index to move to the next event in the array
            event_index+=1
            
        # update the starting_index, moving in the next integration_window time slot
        start_index = end_index
    
    if visualize:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        position= (10,30)
        font_color = (255,255,255)
        font_thickness = 1 

        for i in range(number_of_frames):
            frame = events_frames[:, :, i]
            
            frame_scaled = (frame * 255).astype(np.uint8)
            # Convert grayscale (if needed) to BGR for proper display
            frame_bgr = cv2.cvtColor(frame_scaled, cv2.COLOR_GRAY2BGR)
            txt = f"frame {i} Event count {np.sum(frame)}"
            cv2.putText(frame_bgr,txt,position,font,font_scale,font_color,font_thickness)
            # Display the frame using OpenCV
            cv2.imshow(f'All Frames Contained ', frame_bgr)

            # Wait for 25 ms between frames (adjust for desired frame rate)
            if cv2.waitKey(fps) & 0xFF == ord('q'):
                break

        # Release the display window
        cv2.destroyAllWindows()
    return events_frames
    

class BallsEnums(Enum):
    '''
    Enum that will be used to denote the class
    '''
    BALL_12UM = 0
    BALL_16UM = 1
    BALL_20UM = 2

    @classmethod
    def from_value(cls, value):
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"No enum member with value {value}")
        
    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
    
    def __str__(self):
        if self == BallsEnums.BALL_12UM:
            return "12"
        elif self == BallsEnums.BALL_16UM:
            return "16"
        elif self == BallsEnums.BALL_20UM:
            return "20"


GUI = True


if GUI == True:
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    dialog = ID.ThresholdDialog(root)
    root.wait_window(dialog.top)  # Wait for the dialog to close

    if dialog.result:
        
        BALL_CLASS = BallsEnums(int(dialog.selected_value.get()))
        print(f"Setting low threshold: {dialog.result[0]} and upper : {dialog.result[1]}\n Ball class is {BALL_CLASS}um")

        low_thr = dialog.result[0]
        high_thr = dialog.result[1]
    else:
        print("Dialog was cancelled or no valid input. Defaulting thr in [100-300]")
        low_thr = 100
        high_thr = 300

    file_path = filedialog.askopenfilename(title="Input File")

    if file_path:
        fname = file_path
    else:    
        fname = f"/home/gmoustakas/Documents/prophesee_processing/balls_raw/{BALL_CLASS}a2.txt"
else:
    BALL_CLASS = BallsEnums.BALL_12UM
    fname = f"/home/gmoustakas/Documents/prophesee_processing/balls_raw/{BALL_CLASS}a.txt"
    low_thr = 100 
    high_thr = 350












print(f"Opening file to parse {fname}\n")
with open(fname, "r") as fpt:
    FullFile = fpt.readlines()
        
EventElement = np.zeros((4))
EventArray = []

fpt.close()
count = 0
print("Parsing File ....")

# Parse the files that are generated by the camera

for lineText in FullFile:
    lineText = lineText.replace('[','')
    lineText = lineText.replace(']', '')
    lineText = lineText.replace(")", '')
    lineText = lineText.replace("(", '')
    lineText = lineText.replace(',', ' ')
    lineText = lineText.replace('\n', '')
    
    if lineText:
        ParsedLine = [int(x) for x in lineText.split() if x.isdigit()]
        for i in range(len(ParsedLine)):
                j = int(i % 4)
                EventElement[j] = ParsedLine[i]
                if (j==3):
                    EventArray.append(EventElement.astype(np.uint32))
                    count+=1
                    




# find the total duration by subtracting the TIMESTAMP of the first from the last event
total_duration = EventArray[-1][3] - EventArray[0][3] #get time in milliseconds




# use a fixed window of 480 microsecond (slicing the 480x640 into discete bins of 1 microsecond)
window = 480
no_frames = int((EventArray[-1][3] - EventArray[0][3])/window) # calculate the total number of frames possible based on 480 microseconds integration_window
print(f"Total number of frames : {no_frames}") 
print("Integration Window is set to:",window,"μsec")
print("Total Duration of the Recording is: ", total_duration*1e-6, "sec")

#split and create the events in the event array
events = formEvents(EventArray,(480,640),no_frames,window,True,1)

print(f"Formed Events have shape : {events.shape}")



#count all the events in each frame, will give you a 1d array of shape (no_frames,) containing the sum of all events at each frame
event_counts = np.sum(events, axis=(0, 1))  # This will give you a 1D array of shape (no_frames,)





# Use np.where to find indices of frames with more than low_thr and less than high_thr events
ball_events = np.where((event_counts > low_thr) &  (event_counts < high_thr)  )[0]





# Display the result
print(f"Found {ball_events.shape} Frames in between {low_thr} - {high_thr} events at indices: {ball_events}")
frames_with_events = np.take(events, ball_events, axis=2)





show_events = True
if show_events:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    position= (10,30)
    font_color = (255,255,255)
    font_thickness = 1 
    for i in range(frames_with_events.shape[2]):
        frame = frames_with_events[:, :, i]

        # Convert grayscale (if needed) to BGR for proper display
        frame_scaled = (frame * 255).astype(np.uint8)
        #frame_bgr = cv2.applyColorMap(frame_scaled, cv2.COLORMAP_JET)
        txt = f"frame {ball_events[i]} Total Events {np.sum(frame)}"
        cv2.putText(frame_scaled,txt,position,font,font_scale,font_color,font_thickness)
    
        # Display the frame using OpenCV
        cv2.imshow(f'Frames in between {low_thr} - {high_thr}', frame_scaled)

        # Wait for 25 ms between frames (adjust for desired frame rate)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()







#=============================== CROP THE FRAMES THAT CONTAIN EVENTS ================================#
import torch
from skimage import filters
from skimage.measure import regionprops
def Mass_Center (inco_image):

    threshold_value = filters.threshold_otsu(inco_image)
    
    labeled_foreground = (inco_image > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, inco_image)
    if (len(properties) != 0):
        center_of_mass = properties[0].centroid
        weighted_center_of_mass = properties[0].weighted_centroid
        return center_of_mass
    else:
        print("Something went Wrong")

BOX_WIDTH = 100
BOX_HEIGHT = 100
boxed_frames = []

for i in range(frames_with_events.shape[2]):
    frame = frames_with_events[:, :, i]

    center_of_mass = Mass_Center(frame) # find the center of mass
    y_center = center_of_mass[0] # row
    x_center = center_of_mass[1] # column
    
    # Make sure that we do not go off bounds
    if (x_center > 50 ) and (x_center < 590):
        x_min = x_center - BOX_WIDTH // 2

        y_min = y_center - BOX_HEIGHT // 2
        
        x_max = x_center + BOX_WIDTH // 2
        
        y_max = y_center + BOX_HEIGHT // 2

        

        boxed_frame = frame[int(y_min): int(y_max),int(x_min): int(x_max) ]

        bframe_events = np.sum(boxed_frame)

        # make sure we do not have any empty frames in the boxed frames
        if bframe_events < low_thr:
            continue
        else:
            boxed_frames.append(torch.from_numpy(boxed_frame))

        





# stack the frames
boxed_frames = torch.stack(boxed_frames)

print(f" Preparing labels for CLASS {BALL_CLASS} with value {BALL_CLASS.value}")
labels = np.ones(boxed_frames.shape[0])*BALL_CLASS.value


print(f"Final Cropped frames {boxed_frames.shape}")


event_counts_boxed = np.sum(boxed_frames.numpy(), axis=(1, 2))  # This will give you a 1D array of shape (no_frames,)



plt.figure()
plt.title("Cropped frames event count")
plt.bar(range(len(event_counts_boxed)), event_counts_boxed, width=1)
plt.show()

plt.figure()
plt.imshow(boxed_frames[0,:,:])
plt.show()


import h5py
with h5py.File(f'/home/gmoustakas/Documents/prophesee_processing/{BALL_CLASS}.hdf5', 'w') as hdf:
    hdf.create_dataset('data', data=boxed_frames)

    hdf.create_dataset('labels', data=labels)
