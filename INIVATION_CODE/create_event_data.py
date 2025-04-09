import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import tkinter as tk
from tkinter import filedialog,simpledialog
import numpy as np
import re
from datetime import datetime
import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def select_folder():
    """
    Opens a file dialog for the user to select a file from a predefined directory.

    Returns
    -------
    str or None
        The selected file path if a file is chosen, otherwise None.
    """
    # Open the folder dialog
    folder_path = filedialog.askopenfilename(initialdir='/mnt/sdb1/MOUSTAKAS/INIVATION_DATA/')
    
    # Print the selected folder path
    if folder_path:
        print("Selected File:", folder_path)
        return folder_path
    else:
        print("No folder selected")
        return None

def extract_path_metadata(path: str) -> tuple[str, str, str]:    
    """
    Extracts metadata from a given file path, including class, velocity, and timestamp.

    Parameters
    ----------
    path : str
        The full file path.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing:
        - class_value (str): The extracted class (e.g., '20um') or None if not found.
        - velocity_value (str): The extracted velocity (e.g., '0.02') as a string or None if not found.
        - datetime_value (str): The formatted datetime string ('YYYY-MM-DD_HH:MM:SS') or None if not found.
    """

    class_pattern = r'/(\d+um)/'  # Matches the class like '20um'
    velocity_pattern = r'(\d+KHz)_(\d+\.\d+)'  # Matches '10KHz_0.02' and extracts '0.02'
    datetime_pattern = r'dvSave-(\d{4}_\d{2}_\d{2})_(\d{2}_\d{2}_\d{2})'  # Matches the date and time

    # Extract class
    class_match = re.search(class_pattern, path)
    class_value = class_match.group(1) if class_match else None

    # Extract velocity
    velocity_match = re.search(velocity_pattern, path)
    velocity_value = float(velocity_match.group(2)) if velocity_match else None

    # Extract datetime
    datetime_match = re.search(datetime_pattern, path)
    if datetime_match:
        date_str = datetime_match.group(1)  # '2024_11_27' 
        time_str = datetime_match.group(2)  # '15_11_52'
        datetime_value = datetime.strptime(f"{date_str}_{time_str}", '%Y_%m_%d_%H_%M_%S')
    else:
        datetime_value = None

    return class_value, str(velocity_value), "_".join(str(datetime_value).split(" "))

def get_parameters():
    """
    Opens a dialog window to collect user input for filtering parameters.

    Returns
    -------
    tuple[float, float, float] or None
        A tuple containing:
        - threshold_min (float): Minimum threshold value.
        - threshold_max (float): Maximum threshold value.
        - integration_window (float): Integration window in microseconds.
        
        Returns None if input is canceled or invalid.
    """
    # Create the main window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Create a dialog to get the threshold value
    threshold_min = simpledialog.askfloat("Input", "Enter min threshold value:", parent=root)
    threshold_max = simpledialog.askfloat("Input", "Enter max threshold value:", parent=root)
    # Create a dialog to get the integration window value
    integration_window = simpledialog.askfloat("Input", "Enter integration window value: (μs)", parent=root)

    # Check if values are provided
    if threshold_min is not None and threshold_max is not None and integration_window is not None:
        print("Threshold min:", threshold_min)
        print("Threshold max:", threshold_max)
        print("Integration Window:", integration_window)
        return threshold_min, threshold_max,integration_window
    else:
        print("Input was canceled or invalid.")
        return None
    





def create_video(filtered_data,filename,brigthness,resize=False,playback=True):
    """
    Creates a video from filtered event data, applying color encoding for positive and negative events.

    Parameters
    ----------
    filtered_data : np.ndarray
        A 4D numpy array with shape (T, Y, X, 2), where T is the number of frames,
        Y and X are spatial dimensions, and the last dimension represents event polarities
        (filtered_data[..., 0] for positive, filtered_data[..., 1] for negative).
    filename : str
        Output filename for the generated video (e.g., 'output.mp4').
    brightness : float
        Scaling factor for brightness (values > 1 increase brightness, values < 1 decrease brightness).
    resize : bool, optional
        If True, resizes the video to 640x480; otherwise, keeps the original dimensions (default: False).
    playback : bool, optional
        If True, displays the video while processing; otherwise, just writes to the file (default: True).

    Returns
    -------
    None
    """
    import cv2

    # Assuming 'filtered_data' is your array of shape (T, Y, X, 2)
    height, width = filtered_data.shape[1], filtered_data.shape[2]  # Y, X dimensions

    # Define video parameters: You can adjust the frame rate (fps) as needed
    fps = 5

    # Define desired window size (e.g., 1600x1600 for a larger display)
    if resize:
        window_width = 640
        window_height = 480
    else:
        window_height = 150
        window_width = 150

    # Brightness factor (values > 1 will increase brightness)
    brightness_factor = brigthness  # Adjust this value as needed

    # Define the video output file name and codec (e.g., .mp4 using the 'mp4v' codec)
    output_file = filename
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (window_width, window_height))

    # Loop through filtered_data and display video frames
    for t in range(filtered_data.shape[0]):
        # Convert the frame into an appropriate format (e.g., 8-bit uint8 for grayscale)
        frame = filtered_data[t].astype(np.uint8)

        # Create a 3-channel image (Red for negative, Green for positive)
        color_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Assign positive polarity to the green channel (scale brightness)
        color_frame[:, :, 1] = np.clip(frame[:, :, 0] * brightness_factor, 0, 255)  # Green channel for positive polarity
        
        # Assign negative polarity to the red channel (scale brightness)
        color_frame[:, :, 2] = np.clip(frame[:, :, 1] * brightness_factor, 0, 255)  # Red channel for negative polarity

        # Display the resized frame
        if playback:
            

            
            if resize:
                # Resize the frame to the desired window size
                resized_frame = cv2.resize(color_frame, (window_width, window_height))

                cv2.imshow('Polarity Visualization', resized_frame)
                video_writer.write(resized_frame)
            else:
                cv2.imshow('Polarity Visualization', color_frame)
                video_writer.write(color_frame)
                # Write the resized frame to the video file

            # Wait for a key press to move to the next frame or stop (key press 'q' will exit)
            if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
                break
        else:
            if resize:
                resized_frame = cv2.resize(color_frame, (window_width, window_height))
                video_writer.write(resized_frame)
            else:
                video_writer.write(color_frame)

    # Release the video writer and close the OpenCV window
    video_writer.release()
    cv2.destroyAllWindows()

def plot_events(filtered_events,gridsize=150):
    """
    Visualizes event data on a grid, with events placed according to their X and Y coordinates.
    The color of each event represents its polarity (0 or 1).

    Parameters:
    -----------
    filtered_events : numpy.ndarray
        A 2D array where each row represents an event. The first two columns represent the X and Y coordinates,
        and the fourth column represents the polarity (0 or 1).
    
    gridsize : int, optional, default: 150
        The size of the grid on which events will be plotted.

    Returns:
    --------
    None
        The function generates a plot displaying the events on the grid.

    Example:
    --------
    plot_events(filtered_events, gridsize=200)
    """
    x_val = filtered_events[:,0] # get the X coordinates
    y_val = filtered_events[:,1] # get the Y coordinates
    pol = filtered_events[:,3] # get the polarization
    data = np.column_stack((x_val, y_val, pol)) # Stack to create X,Y,P 
    grid_size = gridsize  # Size of the grid
    grid = np.zeros((grid_size, grid_size))  # Initialize grid with zeros

    # Fill the grid based on (X, Y) coordinates and polarity
    for x, y, p in data:
        if p == 0:
            grid[y, x] = -1  # Set polarity value at the (Y, X) coordinate to negative if the P == False (0)
        else:
            grid[y,x] = 1

    # Visualize using imshow
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='coolwarm', origin='lower')  # Use a colormap (e.g., 'coolwarm')
    plt.gca().invert_yaxis()  # Invert the y-axis if necessary to match expected coordinate system
    plt.colorbar(label='Polarity')
    plt.title("X, Y Data Visualization with Polarity", fontsize=14)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.grid(False)  # Remove grid lines for clarity
    plt.show()



def cluster_events(filtered_events, n_clusters=3, crop=False, visualise=False, box_size = 150):
    """
    Clusters event data using KMeans and optionally crops and visualizes the events within a bounding box.
    Optionally, returns cropped and centered event data, and visualizes the clustering process.

    Parameters:
    -----------
    filtered_events : object
        The event data containing X and Y coordinates, timestamps, and polarities. Expected to have methods
        `coordinates()` and `timestamps()`.
    
    n_clusters : int, optional, default: 3
        The number of clusters to form using the KMeans algorithm.
    
    crop : bool, optional, default: False
        If True, crops the event data into a bounding box based on the global coordinates of all events.
    
    visualise : bool, optional, default: False
        If True, visualizes the events, the cropping process, and the bounding box.

    box_size : int, optional, default: 150
        The size of the box to which the cropped event data will be adjusted.

    Returns:
    --------
    cropped_events_adjusted : numpy.ndarray
        The event data after cropping and adjusting the coordinates to fit within the bounding box.

    centered_events : numpy.ndarray
        The event data after adjusting the coordinates so that the events are centered within the bounding box.

    (width, height) : tuple
        The width and height of the bounding box if cropping is performed.

    If cropping is not performed, returns the original `filtered_events` data.

    Example:
    --------
    cropped_events_adjusted, centered_events, (width, height) = cluster_events(filtered_events, n_clusters=5, crop=True, visualise=True)
    """
    # Extract event data
    x_coords = filtered_events.coordinates()[:, 0]
    y_coords = filtered_events.coordinates()[:, 1]
    timestamps = filtered_events.timestamps()
    polarities = filtered_events.polarities()

    # Combine data into a single array for clustering
    data = np.column_stack((x_coords, y_coords, timestamps, polarities))
    data_normalized = StandardScaler().fit_transform(data)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(data_normalized)

    if crop:
        # Find the global bounding box that contains all clusters
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Crop events based on the global bounding box
        cropped_mask = (x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)
        x_cropped = x_coords[cropped_mask]
        y_cropped = y_coords[cropped_mask]
        time_cropped = timestamps[cropped_mask]
        polarities_cropped = polarities[cropped_mask]

        # Combine cropped data into one array to return
        cropped_events = np.column_stack((x_cropped, y_cropped, time_cropped, polarities_cropped))
        x_cropped_adjusted = x_cropped - x_min
        y_cropped_adjusted = y_cropped - y_min
        
        width = x_max - x_min
        height = y_max - y_min
        

        center_x = box_size // 2
        center_y = box_size // 2

        # Calculate the center of the cropped adjusted data
        cropped_center_x = x_cropped_adjusted.mean()
        cropped_center_y = y_cropped_adjusted.mean()

        # Compute offsets to center the cropped data in the box
        x_offset = center_x - cropped_center_x
        y_offset = center_y - cropped_center_y

        # Apply offsets
        x_centered = x_cropped_adjusted + x_offset
        y_centered = y_cropped_adjusted + y_offset

        # Ensure the data stays within the box bounds
        x_centered = np.clip(x_centered, 0, box_size - 1)
        y_centered = np.clip(y_centered, 0, box_size - 1)

        # Combine normalized data into one array to return
        
        cropped_events_adjusted = np.column_stack((x_cropped_adjusted, y_cropped_adjusted, time_cropped, polarities_cropped))
        
        centered_events = np.column_stack((x_centered, y_centered, time_cropped, polarities_cropped))
        # Visualize if requested
        if visualise:
            plt.scatter(x_coords, y_coords, c='lightgray', s=1, label='All Events')
            plt.scatter(x_cropped, y_cropped, c='red', s=1, label='Cropped Events')
            plt.gca().invert_yaxis()  # Invert the y-axis if necessary to match expected coordinate system
            plt.axvline(x_min, color='blue', linestyle='--', label='Crop Boundary')
            plt.axvline(x_max, color='blue', linestyle='--')
            plt.axhline(y_min, color='green', linestyle='--')
            plt.axhline(y_max, color='green', linestyle='--')
            plt.legend()
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('Global Bounding Box for All Clusters')
            plt.show()

            print(f"Box width is {x_max - x_min} and height is {y_max - y_min}")

        # Return the cropped events along with all their data
        return cropped_events_adjusted,centered_events, (x_max-x_min,y_max-y_min)

    # If no cropping is done, return None or the full data (for clustering purposes)
    return filtered_events

# First retreive the file
path = select_folder()

# After getting the file, retrieve essential metadata like class, speed and datetime
selected_class, speed, date_time = extract_path_metadata(path)
class_dict = {
    "20um":2,
    "16um":1,
    "12um":0
}

# use a default path if the user did not choose one
if path is None:
    path = "/home/gmoustakas/Documents/INIVATION/1Khz_0.01_Trompa/dvSave-2024_11_12_15_11_52.aedat4"


# Create a reader object that will start reading event batches from the .aedat4 file
reader = dv.io.MonoCameraRecording(path)
print(f"Opened an AEDAT4 file which contains data from [{reader.getCameraName()}] camera")
i = 0

# Ask the user for threshold_min, max and integration window
threshold_min, threshold_max,integration_window = get_parameters()

# Some default values
visualize = True
silence = True

if threshold_min is None:
    threshold_min = 250
if threshold_max is None:
    threshold_max = 2*threshold_min
if integration_window is None:
    integration_window = 100


# Configuration for the EventVisualiser object, both positive and negative events are shown as white (no visible polarization)
visualizer = dv.visualization.EventVisualizer(reader.getEventResolution())
visualizer.setBackgroundColor(dv.visualization.colors.black())
visualizer.setPositiveColor(dv.visualization.colors.white())
visualizer.setNegativeColor(dv.visualization.colors.white())


slicer = dv.EventStreamSlicer()
print(reader.getEventResolution())


total_events = []
event_structs = []

first_filtered_found = False
global_batch_start = None
timesteps = 50_000 #Assume a large number of timesteps

# Create the array that is going to hold the events, assuming a 150x150 bounding box and 2 polarizations
data = np.zeros((timesteps,150,150,2))
non_normalized_data = np.zeros_like(data)
final_data = np.zeros_like(data) # Here we will store the data

diameters = []
final_events = []
i = 0
j = 0

all_event_batches = []
def slicing_callback(events: dv.EventStore):
    ''' 
    Callback function that is called every `integration_window` to process the event batch.

    This function filters out redundant or isolated pixels based on neighborhood activity using the 
    `dv.noise.BackgroundActivityNoiseFilter`. After filtering, it performs additional checks and processes
    the events depending on their count relative to specified thresholds. If the number of filtered events
    falls between a low and high threshold, the function proceeds with clustering, calculating diameters, 
    and updating event data.

    Parameters
    ----------
    events : dv.EventStore
        An instance of `dv.EventStore` containing the events to be processed.

    Returns
    -------
    None
        This function doesn't return any value. It processes and stores event data globally, updates visualizations,
        and adjusts internal variables and data structures.

    Side Effects
    ------------
    - Filters the events using `dv.noise.BackgroundActivityNoiseFilter`.
    - Updates global variables such as `i`, `timesteps`, `global_batch_start`, and `data`.
    - Appends filtered event counts to `all_event_batches`.
    - Optionally visualizes filtered events using `cv.imshow`.
    - Updates global event-related arrays like `final_data` and `data`.

    '''
    global visualize, silence,i,timesteps,global_batch_start,time_interval,data,first_filtered_found

    # Use a back
    filter = dv.noise.BackgroundActivityNoiseFilter(reader.getEventResolution(), backgroundActivityDuration=timedelta(microseconds=integration_window))
    

    filter.accept(events)


    # Call generate events to apply the noise filter

    filtered = filter.generateEvents()
    # in the first batch, get the time from the first batch
    


    if silence == False:
        print(f"From {len(events)} to {len(filtered)}")

    all_event_batches.append(len(filtered))
    #print(events.shape)
    frame = visualizer.generateImage(events)

    frame_filtered = visualizer.generateImage(filtered)
    
    

    if visualize == True:
        cv.imshow("Tsilikas unfiltered", frame)
        cv.imshow("Tsilikas ",frame_filtered)

    if len(filtered) >50: # remove empty frames
        total_events.append(len(filtered))
    
    
    if len(filtered) > threshold_min and len(filtered) < threshold_max:
        if visualize == True:   
            cv.waitKey(0)
        
        # create the timesteps array that will be updated every time we find events between the min/max, keeping only the times when we found a PMMA particle, removing redundant frames and saving memory
        timesteps_norm = np.ones_like(filtered.timestamps())*i
        cropped, centered, _ = cluster_events(filtered,3,True,False) # retrieve the centered and cropped event frames
        p = (centered[:,3]).astype(int) # fetch their polarities, w and h
        w = (centered[:,0]).astype(int)
        h = (centered[:,1]).astype(int)
        
        # calculate the coordinates along with diameters 
        x_centered = (centered[:,0]).astype(int)
        y_centered = (centered[:,1]).astype(int)
        
        x_min, x_max = x_centered.min(), x_centered.max()
        y_min, y_max = y_centered.min(), y_centered.max()
        width = x_max - x_min
        height = y_max - y_min
        diameter = np.sqrt(width**2 + height**2)

        diameters.append(np.array([(width,height,width*height,diameter)]))
        if diameter >=50 and diameter <=75:
            np.add.at(final_data, (timesteps_norm, h, w, p), 1.0)
        
        crp = cropped[:,3]
        crw = cropped[:,0]
        crh = cropped[:,1]
        
        
        try:
            np.add.at(data, (timesteps_norm, h, w, p), 1.0)
            np.add.at(non_normalized_data,(timesteps_norm,crh,crw,crp),1.0)
        except:
            y_coords = filtered.coordinates()[:, 1]
            x_coords = filtered.coordinates()[:, 0]
            x_cropped = centered[:,0]
            y_cropped = centered[:,1]
            #print("ignoring")  
        i = i + 1


    if visualize == True:
        key = cv.waitKey(1)

        if key == 32:
            cv.waitKey()
        elif key == ord('q'): 
            visualize = False
            silence = True
            cv.destroyAllWindows()

    




# Every `integration_window` us, it is going to call the slicing_callback function, each time passing a new event batch retrieved from the file
slicer.doEveryTimeInterval(timedelta(microseconds=integration_window),slicing_callback)



cv.namedWindow("Event Frames Viewer", cv.WINDOW_NORMAL)
with tqdm.tqdm(desc="Processing Batches ", unit=" batch") as pbar:
    while reader.isRunning():
        event = reader.getNextEventBatch()
        #print(f"{event}")
        if event is not None:
            slicer.accept(event)
            pbar.update(1)  # Update tqdm progress bar for each batch

# The code is checking for non-zero values across the X, Y, and P axes of a multi-dimensional array
# `data`. It creates a boolean array `timesteps_with_events` where each element represents whether
# there are any non-zero values at a particular timestep.

timesteps_with_events = np.any(data != 0, axis=(1, 2, 3))  # Check for non-zero values across X, Y, and P axes

last_event_index = np.where(timesteps_with_events)[0][-1]

# Keep all timesteps until the last one with events, so this might keep also empty/erroneous event frames, be cautious
filtered_data = data[:last_event_index + 1]

print(f"Original number of batch events: {pbar.last_print_n + 1}")
print(f"Number of timesteps after filtering: {filtered_data.shape[0]}")




CLASS = class_dict[selected_class] # 2 for 20um, 1 for 16um and 0 for 12um

import h5py
labels = np.ones(filtered_data.shape[0])*CLASS
filename = f"./{selected_class}/{speed}/{speed}_{date_time}.h5"
with h5py.File(filename, 'w') as h5f:
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('diameters', data=diameters)
    h5f.create_dataset('data', data=filtered_data)
# create a video
video_filename = f"./{selected_class}/{speed}/videos/{speed}_{date_time}.mp4"
create_video(filtered_data,video_filename,255.0,playback=False)
diameters_array = np.array(diameters).squeeze(1)


# plot histograms if needed
plt.hist(all_event_batches)
plt.title("Total Events Histogram")
plt.show()

plt.hist(diameters_array[:,3])
plt.title("Diameters histogram")
plt.show()
