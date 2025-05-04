# Edge Impulse - OpenMV FOMO Object Detection Example
#
# This work is licensed under the MIT license.
# Copyright (c) 2013-2024 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE

import sensor, image, time, ml, math, uos, gc

# Function to calculate the mode of a list
def mode(lst):
    counts = {}
    for item in lst:
        counts[item] = counts.get(item, 0) + 1
    max_count = 0
    mode_value = None
    for k, v in counts.items():
        if v > max_count:
            max_count = v
            mode_value = k
    return mode_value, max_count

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
min_confidence = 0.5

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]

threshold_list = [(math.ceil(min_confidence * 255), 255)]

def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]

    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh

    scale = min(x_scale, y_scale)

    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2) + inputs[0].roi[1]

    l = [[] for i in range(oc)]

    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(
            threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1
        )
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = (
                img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            )
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l

# Function to calculate angle in degrees between two points with respect to center
def calculate_angle(point, center):
    dx = point[0] - center[0]
    dy = center[1] - point[1]  # Y is inverted in image coordinates
    angle = math.atan2(dy, dx)
    angle_deg = math.degrees(angle)
    # Normalize to 0-360 range
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

clock = time.clock()

reading_history = []
window_size = 50  # Number of readings to consider for mode

while(True):
    clock.tick()

    img = sensor.snapshot()

    # Reset positions for each frame
    center_pos = None
    pointer_pos = None
    min_pos = None
    max_pos = None

    # Process detections and store coordinates
    for i, detection_list in enumerate(net.predict([img], callback=fomo_post_process)):
        if i == 0: continue  # background class
        if len(detection_list) == 0: continue  # no detections for this class?

        label = labels[i]
        for x, y, w, h, score in detection_list:
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))

            # Store coordinates based on label
            if label == "centre":
                center_pos = (center_x, center_y)
            elif label == "pointer":
                pointer_pos = (center_x, center_y)
            elif label == "min":
                min_pos = (center_x, center_y)
            elif label == "max":
                max_pos = (center_x, center_y)

    # Calculate angle and reading if all required points are detected
    if center_pos and pointer_pos and min_pos and max_pos:
        # Calculate angles
        min_angle = calculate_angle(min_pos, center_pos)
        max_angle = calculate_angle(max_pos, center_pos)
        pointer_angle = calculate_angle(pointer_pos, center_pos)

        if max_angle > min_angle:
            total_angle = 360 - (max_angle - min_angle)
        else:
            total_angle = (max_angle - min_angle) + 360

        if min_angle > pointer_angle:
            angle_diff = min_angle - pointer_angle
        else:
            angle_diff = (min_angle - pointer_angle) + 360

        # Calculate the reading using linear interpolation
        min_value = 0  # Minimum value on the meter
        max_value = 30  # Maximum value on the meter
        reading = min_value + (angle_diff / total_angle) * (max_value - min_value)

        # Round the reading to 2 decimal places
        # Using a window size of readings to calculate the mode
        reading_rounded = round(reading, 2)
        reading_history.append(reading_rounded)
        if len(reading_history) > window_size:
            # Keep only the last 'window_size' readings (sliding window)
            reading_history.pop(0)

        # Only print when we have enough readings for the mode
        if len(reading_history) == window_size:
            mode_value, mode_count = mode(reading_history)
            print("********** MODE READING **********")
            print("Mode of last %d readings: %.2f (occurred %d times)" % (window_size, mode_value, mode_count))
            img.draw_string(12, 12, "Reading: %.2f" % mode_value, color=(255,255,255), scale=2)

        # Draw lines to visualize the angles
        img.draw_line(center_pos[0], center_pos[1], min_pos[0], min_pos[1], color=(255, 0, 0))
        img.draw_line(center_pos[0], center_pos[1], max_pos[0], max_pos[1], color=(255, 0, 0))
        img.draw_line(center_pos[0], center_pos[1], pointer_pos[0], pointer_pos[1], color=(255, 0, 0))
