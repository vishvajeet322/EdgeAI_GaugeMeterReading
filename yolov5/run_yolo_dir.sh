#!/bin/bash 

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

parent_dir="$1"  # <<< Update this

# Loop through all directories
for dir in "$parent_dir"/*/; do
    if [ -d "$dir" ]; then
        echo "✅ Found directory: $dir" 
        python detect.py --weights runs/train/gauge_reader_yolov5n/weights/best.pt --source "$dir" --conf 0.25 --save-txt
        
    fi
done

