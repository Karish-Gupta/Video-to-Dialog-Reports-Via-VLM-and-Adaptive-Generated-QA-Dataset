#!/bin/bash

# Script to move timestamped transcripts to parent folder and delete subfolders

OUTPUT_DIR="/home/kagupta/Video-to-Dialog-Reports-Via-VLM-and-Adaptive-Generated-QA-Dataset/WhisperX/output"

cd "$OUTPUT_DIR" || exit 1

echo "Moving timestamped transcript files to parent directory..."

# Counter for moved files
moved_count=0

# Loop through all video* subdirectories
for dir in video*/; do
    if [ -d "$dir" ]; then
        # Extract video number from directory name
        video_num="${dir#video}"
        video_num="${video_num%/}"
        
        # Construct the timestamped file path
        timestamped_file="${dir}Video${video_num}Transcript_timestamped.txt"
        
        # Check if the file exists and move it
        if [ -f "$timestamped_file" ]; then
            mv "$timestamped_file" .
            ((moved_count++))
            echo "Moved: Video${video_num}Transcript_timestamped.txt"
        else
            echo "Warning: Timestamped file not found for video${video_num}"
        fi
    fi
done

echo ""
echo "Moved $moved_count timestamped transcript files."
echo ""
echo "Deleting video subfolders..."

# Delete all video* subdirectories
rm -rf video*/

echo "Done! All video subfolders have been deleted."
echo ""

# Count final files
final_count=$(ls -1 | grep "^Video.*_timestamped\.txt$" | wc -l)
echo "Total timestamped transcripts in output directory: $final_count"
