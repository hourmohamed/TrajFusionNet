#this script creates sequences of frames from the videos in clips directory
#using FFMPEG
#If you don't have ffmpeg installed on your system
#run sudo apt-get install ffmpeg

CLIPS_DIR=/kaggle/working/TrajFusionNet/data/JAAD/JAAD_clips
FRAMES_DIR=/kaggle/working/TrajFusionNet/data/JAAD/images

################################################################


for file in ${CLIPS_DIR}/*.mp4
do
if [ -d ${file} ]; then
continue;
fi

#make a directory to save frame sequences
mkdir ${FRAMES_DIR}

filename=$(basename "$file")
fname="${filename%.*}"
echo $fname

#create a directory for each frame sequence
mkdir ${FRAMES_DIR}/$fname
ffmpeg -i $file -vf fps=1 -start_number 0 -f image2 -qscale 1 ${FRAMES_DIR}/$fname/%05d.png

done