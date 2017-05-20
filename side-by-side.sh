module load ffmpeg
ffmpeg -r 2 -i dust-map-cl-left.%05d.png dust-map-left.mp4
ffmpeg -r 2 -i dust-map-cl-right.%05d.png dust-map-right.mp4
ffmpeg \
  -i dust-map-left.mp4 \
  -i dust-map-right.mp4 \
  -filter_complex '[0:v]scale=iw:ih,setsar=sar/2,pad=2*iw:ih [left];[1:v]scale=iw:ih, setsar=sar/2[right];[left][right]overlay=main_w/2:0 [out]' \
  -map [out] \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  output.mp4
