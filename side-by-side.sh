load module ffmpeg
ffmpeg -r 2 -i dust-map-left.%05d.png dust-map-left.mp4
ffmpeg -r 2 -i dust-map-right.%05d.png dust-map-right.mp4
ffmpeg \
  -i dust-map-left.mp4 \
  -i dust-map-right.mp4 \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map [vid] \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  output.mp4
