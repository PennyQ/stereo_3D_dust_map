exiftool 2D_dust_map.JPG \
-ImageMime="image/jpeg" \
-ProjectionType="equirectangular" \
-UsePanoramaViewer="True" \
-CaptureSoftware="PhotoSphere" \
-StitchingSoftware="PhotoSphere" \
-PoseHeadingDegrees=150 \
-CroppedAreaImageWidthPixels=4800 \
-CroppedAreaImageHeightPixels=2400 \
-FullPanoWidthPixels=4800 \
-FullPanoHeightPixels=2400 \
-CroppedAreaLeftPixels=0 \
-CroppedAreaTopPixels=0 \
-"PoseHeadingDegrees<$exif:GPSImgDirection"

exiftool -xmp -b 2D_dust_map.JPG > 2D_dust_map.xmp
