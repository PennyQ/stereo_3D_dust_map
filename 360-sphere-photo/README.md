### Here are two solutions to add metadata to image:

1. Using exiftool (add_xmp_by_exiftool.sh), you can install the exiftool from this link and then change the file name on this shell script and run it. And you can check the metadata by checking the XXX.xmp.

2. Using a Python tool as on https://github.com/python-xmp-toolkit/python-xmp-toolkit (yeah the one Doug suggested before), the installation is on the doc . The code is attached as python-xml-toolkit.py


To view the metadata, you can use exiftool as typing in terminal:

`>>> exiftool -b -xmp 2D_dust_map.JPG > output.xmp`

Or use Python terminal as:
```from libxmp.utils import XMPFiles
xmpfile=XMPFiles(file_path=‘2D_dust_map.JPG')
xmp=xmpfile.get_xmp()
print(xmp)```
