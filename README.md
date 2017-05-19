# stero_3D_dust_map

### Required dependency:
The code needs Python libraries `healpy` and `h5py` for calculating coordinate and loading hdf5 files, you can them get by:
  
    pip install healpy h5py

Also `dvipng` program to convert the DVI output of the TeX typesetting system into PNG image format. On the server you can:
    
    module  load divpng (error as: “dvipng: not found” when creating matplotlib figure)
    
### Usage: 
Please modify the data and output directory in `config.py`, and then run `render3d.py` in the terminal. The code will collect your input to set rendering quality.

### Generate videos:
`render3d.py` will output a bunch of frame images, in the output directory stored in `config.py`. You can use 

    ffmpeg -r 4 -i figure_name.%05d.png output.mp4

-r is the frame rate (like how many frames per sec).

The script `side-to-side.sh` to generate a side-by-side video is also in this repo, please copy it to your output dir and modify it with correct figure name. It requires frame images for left and right camera.
  
