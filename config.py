from camera_route import *
import os


'''Run on local or server'''
a = raw_input("run on local or server: [l/s] ")

# render in server
if str(a) == 's':
    # Output figure store place
    pan1 = '/n/fink2/xrqian'
    if not os.path.isdir(pan1):
        pan1 = '~/n/home15/xrqian/'
    # Input file dir
    # map_fname = '/n/fink1/ggreen/bayestar/output/allsky_2MASS/compact/dust-map-3d-uncompressed.h5'#compact_10samp.h5'
    map_fname = '../dust-map-3d-uncompressed.h5'
    # fname = '/3d/allsky_2MASS/dust-map-stereo-pair/dust-map.png'
    
# render on local laptop, need to modify config file   
if str(a) == 'l':
    pan1 = '.'
    map_fname = 'data/AquilaSouthLarge2_unified.h5' 
    # fname = '/3d/allsky_2MASS/AqS/AqS-loop-hq.png'
    
# users set input & output dir
if str(a) not in ['s', 'l']:
    pan1 = str(raw_input('Enter the main dir for storing output figures: '))
    # fname = str(raw_input('Enter figure name: (end as .png)'))
    if not os.path.isdir(pan1):
        print('Input dir not exist, exiting...')
        exit()
    
# Camera path/orientation
#camera_pos = local_dust_path(n_frames=400)
#camera_pos = paper_renderings()
#camera_pos = Orion_flythrough(n_frames=400)
#camera_pos = grand_tour_path(n_frames=20)#1600)
# camera_pos = circle_local(n_frames=3, l_0=30., b_0=5.)

try:
    f = int(raw_input('How many frames? '))
except ValueError:
    print('Not a number, default frame num as 20')
    f = 20

'''Set camera render mode'''
mode = str(raw_input('Which camera mode? [circle local(cl)|grand tour(gt)|local dust(ld)]'))
if mode == 'cl':
    fname = '/3d/allsky_2MASS/circle-local/dust-map-cl.png'
    
    b = raw_input('Render side-by-side? [y/n] ')
    if str(b) == 'y':
        camera_pos = [circle_local_left(n_frames=f), circle_local_right(n_frames=f)]
    if str(b) == 'n':
        camera_pos = circle_local(n_frames=f)
        
# TODO: add side-by-side render
if mode == 'gt':
    fname = '/3d/allsky_2MASS/grand-tour/dust-map-gt.png'
    camera_pos = grand_tour_path(n_frames=f)
    
if mode == 'ld':
    fname = '/3d/allsky_2MASS/local-dust/dust-map-ld.png'
    camera_pos = grand_tour_path(n_frames=f)

# Initiate processing core numbers
n_procs = 10

# Misc settings
plot_props = {
    'fname': pan1 + fname, #'3d/allsky_2MASS/grand-tour/simple-loop-att-v2-lq.png',
    'figsize': (10, 7),  # figure aspect ratio
    'dpi': 100,
    'n_averaged': 3,
    'gamma': 1.,
    'R': 3.1,
    'scale_opacity': 1.,
    'sigma': 0,
    'oversample': 2,
    'n_stack': 10,
    'randomize_dist': True,
    'randomize_ang': True,
    'foreground': (255, 255, 255),
    'background': (0, 0, 0)
}

#for key in camera_pos:
#    camera_pos[key] = camera_pos[key][:20]
try:
    q = int(raw_input('How good the quality is? [1 the lowest, 10 the highest]'))
except ValueError:
    print('Not a number, default as 2')
    q = 2
# Camera properties
camera_props = {
    'proj_name': 'stereo',
    'fov': 140.,
    'n_x': 200*q, # num of pixels
    'n_y': 140*q,
    'n_z': 500*q,
    'dr': 10./q,  # 10pc per step
    'z_0': 1., #(0., 0., 0.)
}


# General label properties
label_props = {
    'text_color': (0, 166, 255),#(255, 255, 255),
    'stroke_color': (9, 73, 92),#(0, 0, 0) #(192, 225, 235)
}
 