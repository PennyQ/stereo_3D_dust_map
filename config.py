from camera_route import *
import os
import datetime

logfile = open('log.txt', 'a')
logfile.truncate()
logfile.write('\n')
logfile.write('---------'+str(datetime.datetime.now())+'-------'+'\n')

'''    
Add environment path to ~/.bashrc (depent on which shell the user use):

PAN1 is the output dir
MAP_FNAME is the dust map dataset dir

'''
pan1 = os.environ['PAN1']
map_fname = os.environ['MAP_FNAME']
    
# Camera path/orientation
#camera_pos = local_dust_path(n_frames=400)
#camera_pos = paper_renderings()
#camera_pos = Orion_flythrough(n_frames=400)
#camera_pos = grand_tour_path(n_frames=20)#1600)
# camera_pos = circle_local(n_frames=3, l_0=30., b_0=5.)
axis_on = True

stop_f = int(raw_input('Render interrupt at (0 mean none)[file name change required in render3d.py]'))
if stop_f == 0:
    stop_f = None

try:
    f = int(raw_input('How many frames? '))
    logfile.write('How many frames? '+ str(f)+ '\n')
except ValueError:
    print('Not a number, default frame num as 20')
    f = 20

'''Set camera render mode'''
mode = str(raw_input('Which camera mode? [circle local(cl)|grand tour(gt)|local dust(ld)|nw-270(nw)]'))
logfile.write('Which camera mode? ' + mode + '\n')
if mode == 'cl':
    fname = '/3d/allsky_2MASS/circle-local/dust-map-cl.png'    
    b = raw_input('Render side-by-side? [y/n] ')
    if str(b) == 'y':
        camera_pos = circle_local(n_frames=f, side_by_side=True, stop_f=stop_f)
    if str(b) == 'n':
        camera_pos = circle_local(n_frames=f, stop_f=stop_f)
        
# TODO: add side-by-side render
if mode == 'gt':
    fname = '/3d/allsky_2MASS/grand-tour/dust-map-gt.png'
    b = raw_input('Render side-by-side? [y/n] ')
    if str(b) == 'y':
        camera_pos = grand_tour_path(n_frames=f, side_by_side=True, stop_f=stop_f)
    if str(b) == 'n':
        camera_pos = grand_tour_path(n_frames=f, stop_f=stop_f)
    
if mode == 'ld':
    fname = '/3d/allsky_2MASS/local-dust/dust-map-ld.png'
    camera_pos = local_dust_path(n_frames=f)
    
if mode == 'nw':
    fname = '/3d/allsky_2MASS/nw-270/nw-270.png'
    camera_pos = nw_270(n_frames=f)
    axis_on = False

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
    logfile.write('Rendering quality ' + str(q) + '\n')
    
except ValueError:
    print('Not a number, default as 2')
    q = 2
# Camera properties
camera_props = {
    'proj_name': 'stereo',
    'fov': 110.,  # degrees
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
logfile.write('\n')
logfile.write('Image Rendering Time(s) \n')
logfile.close()