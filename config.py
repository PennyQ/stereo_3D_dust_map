from camera_route import circle_local_left

# Output figure store place
pan1 = '.'

# Input file dir
# map_fname = '/n/fink1/ggreen/bayestar/output/allsky_2MASS/compact/dust-map-3d-uncompressed.h5'#compact_10samp.h5'
map_fname = 'data/AquilaSouthLarge2_unified.h5' 

# Initiate processing core numbers
n_procs = 10

# Misc settings
plot_props = {
    'fname': pan1 + '/3d/allsky_2MASS/AqS/AqS-loop-hq.png', #'3d/allsky_2MASS/grand-tour/simple-loop-att-v2-lq.png',
    'figsize': (10, 7),
    'dpi': 100,
    'n_averaged': 1,
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

# Camera path/orientation
#camera_pos = local_dust_path(n_frames=400)
#camera_pos = paper_renderings()
#camera_pos = Orion_flythrough(n_frames=400)
#camera_pos = grand_tour_path(n_frames=20)#1600)
# camera_pos = circle_local(n_frames=3, l_0=30., b_0=5.)
camera_pos = circle_local_left(n_frames=2, l_0=30., b_0=5.)
#camera_pos = stereo_pair()

#for key in camera_pos:
#    camera_pos[key] = camera_pos[key][:20]

# # TODO: Camera properties
camera_props = {
    'proj_name': 'stereo',
    'fov': 140.,
    'n_x': 200, # num of pixels
    'n_y': 140,
    'n_z': 500,
    'dr': 10./2,  # 10pc per step
    'z_0': 1., #(0., 0., 0.)
}


# General label properties
label_props = {
    'text_color': (0, 166, 255),#(255, 255, 255),
    'stroke_color': (9, 73, 92),#(0, 0, 0) #(192, 225, 235)
}
 