#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  point_rendering.py
#  
#  Copyright 2014 greg <greg@greg-UX301LAA>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np
import scipy, scipy.misc

from alpha import blend_images

from PIL import Image, ImageDraw

def rasterize_point(canvas, center, radius=5,
                                    color=(0,0,0),
                                    outline=(0,0,0,0),
                                    oversample=1):
    
    if isinstance(radius, int) or isinstance(radius, float):
        radius = (radius, radius)
    
    # Determine oversampled image size
    canvas_large = [oversample*c for c in canvas]
    
    xy = np.array([center[0] - radius[0], center[1] - radius[1],
                   center[0] + radius[0], center[1] + radius[1]])
    xy = np.round(oversample * xy).astype('u4').tolist()
    
    # Set up canvas and image renderer
    image = Image.new('RGBA', canvas_large, (1,1,1,0))
    d = ImageDraw.Draw(image)
    
    # Draw circle
    d.ellipse(xy, fill=color, outline=outline)
    
    # Downsample image
    if oversample != 1:
        image = image.resize(canvas, Image.ANTIALIAS)
    
    # Convert to numpy array
    image = np.array(image).astype('f8') / 255.
    
    image = np.swapaxes(image, 0, 1)
    
    return image


def test_point():
    import matplotlib.pyplot as plt
    
    canvas = (500, 300)
    radius = (150, 100)
    center = (250, 150)
    fill = (0, 100, 0)
    
    img = rasterize_point(canvas, center, radius=radius, color=fill)
    bg = np.ones(img.shape, dtype='f8')
    img = blend_images(bg, img)
    
    figsize = (c/100. for c in canvas)
    
    fig = plt.figure(figsize=figsize, dpi=120)
    ax = fig.add_subplot(1,1,1)
    
    ax.imshow(img[:,:,:3], aspect='equal', interpolation='none')
    
    fig.savefig('test.png', dpi=120)
    plt.show()


def main():
    test_point()
    
    return 0

if __name__ == '__main__':
    main()
