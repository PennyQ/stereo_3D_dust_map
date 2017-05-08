#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  font_rendering.py
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

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import scipy, scipy.misc

from alpha import blend_images


def rasterize_text(canvas, txt, pos, font='fonts/cmunbx.ttf',
                                     fontsize=24, fontcolor=(0,0,0),
                                     stroke_width=0, stroke_color=(255,255,255),
                                     stroke_angles=16, oversample=1,
                                     ha='center', va='center'):
    
    #fontcolor = [int(np.round(255.*f)) for f in fontcolor]
    fontsize = int(round(oversample * fontsize))
    stroke_width = oversample * stroke_width
    
    #tmp =  '\n --> Text: ' + txt.encode('utf-8').strip()
    #tmp += '\n     stroke_width: %.3f' % stroke_width
    #tmp += '\n'
    #print tmp
    
    # Determine oversampled image size
    canvas_large = [oversample*c for c in canvas]
    pos_large = np.array([oversample*p for p in pos]).astype('f8')
    
    # Set up canvas and image renderer
    image = Image.new('RGBA', canvas_large, (1,1,1,0))
    img_font = ImageFont.truetype(font, fontsize)
    d = ImageDraw.Draw(image)
    d.fontmode = 'L'
    
    # Handle alignment
    w, h = d.textsize(txt, font=img_font)
    w_off, h_off = img_font.getoffset(txt)
    w -= w_off
    h -= h_off
    
    if ha == 'left':
        offset = [0]
    elif ha == 'center':
        offset = [-w/2.]
    elif ha == 'right':
        offset = [-w]
    else:
        raise ValueError('Unrecognized horizontal alignment: "%s"' % ha)
    
    if va == 'top':
        offset.append(0)
    elif va == 'center':
        offset.append(-h/2.)
    elif va == 'bottom':
        offset.append(-h)
    else:
        raise ValueError('Unrecognized vertical alignment: "%s"' % va)
    
    #offset[1] -= 0.35 * h # This shouldn't be necessary, but it is
    offset[0] -= w_off
    offset[1] -= h_off
    
    pos_large += offset
    
    # Render text
    x, y = pos_large
    d.text((x, y), txt, (0,0,0), font=img_font)
    
    image = np.array(image).astype('f8') / 255.
    
    # Stroke text
    bg = None
    
    if stroke_width > 1.e-5:
        theta = np.linspace(0., 2.*np.pi, stroke_angles)[::-1]
        dx = np.round(np.cos(theta) * stroke_width).astype('i4')
        dy = np.round(np.sin(theta) * stroke_width).astype('i4')
        
        bg = np.zeros(image.shape, dtype='f8')
        
        for x,y in zip(dx, dy):
            if (x == 0) and (y == 0):
                continue
            shift = np.roll(np.roll(image, x, axis=0), y, axis=1)
            bg = blend_images(bg, shift, limit_alpha=True)
        
        for k in xrange(3):
            bg[:,:,k] = stroke_color[k] / 255.
        
        idx = image[:,:,3] < 1. - 1.e-10
        
        if len(stroke_color) == 4:
            bg[:,:,3] *= stroke_color[3] / 255.
        
        bg[:,:,3] *= idx.astype('f8')
        #image[:,:,3] *= (~idx).astype('f8')
    
    for k in xrange(3):
        image[:,:,k] = fontcolor[k] / 255.
    
    if len(fontcolor) == 4:
        image[:,:,3] *= fontcolor[3] / 255.
    
    if bg != None:
        image = blend_images(bg, image)
    
    # Downsample image
    if oversample != 1:
        image = (255. * image).astype('i4')
        image = scipy.misc.imresize(image, canvas[::-1])
        #image = image.resize(canvas, Image.ANTIALIAS)
        image = image.astype('f8') / 255.
    
    image = np.swapaxes(image, 0, 1)
    
    return image


def test_PIL():
    import matplotlib.pyplot as plt
    
    font = 'fonts/cmunbx.ttf'
    fontsize = 200
    
    bg = np.ones((800, 500, 4), dtype='f8')
    
    img1 = rasterize_text((800, 500), 'Sol', (0, 0),
                               font='fonts/cmunss.ttf',
                               fontsize=fontsize,
                               fontcolor=(0, 255, 0, 255),
                               ha='left', va='top',
                               stroke_width=8,
                               oversample=2,
                               stroke_color=(0, 0, 0, 20))
    
    img2 = rasterize_text((800, 500), u'λ = 10', (800, 500),
                               font=font, fontsize=fontsize,
                               fontcolor=(0, 0, 0, 255),
                               ha='right', va='bottom',
                               stroke_width=3, stroke_color=(0, 0 , 255, 100))
    
    print 'bg.shape =', bg.shape
    print 'img1.shape =', img1.shape
    print 'img2.shape =', img2.shape
    
    fig = plt.figure(figsize=(8,5), dpi=120)
    ax = fig.add_subplot(1,1,1)
    
    print np.max(img1)
    print np.max(img2)
    
    img = blend_images(bg, img1)
    img = blend_images(img, img2)
    
    img = img.swapaxes(0,1)
    
    ax.imshow(img, aspect='equal', interpolation='none')
    
    fig.savefig('test_font.png', dpi=120)
    plt.show()


def main():
    test_PIL()
    
    return 0

if __name__ == '__main__':
    main()

