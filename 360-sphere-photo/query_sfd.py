from __future__ import print_function
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery

coords = SkyCoord('12h30m25.3s', '15d15m58.1s', frame='icrs')
sfd = SFDQuery()
ebv = sfd(coords)

print('E(B-V) = {:.3f} mag'.format(ebv))