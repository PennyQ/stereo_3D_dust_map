s = """<rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">
 <GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer> 
 <GPano:CaptureSoftware>Photo Sphere</GPano:CaptureSoftware> 
 <GPano:StitchingSoftware>Photo Sphere</GPano:StitchingSoftware> 
 <GPano:ProjectionType>equirectangular</GPano:ProjectionType> 
 <GPano:PoseHeadingDegrees>350.0</GPano:PoseHeadingDegrees> 
 <GPano:InitialViewHeadingDegrees>90.0</GPano:InitialViewHeadingDegrees> 
 <GPano:InitialViewPitchDegrees>0.0</GPano:InitialViewPitchDegrees> 
 <GPano:InitialViewRollDegrees>0.0</GPano:InitialViewRollDegrees> 
 <GPano:InitialHorizontalFOVDegrees>75.0</GPano:InitialHorizontalFOVDegrees> 
 <GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels> 
 <GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels> 
 <GPano:CroppedAreaImageWidthPixels>4000</GPano:CroppedAreaImageWidthPixels> 
 <GPano:CroppedAreaImageHeightPixels>2000</GPano:CroppedAreaImageHeightPixels>
 <GPano:FullPanoWidthPixels>4000</GPano:FullPanoWidthPixels> 
 <GPano:FullPanoHeightPixels>2000</GPano:FullPanoHeightPixels> 
 <GPano:FirstPhotoDate>2012-11-07T21:03:13.465Z</GPano:FirstPhotoDate> 
 <GPano:LastPhotoDate>2012-11-07T21:04:10.897Z</GPano:LastPhotoDate> 
 <GPano:SourcePhotosCount>50</GPano:SourcePhotosCount> 
 <GPano:ExposureLockUsed>False</GPano:ExposureLockUsed> 
</rdf:Description> 
""" 

s2 = """<rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">
 <GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer> 
</rdf:Description> 
""" 

import datetime
from libxmp import XMPFiles, XMPMeta 
import pytz

try:
    xmpfile = XMPFiles(file_path=raw_input('image file path:'), open_forupdate=True)
except:
    print('file path invalid!')
    exit()
# xmp = XMPMeta() 
xmp = xmpfile.get_xmp()
if xmp is None:
    xmp = XMPMeta() 
GPANO = 'GPano'
NS_GPANO = 'http://ns.google.com/photos/1.0/panorama/'
ns = xmp.register_namespace(NS_GPANO, GPANO) 
print(ns)
xmp.get_prefix_for_namespace(NS_GPANO) 
xmp.set_property_bool(NS_GPANO, 'UsePanaramaViewer', True)
xmp.set_property(NS_GPANO, 'CaptureSoftware', 'Photo Sphere')
xmp.set_property(NS_GPANO, 'StichingSoftware', 'Photo Sphere')
xmp.set_property(NS_GPANO, 'ProjectionType', 'equirectangular') 
xmp.set_property_float(NS_GPANO, 'PoseHeadingDegrees', 350.0) 
xmp.set_property_float(NS_GPANO, 'InitialViewHeadingDegrees', 90.0) 
xmp.set_property_float(NS_GPANO, 'InitialViewPitchDegrees', 0.0) 
xmp.set_property_float(NS_GPANO, 'InitialViewRollDegrees', 0.0) 
xmp.set_property_float(NS_GPANO, 'InitialHorizontalFOVDegrees', 75.0) 
xmp.set_property_int(NS_GPANO, 'CroppedAreaLeftPixels', 0) 
xmp.set_property_int(NS_GPANO, 'CroppedAreaTopPixels', 0) 
xmp.set_property_int(NS_GPANO, 'CroppedAreaImageWidthPixels', 5100) 
xmp.set_property_int(NS_GPANO, 'CroppedAreaImageHeightPixels', 2550)
xmp.set_property_int(NS_GPANO, 'FullPanoWidthPixels', 5100) 
xmp.set_property_int(NS_GPANO, 'FullPanoHeightPixels', 2550) 

dt = datetime.datetime(2012, 11, 7, 21, 3, 13, 465000, tzinfo=pytz.utc)
xmp.set_property_datetime(NS_GPANO, 'FirstPhotoDate', dt)

dt = datetime.datetime(2012, 11, 7, 21, 4, 10, 897000, tzinfo=pytz.utc)
xmp.set_property_datetime(NS_GPANO, 'LastPhotoDate', dt)

xmp.set_property_int(NS_GPANO, 'SourcePhotosCount', 50) 
xmp.set_property_bool(NS_GPANO, 'ExposureLockUsed', False) 

print(xmp)
xmpfile.can_put_xmp(xmp)
xmpfile.put_xmp(xmp)
xmpfile.close_file()
print('Done!')