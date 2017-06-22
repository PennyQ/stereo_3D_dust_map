from libxmp.utils import XMPFiles
fn=raw_input('Input image file:')
xmpfile=XMPFiles(file_path=str(fn))
xmp=xmpfile.get_xmp()
print(xmp)
xmpfile.close_file()