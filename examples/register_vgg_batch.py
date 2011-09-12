import sys
import os
from os.path import basename


def DirFound(arg, dir, files):
    for file in files:
        arg.append(str(os.path.join(dir, file)))

dir = sys.argv[1]
#ref = sys.argv[2]

files = []
os.path.walk(dir, DirFound, files)
for file in files:
    filenum = int(basename(file)[5:8])
    if filenum > 0 and filenum <= 50:
        filename = dir + '/%08d.jpg.tiff.cropped' % filenum
        refname = dir + '/%08d.jpg.tiff.cropped' % (filenum - 1)
        os.system("python register_vgg.py %s %s" % (filename, refname))
    
