import sys
import os
import re
from os.path import basename


def DirFound(arg, dir, files):
    for file in files:
        arg.append(str(os.path.join(dir, file)))

dir = sys.argv[1]
out = sys.argv[2]
#pfx = sys.argv[3]

files = []
os.path.walk(dir, DirFound, files)
for file in files: 
    numstr = re.search('\\d+', basename(file)).group(0)
    filenum = int(numstr)
    ext = basename(file)[-3:]
    pfx = re.search('[^(\d+.%s)]+.' % ext, basename(file)).group(0)
    fmt = '%%0%dd.%s' % (len(numstr), ext)
    if filenum > 0 and filenum <= 50:
        filename = dir + '/' + pfx + fmt % filenum
        refname = dir + '/' + pfx + fmt % (filenum - 1)
        os.system("python register_vgg.py %s %s" % (filename, refname))
    
