import sys
import os

def DirFound(arg, dir, files):
    for file in files:
        arg.append(str(os.path.join(dir, file)))

dir = sys.argv[1]
ref = sys.argv[2]

files = []
os.path.walk(dir, DirFound, files)
for file in files:
    os.system("python register_vgg.py %s %s" % (file, ref))
    
