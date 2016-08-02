#!/usr/bin/python
import os
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def logfile(stringtolog,filename):
    fileobject=open(filename,'a')
    fileobject.write(stringtolog)
    fileobject.write('\n')
    fileobject.close()

def flatten(alist):
    return [item for sublist in alist for item in sublist]

def main():
    alist=[[1,2,3],[4,5,6],[7]]
    print("Should return a flattened list if flatten is working properly: ", flatten(alist))

if __name__=="__main__":
    main()
