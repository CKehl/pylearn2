'''
Created on Apr 18, 2015

@author: christian
'''

from optparse import OptionParser
import numpy

import os
from os import listdir
from os.path import isfile, join
import cPickle
import glob
from numpy import dtype

def unpickle(file):
    fo = open(file, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary

if __name__ == '__main__':
    optionParser = OptionParser("usage: %prog [options] INPUT_FILE")
    optionParser.add_option("-i", "--input", action="store", dest="input", type="string", metavar="FILE", help="pickled python dataset file")
    optionParser.add_option("-m", "--meta", action="store", dest="meta", type="string", metavar="FILE", help="pickled python metadata file")

    (options, args) = optionParser.parse_args()
    
    my_obj = dict()
    meta_obj = dict()
    

    if options.input != None:
        my_obj = unpickle(options.input)
    else:
        exit()

    print my_obj.keys()
    
    ds = 0
    if("fine_labels" in my_obj.keys()):
        ds = 1 #CIFAR-100
    else:
        ds = 0 #CIFAR-10 and combined

    meta_inputs = []
    if(options.meta == None) or (options.meta == ""):
        meta_inputs = glob.glob(os.path.dirname(options.input)+os.path.sep+"*meta*")
    else:
        meta_inputs.append(options.meta)
    meta_obj = unpickle(meta_inputs[0])
    print meta_obj.keys()
    
    if(ds == 1):
        print "Dataset size: "+str(len(my_obj['fine_labels']))
        print "Labels:"
        total = 0
        for i in range(0, len(meta_obj['coarse_label_names'])):
            print "Tag "+str(i)+"(C), "+str(total)+"(B)"+" => "+meta_obj['coarse_label_names'][i]
            total+=1
        for i in range(0, len(meta_obj['fine_label_names'])):
            print "Tag "+str(i)+"(F), "+str(total)+"(B)"+" => "+meta_obj['fine_label_names'][i]
            total+=1
        
        inverse_map = []
        for i in range(len(meta_obj['fine_label_names'])):
            current_map = []
            for j in range(len(my_obj['fine_labels'])):
                if my_obj['fine_labels'][j] == i:
                    current_map.append(j)
            inverse_map.append(current_map)
        print "Inverse Tag Map: "
        #print numpy.array(inverse_map, dtype=numpy.uint8)
        for i in range (len(inverse_map)):
            print str(inverse_map[i])+" ("+str(len(inverse_map[i]))+")"
        #print inverse_map
    elif(ds == 0):
        print "Dataset size: "+str(len(my_obj['labels']))
        print "Labels:"
        for i in range(0, len(meta_obj['label_names'])):
            print "Tag "+str(i)+" => "+meta_obj['label_names'][i]

        inverse_map = []
        for i in range(len(meta_obj['label_names'])):
            current_map = []
            for j in range(len(my_obj['labels'])):
                if my_obj['labels'][j] == i:
                    current_map.append(j)
            inverse_map.append(current_map)
        print "Inverse Tag Map: "
        #print numpy.array(inverse_map, dtype=numpy.uint8)
        for i in range (len(inverse_map)):
            print str(inverse_map[i])+" ("+str(len(inverse_map[i]))+")"
        #print inverse_map

