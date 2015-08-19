'''
Created on Apr 15, 2015

@author: christian
'''

from optparse import OptionParser
import numpy

import os
from os import listdir
from os.path import isfile, join
import cPickle
import glob

def unpickle(file):
    fo = open(file, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary

if __name__ == '__main__':
    optionParser = OptionParser("usage: %prog [options] INPUT_FILE OUTPUT_FILE [WITH_ROTATION]")
    optionParser.add_option("-i", "--input", action="store", dest="input", type="string", metavar="FILE", help="pickled python dataset folder")
    optionParser.add_option("-o", "--output", action="store", dest="output", type="string", metavar="FILE", help="output experiment dataset folder")
    optionParser.add_option("-C", "--classes", action="store", dest="nclasses", type="int", help="number of classes (all, if omitted)", default = 100)
    optionParser.add_option("-I", "--images", action="store", dest="nimages", type="int", help="number of images per class", default = 10)
    
    
    #optionParser.add_option("-i", "--input-file", action="store", dest="inputFile", type="string", metavar="FILE", help="additional pickled python dataset (no addition, if omitted)")
    #optionParser.add_option("-R", "--RISCAN", action="store_true", dest="transform", help="add (true) or remove (true) tags", default=False)
    #optionParser.add_option("-Z", "--offset_z", action="store", dest="offset_z", type="float", help="number of images per class added", default = 0.0)
    #optionParser.add_option("-Z", "--offset_z", action="store", dest="offset_z", type="float", help="tag number to add", default = 0.0)

    (options, args) = optionParser.parse_args()

    num_img_base_array = [0]*(options.nclasses)
    img_base_array = [[0]*3072]*(options.nclasses*options.nimages)
    img_array = numpy.array(img_base_array, dtype=numpy.uint8)
    class_array = [0]*(options.nclasses*options.nimages)
    
    test_img_array = None
    test_array = []
    test_classes = []

    if options.output==None:
        exit()
    
    if options.input != None:
        #onlyfiles = [ f for f in listdir(options.input) if isfile(join(options.input,f)) ]
        onlyfiles = glob.glob(options.input+os.path.sep+"data_batch*")
        cursor_point = 0
        #for entry in onlyfiles:
        for k in range(0, len(onlyfiles)):
            entry = onlyfiles[k]
            print "Processing file "+entry+" ..."
            #my_obj = dict()
            my_obj = unpickle(entry)
            #print my_obj
            print "Data fieldsize: "+str(my_obj['data'].shape[0])
            for i in range(0, my_obj['data'].shape[0]):
                data_entry = my_obj['data'][i]
                tag_no = my_obj['labels'][i]
                #print "Current tagnumber: "+str(tag_no)
                if (tag_no < options.nclasses) and (num_img_base_array[tag_no]<options.nimages):
                    print "Image: "+str(i)+" => Tag: "+str(tag_no)
                    img_array[cursor_point] = data_entry
                    class_array[cursor_point] = tag_no
                    num_img_base_array[tag_no]+=1
                    cursor_point+=1
    else:
        exit()

    if options.input != None:
        #onlyfiles = [ f for f in listdir(options.input) if isfile(join(options.input,f)) ]

        tcursor_point = 0
        entry = os.path.join(options.input,"test_batch")
        print "Processing file "+entry+" ..."
        #my_obj = dict()
        test_obj = unpickle(entry)
        print "Test Data fieldsize: "+str(test_obj['data'].shape[0])
        #test_img_array = [[0]*3072]*(test_obj['data'].shape[0])
        for i in range(0, test_obj['data'].shape[0]):
            data_entry = test_obj['data'][i]
            tag_no = test_obj['labels'][i]
            if (tag_no < options.nclasses):
                print "Test Image: "+str(i)+" => Tag: "+str(tag_no)
                #test_img_array[tcursor_point] = data_entry
                #class_array[cursor_point] = tag_no
                test_array.append(data_entry.tolist())
                test_classes.append(tag_no)
                tcursor_point+=1
        test_img_array = numpy.array(test_array, dtype=numpy.uint8)
    else:
        exit()

    meta_filenames = []
    meta_filenames += glob.glob(options.input+os.path.sep+"batches.meta*")
    #meta_filenames += glob.glob(options.input+os.path.sep+"*.meta*")
    meta_obj = unpickle(meta_filenames[0])
    label_names = meta_obj['label_names']
    label_names_new = []
    for i in range(0, max(options.nclasses, len(label_names))):
        tag_name = label_names[i]
        print "Tag: "+str(i)+" => Name: "+tag_name
        label_names_new.append(tag_name)
    
    out_obj = dict()
    out_obj['data']=img_array
    out_obj['labels']=class_array
    #fo = open(os.path.join(options.output,"experiment"), 'wb')
    cPickle.dump(out_obj, open(os.path.join(options.output,"experiment"), "wb"), protocol=2)
    
    test_n_obj = dict()
    test_n_obj['data']=test_img_array
    test_n_obj['labels']=test_classes
    cPickle.dump(test_n_obj, open(os.path.join(options.output,"test"), "wb"), protocol=2)
    
    meta_obj_out = dict()
    meta_obj_out['label_names'] = label_names_new
    cPickle.dump(meta_obj_out, open(os.path.join(options.output,"meta"), "wb"), protocol=2)