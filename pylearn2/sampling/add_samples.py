'''
Created on Apr 16, 2015

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
    optionParser = OptionParser("usage: %prog -i INPUT_FILE -m META_FILE -a ADDITIONAL_FILE -t ADDITIONAL_TAG -I NUM_IMAGES")
    optionParser.add_option("-i", "--input", action="store", dest="input", type="string", metavar="FILE", help="pickled python dataset file")
    optionParser.add_option("-m", "--meta", action="store", dest="meta", type="string", metavar="FILE", help="pickled python metadata file (meaning of classes)")
    optionParser.add_option("-a", "--additional", action="store", dest="additional", type="string", metavar="FILE", help="additional pickled python dataset")
    #optionParser.add_option("-o", "--output", action="store", dest="output", type="string", metavar="FILE", help="output experiment dataset file")
    optionParser.add_option("-t", "--tag", action="store", dest="tag", type="int", help="selected tag to add the image", default = 0)
    optionParser.add_option("-I", "--images", action="store", dest="nimages", type="int", help="number of images to add", default = 1)

    
    (options, args) = optionParser.parse_args()

    my_obj = dict()
    add_obj = dict()
    add_img_only = False
    add_tag_no = 0
    img_before_addition = 0

    test_img_array = None
    test_array = []
    test_classes = []
    test_obj = dict()
    
    if options.input != None:
        my_obj = unpickle(options.input)
        in_dir = os.path.dirname(options.input)
        test_obj = unpickle(os.path.join(in_dir, "test"))
    else:
        exit()

    if options.additional != None:
        add_obj = unpickle(options.additional)
    else:
        exit()

    num_img_base_array = [0]*(len(my_obj['labels']))
    img_base_array = [[0]*3072]*(my_obj['data'].shape[0] + options.nimages)
    img_array = numpy.array(img_base_array, dtype=numpy.uint8)
    class_array = [0]*(my_obj['data'].shape[0] + options.nimages)

    cursor_point = 0
    for i in range(0, my_obj['data'].shape[0]):
        data_entry = my_obj['data'][i]
        tag_no = my_obj['labels'][i]
        img_array[cursor_point] = data_entry
        class_array[cursor_point] = tag_no
        num_img_base_array[tag_no]+=1
        cursor_point+=1

    # Test array generation
    tcursor_point = 0
    print "Test Data fieldsize: "+str(test_obj['data'].shape[0])
    for i in range(0, test_obj['data'].shape[0]):
        data_entry = test_obj['data'][i]
        tag_no = test_obj['labels'][i]
        #print "Test Image: "+str(i)+" => Tag: "+str(tag_no)
        test_array.append(data_entry.tolist())
        test_classes.append(tag_no)
        tcursor_point+=1
    


    meta_inputs = []
    if(options.meta == None) or (options.meta == ""):
        meta_inputs = glob.glob(os.path.dirname(options.input)+os.path.sep+"*meta*")
    else:
        meta_inputs.append(options.meta)
    meta_filename = meta_inputs[0]
    meta_obj = unpickle(meta_filename)
    label_names = []
    label_names = meta_obj['label_names']

    meta_add_filename = glob.glob(os.path.dirname(options.additional)+os.path.sep+"*meta*")[0]
    meta_add_obj = unpickle(meta_add_filename)
    add_label_names = meta_add_obj['fine_label_names']
    
    test_add_filename = os.path.join(os.path.dirname(options.additional),"test")
    test_add_obj = dict()
    
    #---------------------------------------------------------------
    #   Handle adding images to a class that is already present
    #---------------------------------------------------------------    
    if(add_label_names[options.tag] in label_names):
        add_img_only = True
        add_tag_no = label_names.index(add_label_names[options.tag])
        img_before_addition = num_img_base_array[add_tag_no]
    else:
        add_img_only = False
        num_img_base_array.append(0)
        label_names.append(add_label_names[options.tag])
        add_tag_no = len(label_names)-1
        # Add test data
        test_add_obj = unpickle(test_add_filename)
    
    for i in range(0, add_obj['data'].shape[0]):
        data_entry = add_obj['data'][i]
        tag_no = add_obj['fine_labels'][i]
        if (tag_no == options.tag) and (num_img_base_array[add_tag_no] < (img_before_addition+options.nimages)):
            img_array[cursor_point] = data_entry
            class_array[cursor_point] = add_tag_no
            num_img_base_array[add_tag_no]+=1
            cursor_point+=1

    if(add_img_only == False):
        for i in range(0, test_add_obj['data'].shape[0]):
            tag_no = test_add_obj['fine_labels'][i]
            if (tag_no == options.tag):
                test_array.append(test_add_obj['data'][i].tolist())
                test_classes.append(add_tag_no)
                tcursor_point+=1
                

    test_img_array = numpy.array(test_array, dtype=numpy.uint8)

    out_obj = dict()
    out_obj['data']=img_array
    out_obj['labels']=class_array
    out_dir = os.path.dirname(options.input)
    #fo = open(os.path.join(options.output,"experiment"), 'wb')
    cPickle.dump(out_obj, open(os.path.join(out_dir,"experiment_add"), "wb"), protocol=2)

    test_n_obj = dict()
    test_n_obj['data']=test_img_array
    test_n_obj['labels']=test_classes
    cPickle.dump(test_n_obj, open(os.path.join(out_dir,"test_add"), "wb"), protocol=2)

    meta_obj_out = dict()
    meta_obj_out['label_names'] = label_names
    cPickle.dump(meta_obj_out, open(os.path.join(out_dir,"meta_add"), "wb"), protocol=2)
    
    
