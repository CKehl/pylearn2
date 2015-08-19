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
    optionParser = OptionParser("usage: %prog -i INPUT_FILE -m META_FILE -t SOURCE_TAG -r DESTINATION_TAG")
    optionParser.add_option("-i", "--input", action="store", dest="input", type="string", metavar="FILE", help="pickled python dataset file")
    optionParser.add_option("-m", "--meta", action="store", dest="meta", type="string", metavar="FILE", help="pickled python metadata file")
    optionParser.add_option("-t", "--tag", action="store", dest="tag", type="int", help="selected tag to add the image", default = 0)
    optionParser.add_option("-r", "--replace_tag", action="store", dest="rtag", type="int", help="replacement tag number for <tag>", default = 0)

    (options, args) = optionParser.parse_args()
    
    my_obj = dict()
    meta_obj = dict()
    label_str = ''
    meta_label_str = ''
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
    
    ds = 0
    if("fine_labels" in my_obj.keys()):
        ds = 1 #CIFAR-100
        label_str = 'fine_labels'
        meta_label_str = 'fine_label_names'
    else:
        ds = 0 #CIFAR-10 and combined
        label_str = 'labels'
        meta_label_str = 'label_names'

    meta_inputs = []
    if(options.meta == None) or (options.meta == ""):
        meta_inputs = glob.glob(os.path.dirname(options.input)+os.path.sep+"*meta*")
    else:
        meta_inputs.append(options.meta)
    meta_obj = unpickle(meta_inputs[0])


    num_img_base_array = [0]*(len(my_obj[label_str]))
    img_base_array = [[0]*3072]*(my_obj['data'].shape[0])
    img_array = numpy.array(img_base_array, dtype=numpy.uint8)
    class_array = [0]*(my_obj['data'].shape[0])

    for i in range(0, my_obj['data'].shape[0]):
        data_entry = my_obj['data'][i]
        tag_no = my_obj[label_str][i]
        img_array[i] = data_entry
        class_array[i] = tag_no
        num_img_base_array[tag_no]+=1

    # Test array generation
    tcursor_point = 0
    #print "Test Data fieldsize: "+str(test_obj['data'].shape[0])
    for i in range(0, test_obj['data'].shape[0]):
        data_entry = test_obj['data'][i]
        tag_no = test_obj['labels'][i]
        #print "Test Image: "+str(i)+" => Tag: "+str(tag_no)
        test_array.append(data_entry.tolist())
        test_classes.append(tag_no)
        tcursor_point+=1

    tag_img_number = num_img_base_array[options.tag]
    img_of_tag = []
    for i in range(0, len(class_array)):
        if(class_array[i] == options.tag):
            img_of_tag.append(i)
    print "Data with selected tag: "+str(img_of_tag)+" ("+str(tag_img_number)+")"

    print "Dataset size before replacement: "+str(len(class_array))+" | "+str(img_array.shape[0])
    for i in range(0, len(img_of_tag)):
        class_array[img_of_tag[i]]=options.rtag
    del num_img_base_array[options.tag]
    print "Dataset size after replacement: "+str(len(class_array))+" | "+str(img_array.shape[0])

    print "Label dictionary before replacement: "+str(len(meta_obj[meta_label_str]))   
    del meta_obj[meta_label_str][options.tag]
    print "Label dictionary after replacement: "+str(len(meta_obj[meta_label_str]))

    # re-adapt mapping
    for i in range(0, len(class_array)):
        if(class_array[i] > options.tag):
            class_array[i]-=1

    ################
    # TESTING DATA #
    ################
    del img_of_tag[:]
    for i in range(0, len(test_classes)):
        if(test_classes[i] == options.tag):
            img_of_tag.append(i)
    for i in range(0, len(img_of_tag)):
        test_classes[img_of_tag[i]]=options.rtag
    # re-adapt mapping
    for i in range(0, len(test_classes)):
        if(test_classes[i] > options.tag):
            test_classes[i]-=1

    out_obj = dict()
    out_obj['data']=img_array
    out_obj['labels']=class_array
    out_dir = os.path.dirname(options.input)
    #fo = open(os.path.join(options.output,"experiment"), 'wb')
    cPickle.dump(out_obj, open(os.path.join(out_dir,"experiment_rp"), "wb"), protocol=2)

    test_n_obj = dict()
    test_img_array = numpy.array(test_array, dtype=numpy.uint8)
    test_n_obj['data']=test_img_array
    test_n_obj['labels']=test_classes
    cPickle.dump(test_n_obj, open(os.path.join(out_dir,"test_rp"), "wb"), protocol=2)

    meta_obj_out = dict()
    meta_obj_out['label_names'] = meta_obj[meta_label_str]
    cPickle.dump(meta_obj_out, open(os.path.join(out_dir,"meta_rp"), "wb"), protocol=2)


