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
    optionParser = OptionParser("usage: %prog  -i INPUT_FILE -m META_FILE -t ADDITIONAL_TAG -I NUM_IMAGES")
    optionParser.add_option("-i", "--input", action="store", dest="input", type="string", metavar="FILE", help="pickled python dataset file")
    optionParser.add_option("-m", "--meta", action="store", dest="meta", type="string", metavar="FILE", help="pickled python metadata file")
    optionParser.add_option("-t", "--tag", action="store", dest="tag", type="int", help="selected tag to add the image", default = 0)
    optionParser.add_option("-I", "--images", action="store", dest="nimages", type="int", help="number of images to add", default = 0)

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

    cursor_point = 0
    for i in range(0, my_obj['data'].shape[0]):
        data_entry = my_obj['data'][i]
        tag_no = my_obj[label_str][i]
        img_array[cursor_point] = data_entry
        class_array[cursor_point] = tag_no
        num_img_base_array[tag_no]+=1
        cursor_point+=1

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
    test_img_array = numpy.array(test_array, dtype=numpy.uint8)

    rm_images_only = False
    img_to_remove = []
    if(options.nimages > 0):
        rm_images_only = True
        print "just removing images, not the full tag."
    else:
        rm_images_only = False

    tag_img_number = num_img_base_array[options.tag]
    img_of_tag = []
    for i in range(0, len(class_array)):
        if(class_array[i] == options.tag):
            img_of_tag.append(i)
    print "Data with selected tag: "+str(img_of_tag)+" ("+str(tag_img_number)+")"

    for j in range(0, len(img_of_tag)):
        i = (len(img_of_tag)-1)-j
        if(rm_images_only == False):
            img_to_remove.append(img_of_tag[i])
        else:
            if((tag_img_number-i)<=options.nimages):
                img_to_remove.append(img_of_tag[i])

    print "Data to remove: "+str(img_to_remove)

    #rm_img_array = None
    #rm_class_array = []
    print "Dataset size before deletion: "+str(len(class_array))+" | "+str(img_array.shape[0])
    for i in range(0, len(img_to_remove)):
        entry = img_to_remove[i]
        #print "removing "+str(entry)+" ..."
        del class_array[entry]
    
    if(rm_images_only == True):
        num_img_base_array[options.tag]-=len(img_to_remove)
    img_array = numpy.delete(img_array, img_to_remove, 0)
    print "Dataset size after deletion: "+str(len(class_array))+" | "+str(img_array.shape[0])

    print "Label dictionary before deletion: "+str(len(meta_obj[meta_label_str])) 
    if(rm_images_only == False):
        del num_img_base_array[options.tag]
        del meta_obj[meta_label_str][options.tag]
        # re-adapt mapping
        for i in range(0, len(class_array)):
            if(class_array[i] > options.tag):
                class_array[i]-=1
        
        #testing data
        del img_of_tag[:]
        for i in range(0, len(test_classes)):
            if(test_classes[i] == options.tag):
                img_of_tag.append(i)
        for i in range(0, len(img_of_tag)):
            j = (len(img_of_tag)-1)-i
            entry = img_of_tag[j]
            #print "removing "+str(entry)+" ..."
            del test_classes[entry]
        test_img_array = numpy.delete(test_img_array, img_of_tag, 0)
        # re-mapping test data
        for i in range(0, len(test_classes)):
            if(test_classes[i] > options.tag):
                test_classes[i]-=1
    print "Label dictionary after deletion: "+str(len(meta_obj[meta_label_str]))

    out_obj = dict()
    out_obj['data']=img_array
    out_obj['labels']=class_array
    out_dir = os.path.dirname(options.input)
    #fo = open(os.path.join(options.output,"experiment"), 'wb')
    cPickle.dump(out_obj, open(os.path.join(out_dir,"experiment_rm"), "wb"), protocol=2)

    test_n_obj = dict()
    test_n_obj['data']=test_img_array
    test_n_obj['labels']=test_classes
    cPickle.dump(test_n_obj, open(os.path.join(out_dir,"test_rm"), "wb"), protocol=2)

    meta_obj_out = dict()
    meta_obj_out['label_names'] = meta_obj[meta_label_str]
    cPickle.dump(meta_obj_out, open(os.path.join(out_dir,"meta_rm"), "wb"), protocol=2)


