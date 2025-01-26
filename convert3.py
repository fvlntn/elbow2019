from xml.dom import minidom
import os
import glob
import random
from PIL import Image
import csv
import numpy as np
import json
import math
import cv2

lut={}
lut["Elbow"]     =0


def convert_coordinates(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_xml2yolo( lut ):

    for fname in glob.glob("*.xml"):
        
        xmldoc = minidom.parse(fname)
        
        fname_out = (fname[:-4]+'.txt')

        with open(fname_out, "w") as f:

            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                classid =  (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])
                else:
                    label_str = "-1"
                    print ("warning: label '%s' not in look-up table" % classid)

                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))


        print ("wrote %s" % fname_out)

def getimage():    
    with open('links.csv', newline='') as csvf:
        csvr = csv.reader(csvf, delimiter=',')
        i = 0
        for row in csvr:                
            i = i+1 
            set = row[0].split('/')[0]      
            imagepath = '/home/frvalent/dev/RAID/convertwronglabels/' + set + '/crop512-megafix/P_' + str(i) + '.png'
            outpath = '/home/frvalent/dev/RAID/convertwronglabels/' + set + '/crop512-megafix/P_' + str(i) + '.png'
            try:
                img = cv2.imread(imagepath)
                if(img is not None):
                    print('mdr')
                    cv2.imwrite(outpath, img)                    
            except FileNotFoundError: 
                pass
            
            imagepath = '/home/frvalent/dev/RAID/convertwronglabels/' + set + '/crop512-megafix/N_' + str(i) + '.png'
            outpath = '/home/frvalent/dev/RAID/convertwronglabels/' + set + '/crop512-megafix/N_' + str(i) + '.png'
            try:
                img = cv2.imread(imagepath)
                if(img is not None):
                    print('mdr2')
                    cv2.imwrite(outpath, img)                    
            except FileNotFoundError: 
                pass    
            
def write_train(test):           

    with open('train.txt', "w") as train:
        
        with open('valid.txt', "w") as valid:  
                           
            for fname in glob.glob("*.xml"):

                fname_real = 'data/custom/images/' + fname[:-4] + '.png'
                
                r = random.uniform(0,1)     
                if(r <= test):
                    train.write(fname_real + '\n')
                else:  
                    valid.write(fname_real + '\n')
                                    
def get_detectron2_json():    
    with open('links.csv', newline='') as csvf:
        csvr = csv.reader(csvf, delimiter=',')
        i = 0
        region_data_train = {}
        region_data_valid = {}
        
        for row in csvr:
            i = i + 1         
            prefix = 'N'
            if('positive' in row[0]):
                prefix = 'P'
            set = row[0].split('/')[0]
            name = prefix + '_' + str(i)
                       
            if set == 'train':                
                region_data_train[str(name)] = {}
                region_data_train[str(name)]['fileref'] = ""
                region_data_train[str(name)]['size'] = ""
                region_data_train[str(name)]['filename'] = str(name) + '.png'
                region_data_train[str(name)]['base64_img_data'] = ""
                region_data_train[str(name)]['file_attributes']= {}
                region_data_train[str(name)]['regions'] = {}
            else:
                region_data_valid[str(name)] = {}
                region_data_valid[str(name)]['fileref'] = ""
                region_data_valid[str(name)]['size'] = ""
                region_data_valid[str(name)]['filename'] = str(name) + '.png'
                region_data_valid[str(name)]['base64_img_data'] = ""
                region_data_valid[str(name)]['file_attributes']= {}
                region_data_valid[str(name)]['regions'] = {}
           
            
            xmlpath = '/home/frvalent/dev/RAID/label-from-elbow/labelsxml/' + prefix + '_' + str(i) + '.xml'  
            xmldoc = minidom.parse(xmlpath)

            itemlist = xmldoc.getElementsByTagName('object')
            
            imagepath = '/home/frvalent/dev/RAID/label-from-elbow/' + set + '/' + prefix + '_' + str(i) + '.png'
            j = 0
            for item in itemlist:                           
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                
                if set == 'train':
                    region_data_train[str(name)]['regions'][str(j)] = {}
                    region_data_train[str(name)]['regions'][str(j)]['shape_attributes'] = {}
                    region_data_train[str(name)]['regions'][str(j)]['shape_attributes']['name'] = 'bounding_box'
                    region_data_train[str(name)]['regions'][str(j)]['shape_attributes']['all_points_x'] = [int(xmin), int(xmax)]
                    region_data_train[str(name)]['regions'][str(j)]['shape_attributes']['all_points_y'] = [int(ymin), int(ymax)]
                    region_data_train[str(name)]['regions'][str(j)]['regions_attributes'] = {}
                else: 
                    region_data_valid[str(name)]['regions'][str(j)] = {}
                    region_data_valid[str(name)]['regions'][str(j)]['shape_attributes'] = {}
                    region_data_valid[str(name)]['regions'][str(j)]['shape_attributes']['name'] = 'bounding_box'
                    region_data_valid[str(name)]['regions'][str(j)]['shape_attributes']['all_points_x'] = [int(xmin), int(xmax)]
                    region_data_valid[str(name)]['regions'][str(j)]['shape_attributes']['all_points_y'] = [int(ymin), int(ymax)]
                    region_data_valid[str(name)]['regions'][str(j)]['regions_attributes'] = {}
                j = j + 1
    
    with open('train_data.json', 'w') as outfile:
        json.dump(region_data_train, outfile)            
                
    with open('valid_data.json', 'w') as outfile:
        json.dump(region_data_valid, outfile) 
        
def createStudyDatasetfromcrop():
    with open('links.csv', newline='') as csvf:
        csvr = csv.reader(csvf, delimiter=',')
        i = 0
        j = 0
        for row in csvr:
            i = i + 1         
            prefix = 'N'
            if('positive' in row[0]):
                prefix = 'P'
            name = prefix + '_' + str(i)           
            
            imagepath = 'D:\dev\RAID\\newtest-crop-224\\' + name + '.png'
            outpath = 'D:\dev\RAID\\newtest-study-224\\' + row[0]            
            
            try:
                im = Image.open(imagepath)
                if not os.path.exists(os.path.dirname(outpath)):
                    try:
                        os.makedirs(os.path.dirname(outpath))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                im.save(outpath)
            except: 
                try:
                    im = Image.open(outpath)  
                    print(imagepath)
                    print(outpath)
                    print('-'*10)
                except:
                    print('good')
                    pass
                pass      
                       
def rename(current, set, name, newName):
    try:
        path = os.getcwd() + '/' + set + '/' + current + '/' + name
        newPath = os.getcwd() + '/' + set + '/' + current + '/' + newName
        os.rename(path,newPath)
        print('Renamed!')
    except Exception as E:
        pass

def convertPositiveToNegative(current):
    with open('wronglabels.csv', newline='') as csvf:
        csvr = csv.reader(csvf, delimiter=',')
        i = 0
        j = 0
        for row in csvr:
            name = row[0]
            newName = ''
            if 'P_' in name:
                newName = 'N_' + name[2:len(name)]
            elif 'N_' in name:
                newName = 'P_' + name[2:len(name)]  
            rename(current, 'test', name, newName)
            rename(current, 'train', name, newName)
            rename(current, 'valid', name, newName)     
             
def care(min, max, limit, size):    
    diff = int(max) - int(min)
    
    if(diff < limit):
        diff512 = limit - diff
        gapmin = math.floor(diff512/2)
        gapmax = math.floor(diff512/2)
        while(gapmin+gapmax != diff512):
            gapmin = gapmin + 1
        newmin = int(min) - gapmin
        newmax = int(max) + gapmax
        if(newmax > size):
            gap1024 = newmax - size
            newmax = size
            newmin = newmin - gap1024
        if(newmin < 0):
            gap0 = 0 - newmin
            newmin = 0
            newmax = newmax + gap0
    else:
        return min, max
    
    return min, max

def crop_image():
    
    with open('links.csv', newline='') as csvf:
        csvr = csv.reader(csvf, delimiter=',')
        i = 0
        j = 0
        for row in csvr:
            i = i + 1         
            prefix = 'N'
            if('positive' in row[0]):
                prefix = 'P'
            name = prefix + '_' + str(i)
            
            xmlpath = 'D:\dev\RAID\labels\\' + name + '.xml'  

            try:
                xmldoc = minidom.parse(xmlpath)
                itemlist = xmldoc.getElementsByTagName('object')            
                imagepath = 'D:\dev\RAID\dataset\\train\\' + name + '.png'

                for item in itemlist:           
                    factor = 1           
                    xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                    ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                    xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                    ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data  
                    
                    xmin = (int)((int)(xmin)*factor)
                    xmax = (int)((int)(xmax)*factor)
                    ymin = (int)((int)(ymin)*factor)
                    ymax = (int)((int)(ymax)*factor)
                    
                    xmin, xmax = care(xmin, xmax, 224, 1024)
                    ymin, ymax = care(ymin, ymax, 224, 1024)
                                                            
                    im = Image.open(imagepath)
                    im = im.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
                    im.save('D:\dev\RAID\\newtest-study\\' + name + '.png')               		
            except: 
                print('pas de xml pour ' + str(name))
                pass

def main():
	createStudyDatasetfromcrop() 
                     
if __name__ == '__main__':
    main()