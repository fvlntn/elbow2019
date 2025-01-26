from xml.dom import minidom
import os
import glob
import random
import csv
import numpy as np
import cv2
import pydicom
import math
from PIL import Image


def getAgeSex(dicom):
    age = dicom.PatientAge
    if(age[3] == 'Y'):
        age = age[0:3]
    else:
        age = str(math.floor(int(age[0:3])/12+0.5))
    sex = dicom.PatientSex
    return age, sex

def getresizedimage(set2):
    with open('links.csv', newline='') as csvf:
        csvr = csv.reader(csvf, delimiter=',')
        i = 0
        for row in csvr:
            i = i + 1  
            set = row[0].split('/')[0]
            if(set == set2):
                xraypath = '/home/frvalent/dev/RAID' + row[1]
                if('positive' in row[0]):
                    prefix = 'P'
                else:
                    prefix = 'N'
                name = prefix + '_' + str(i) + '.png' 
                try:
                    img = pydicom.dcmread(xraypath) 
                    out = img.pixel_array
                    out = cv2.normalize(out,None,0,16384,cv2.NORM_MINMAX)                                    
                    _, out2 = cv2.threshold(out,200,16384,cv2.THRESH_BINARY)
                    out = out2 - out
                    _, out2 = cv2.threshold(out,16184,16384,cv2.THRESH_BINARY_INV)
                    out = out - out2 
                    _, out2 = cv2.threshold(out,200,16384,cv2.THRESH_BINARY)
                    out = out2 - out
                    _, out2 = cv2.threshold(out,16184,16384,cv2.THRESH_BINARY_INV)
                    out = out - out2   
                    _, out2 = cv2.threshold(out,200,16384,cv2.THRESH_BINARY)
                    out = out2 - out
                    _, out2 = cv2.threshold(out,16184,16384,cv2.THRESH_BINARY_INV)
                    out = out - out2                                       
                    out = cv2.normalize(out,None,0,65535,cv2.NORM_MINMAX)  
                    out = cv2.resize(out, (2341,2341))                        
                    outputpath = '/home/frvalent/dev/RAID/label-from-elbow/new' + set + '/' + name 
                    cv2.imwrite(outputpath, out)
                except FileNotFoundError: 
                    pass

def getradio(set):
    with open('links.csv', newline='') as csvf:
        with open('infos_' + str(set) + '.csv', mode='w') as info:
            info = csv.writer(info, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvr = csv.reader(csvf, delimiter=',')
            i = 0
            for row in csvr:
                i = i + 1   
                if(row[0].startswith(set)):                    
                    xraypath = '/home/frvalent/dev/RAID' + row[1]
                    prefix = 'N'
                    if('positive' in row[0]):
                        prefix = 'P'
                    name = prefix + '_' + str(i) + '.png' 
                    try:
                        img = pydicom.dcmread(xraypath) 
                        age, gender = getAgeSex(img)
                        info.writerow([name, age, gender]) 
                    except FileNotFoundError: 
                        pass

            
def getimage():
    
    with open('links.csv', newline='') as csvf:
        csvr = csv.reader(csvf, delimiter=',')
        i = 0
        for row in csvr:
            i = i + 1         
            imagepath = '/home/frvalent/dev/RAID/elbow/CHRfinal/' + row[0]
            prefix = 'N'
            if('positive' in row[0]):
                prefix = 'P'
            set = row[0].split('/')[0]
            outpath = '/home/frvalent/dev/RAID/label2/' + set + '/' + prefix + '_' + str(i) + '.png'  
            try:
                img = cv2.imread(imagepath)
                #if(img is not None):
                    #cv2.imwrite(outpath, img)
            except FileNotFoundError: 
                pass
            
            

def main():
    getresizedimage('test')
    getresizedimage('train')
    getresizedimage('valid')


if __name__ == '__main__':
    main()