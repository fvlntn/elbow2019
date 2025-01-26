from pprint import pprint

import csv
from shutil import rmtree, copyfile
from os import mkdir, scandir, getcwd
import pydicom
import numpy as np
import math
import cv2
import random
from pydicom.data import get_testdata_files
from pydicom.filereader import read_dicomdir

def checkAnonymize(dicom):
    if(dicom.PatientName != 'NOM^AUCUN'):
        return False
    if(dicom.PatientID != 'AUCUNID'):
        return False
    if(dicom.PatientBirthDate != ''):
        return False
    return True

def getAgeSex(dicom):
    age = dicom.PatientAge
    if(age[3] == 'Y'):
        age = age[0:3]
    elif(age[3] == 'M'):
        age = '00' + str(math.floor(int(age[0:3])/12+0.5))
    sex = dicom.PatientSex == 'M'
    return age, sex

def getMURALikeDataset():
    test_num = 0
    train_num = 0
    val_num = 0
    badanom_num = 0
    try:
        mkdir(cwd + '/CHR')
        mkdir(cwd + '/CHR/train')
        mkdir(cwd + '/CHR/train/RAID_ELBOW_3')
        mkdir(cwd + '/CHR/valid')
        mkdir(cwd + '/CHR/valid/RAID_ELBOW_3')
        mkdir(cwd + '/CHR/test')
        mkdir(cwd + '/CHR/test/RAID_ELBOW_3')
    except FileExistsError:
        rmtree(cwd + '/CHR')
        mkdir(cwd + '/CHR')
        mkdir(cwd + '/CHR/train')
        mkdir(cwd + '/CHR/train/RAID_ELBOW_3')
        mkdir(cwd + '/CHR/valid')
        mkdir(cwd + '/CHR/valid/RAID_ELBOW_3')
        mkdir(cwd + '/CHR/test')
        mkdir(cwd + '/CHR/test/RAID_ELBOW_3')
    with open('CHR/train_image_studies.csv', mode='w') as train_image_studies:
        with open('CHR/train_image_paths.csv', mode='w') as train_image_paths:
            with open('CHR/test_image_studies.csv', mode='w') as test_image_studies:
                with open('CHR/test_image_paths.csv', mode='w') as test_image_paths:
                    with open('CHR/valid_image_studies.csv', mode='w') as valid_image_studies:
                        with open('CHR/valid_image_paths.csv', mode='w') as valid_image_paths:
                            train_image_studies = csv.writer(train_image_studies, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            valid_image_studies = csv.writer(valid_image_studies, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            test_image_studies = csv.writer(test_image_studies, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            train_image_paths = csv.writer(train_image_paths, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            test_image_paths = csv.writer(test_image_paths, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            valid_image_paths = csv.writer(valid_image_paths, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            patient_id = 0
                            for folder in [f.path for f in scandir(cwd) if f.is_dir()]:
                                if(not folder.endswith('.git')):
                                    if(folder.endswith('P')):
                                        label = 1
                                        folderSuffix = '_positive'
                                    else:
                                        label = 0
                                        folderSuffix = '_negative'
                                    for study in [f.path for f in scandir(folder) if f.is_dir()]:
                                        patient_id += 1
                                        id = 0
                                        r = random.uniform(0,1)  
                                        if(r <= test):
                                            set_label = 'test'
                                            #test_image_studies.writerow([cwd + '/CHR/' + set_label + '/RAID_ELBOW_3/patient' + str(patient_id) + '/study1' + folderSuffix + '/', label])
                                        elif(test < r <= test+validation):
                                            set_label = 'valid'
                                            #valid_image_studies.writerow([cwd + '/CHR/' + set_label + '/RAID_ELBOW_3/patient' + str(patient_id) + '/study1' + folderSuffix + '/', label])
                                        else:                                        
                                            set_label = 'train'
                                            #train_image_studies.writerow([cwd + '/CHR/' + set_label + '/RAID_ELBOW_3/patient' + str(patient_id) + '/study1' + folderSuffix + '/', label])
                                        xraynum = 0
                                        for xray in [f.path for f in scandir(study) if f.is_dir()]:
                                            xrayname = xray.split('\\')[len(xray.split('\\'))-1]
                                            if(not xrayname.startswith('Series 001')):  
                                                for dicomfilename in [f.path for f in scandir(xray) if f.is_file()]: 
                                                    if(dicomfilename.endswith('.dcm')):
                                                        dicom = pydicom.dcmread(dicomfilename)                                                         
                                                        #if(dicom.BodyPartExamined == "UP_EXM"):
                                                        xraynum += 1  
                                                        try:
                                                            mkdir(cwd + '/CHR/' + set_label + '/RAID_ELBOW_3/patient0' + str(patient_id))
                                                            mkdir(cwd + '/CHR/' + set_label + '/RAID_ELBOW_3/patient0' + str(patient_id) + '/study1' + folderSuffix)
                                                        except FileExistsError:
                                                            pass
                                                        if(not checkAnonymize(dicom)):
                                                            badanom_num += 1
                                                        id += 1
                                                        age, gender = getAgeSex(dicom)                 
                                                        out = dicom.pixel_array                                         
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
                                                        out = cv2.resize(out,imagesize)  
                                                        output_filename = 'image' + str(xraynum) + '.png'
                                                        output_path = cwd + '/CHR/' + set_label + '/RAID_ELBOW_3/patient0' + str(patient_id) + '/study1' + folderSuffix + '/' + output_filename
                                                        cv2.imwrite(output_path, out)    
                                                        reout = cv2.imread(output_path)    
                                                        cv2.imwrite(output_path, reout)    														
                                                        #train_image_paths.writerow([cwd + '/CHR/' + set_label + '/RAID_ELBOW_5/patient' + str(patient_id) + '/study1' + folderSuffix + '/', dicomfilename, label])   
                                                        print(dicomfilename.split('RAID'))
                                                        train_image_paths.writerow([set_label + '/RAID_ELBOW_5/patient0' + str(patient_id) + '/study1' + folderSuffix + '/' + output_filename, dicomfilename.split('RAID')[1], label])             
                                                        if(r <= test):
                                                            test_num += 1
                                                            #test_image_paths.writerow([cwd + '/CHR/' + set_label + '/RAID_ELBOW_3/patient' + str(patient_id) + '/study1' + folderSuffix + '/' + output_filename])
                                                        elif(test < r <= test+validation):
                                                            val_num += 1 
                                                            #valid_image_paths.writerow([cwd + '/CHR/' + set_label + '/RAID_ELBOW_3/patient' + str(patient_id) + '/study1' + folderSuffix + '/' + output_filename])
                                                        else:
                                                            train_num += 1
                                                            #train_image_paths.writerow([cwd + '/CHR/' + set_label + '/RAID_ELBOW_3/patient' + str(patient_id) + '/study1' + folderSuffix + '/' + output_filename])
                            if(badanom_num != 0):
                                print(badanom_num + " badly anonymized DICOM")
                            print("Dataset created:")
                            print(str(train_num) + " training images.")
                            print(str(test_num) + " test images.")
                            print(str(val_num) + " validation images.")                                    

                                                                
def getDataset():
    try:
        mkdir(cwd + '/dataset')
        mkdir(cwd + '/dataset/test')
        mkdir(cwd + '/dataset/train')
        mkdir(cwd + '/dataset/valid')
    except FileExistsError:
        rmtree(cwd + '/dataset')
        mkdir(cwd + '/dataset')
        mkdir(cwd + '/dataset/test')
        mkdir(cwd + '/dataset/train')
        mkdir(cwd + '/dataset/valid')
    badanom_num = 0
    test_num = 0
    val_num = 0
    train_num = 0
    count = 0
    with open('infos.csv', mode='w') as info:
        info = csv.writer(info, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for folder in [f.path for f in scandir(cwd) if f.is_dir()]:
            if(not folder.endswith('.git')):
                if(folder.endswith('P')):
                    label = 1
                    folderSuffix = 'P'
                elif(folder.endswith('N')):
                    label = 0
                    folderSuffix = 'N'    
                else:
                    pass
                for patient in [f.path for f in scandir(folder) if f.is_dir()]: 
                    notwritten = True
                    patientname = patient.split('\\')[len(patient.split('\\'))-1]
                    for study in [f.path for f in scandir(patient) if f.is_dir()]:  
                        studyname = study.split('\\')[len(study.split('\\'))-1]
                        if(not studyname.startswith('Series 001')):
                            for dicomfilename in [f.path for f in scandir(study) if f.is_file()]:      
                                if(dicomfilename.endswith('.dcm')):
                                    dicom = pydicom.dcmread(dicomfilename)   
                                    out = dicom.pixel_array
                                    age, sex = getAgeSex(dicom)
                                    name = folderSuffix + '_' + str(count)
                                    if(notwritten):
                                        info.writerow([name, age, sex])
                                        notwritten = False
                                    #if(dicom.BodyPartExamined == "UP_EXM"):
                                    #count += 1
                                   #if(normalize):                                            
                                   #    out = cv2.normalize(out,None,0,16384,cv2.NORM_MINMAX)                                    
                                   #    _, out2 = cv2.threshold(out,200,16384,cv2.THRESH_BINARY)
                                   #    out = out2 - out
                                   #    _, out2 = cv2.threshold(out,16184,16384,cv2.THRESH_BINARY_INV)
                                   #    out = out - out2 
                                   #    _, out2 = cv2.threshold(out,200,16384,cv2.THRESH_BINARY)
                                   #    out = out2 - out
                                   #    _, out2 = cv2.threshold(out,16184,16384,cv2.THRESH_BINARY_INV)
                                   #    out = out - out2   
                                   #    _, out2 = cv2.threshold(out,200,16384,cv2.THRESH_BINARY)
                                   #    out = out2 - out
                                   #    _, out2 = cv2.threshold(out,16184,16384,cv2.THRESH_BINARY_INV)
                                   #    out = out - out2                                       
                                   #    out = cv2.normalize(out,None,0,65535,cv2.NORM_MINMAX)
                                   #if(resize):
                                   #    out = cv2.resize(out,(1024,1024))
                                   #if(not checkAnonymize(dicom)):
                                   #    badanom_num += 1                                    
                                   #train_num += 1
                                    #cv2.imwrite(cwd + '/dataset/train/' + str(folderSuffix) + '_' + str(patientname.split(" ")[0]) + '_' + str(patientname.split(" ")[1]) + '_' + str(studyname.split(" ")[0]) + '_' + str(studyname.split(" ")[1]) + '.png', out)
                                    #cv2.imwrite(cwd + '/dataset/train/' + str(folderSuffix) + '_' + str(count) + '.png', out)
                                    #reout = cv2.imread(cwd + '/dataset/train/' + str(folderSuffix) + '_' + str(count) + '.png')
                                    #cv2.imwrite(cwd + '/dataset/train/' + str(folderSuffix) + '_' + str(count) + '.png', reout)

    if(badanom_num != 0):
        print(badanom_num + " badly anonymized DICOM")
    print("Dataset created:")
    print(str(train_num) + " training images.")
    print(str(test_num) + " test images.")
    print(str(val_num) + " validation images.")    
    
def getRSNALikeDataset():
    test_num = 0
    train_num = 0
    val_num = 0
    badanom_num = 0
    try:
        mkdir(cwd + '/RSNA')
        mkdir(cwd + '/RSNA/train')
        mkdir(cwd + '/RSNA/valid')
        mkdir(cwd + '/RSNA/test')
    except FileExistsError:
        rmtree(cwd + '/RSNA')
        mkdir(cwd + '/RSNA')
        mkdir(cwd + '/RSNA/train')
        mkdir(cwd + '/RSNA/valid')
        mkdir(cwd + '/RSNA/test')
    with open('RSNA/train_data.csv', mode='w') as train_image_studies:
        with open('RSNA/test_data.csv', mode='w') as test_image_studies:
            with open('RSNA/valid_data.csv', mode='w') as valid_image_studies:
                with open('RSNA/links.csv', mode='w') as image_studies:
                    train_image_studies = csv.writer(train_image_studies, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    valid_image_studies = csv.writer(valid_image_studies, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    test_image_studies = csv.writer(test_image_studies, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    image_studies = csv.writer(image_studies, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    test_image_studies.writerow(['id', 'boneage', 'male', 'link'])
                    train_image_studies.writerow(['id', 'boneage', 'male', 'link'])
                    valid_image_studies.writerow(['id', 'boneage', 'male','link'])
                    image_studies.writerow(['id', 'boneage', 'male', 'link', 'set'])
                    xraynum = 0
                    for folder in [f.path for f in scandir(cwd) if f.is_dir()]:
                        if(not folder.endswith('.git')):
                            for study in [f.path for f in scandir(folder) if f.is_dir()]:                            
                                for xray in [f.path for f in scandir(study) if f.is_dir()]:                                
                                    xrayname = xray.split('\\')[len(xray.split('\\'))-1]
                                    if(not xrayname.startswith('Series 001')):  
                                        for dicomfilename in [f.path for f in scandir(xray) if f.is_file()]: 
                                            if(dicomfilename.endswith('.dcm')):                                            
                                                dicom = pydicom.dcmread(dicomfilename) 
                                                if(dicom.BodyPartExamined == "UP_EXM"):
                                                    xraynum += 1
                                                    r = random.uniform(0,1)  
                                                    age, gender = getAgeSex(dicom)
                                                    if(r <= test):
                                                        set_label = 'test'
                                                        test_image_studies.writerow([str(xraynum), age, gender, dicomfilename.split('age')[1]])
                                                        image_studies.writerow([str(xraynum), age, gender, dicomfilename.split('age')[1], 'test'])
                                                        test_num += 1
                                                    elif(test < r <= test+validation):
                                                        set_label = 'valid'
                                                        valid_image_studies.writerow([str(xraynum), age, gender, dicomfilename.split('age')[1]])
                                                        image_studies.writerow([str(xraynum), age, gender, dicomfilename.split('age')[1], 'valid'])
                                                        val_num += 1
                                                    else:                                        
                                                        set_label = 'train'
                                                        train_image_studies.writerow([str(xraynum), age, gender, dicomfilename.split('age')[1]])
                                                        image_studies.writerow([str(xraynum), age, gender, dicomfilename.split('age')[1], 'train'])
                                                        train_num += 1
                                                    if(not checkAnonymize(dicom)):
                                                        badanom_num += 1         
                                                    out = dicom.pixel_array 
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
                                                    out = cv2.resize(out,imagesize)  
                                                    output_filename = str(xraynum) + '.png'
                                                    output_path = cwd + '/RSNA/' + set_label + '/' + output_filename
                                                    cv2.imwrite(output_path, out)                     
    if(badanom_num != 0):
        print(badanom_num + " badly anonymized DICOM")
    print("Dataset created:")
    print(str(train_num) + " training images.")
    print(str(test_num) + " test images.")
    print(str(val_num) + " validation images.")      
    
def getAgeSexDataset():
    try:
        mkdir(cwd + '/dataset')
    except FileExistsError:
        rmtree(cwd + '/dataset')
        mkdir(cwd + '/dataset')
    count = 0
    with open('infos_train.csv', mode='w') as info_train:
        with open('infos_test.csv', mode='w') as info_test:
            with open('infos_valid.csv', mode='w') as info_valid:
                with open('links.csv', mode='r') as links:  
                    csvr = csv.reader(links, delimiter=',')
                    info_train = csv.writer(info, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    info_valid = csv.writer(info, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    info_test = csv.writer(info, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for row in csvr:
                        patientid = -1
                        if(row[0].split('/')[2].split('patient')[1] != patientid):
                            notwritten = True
                        else:
                            notwritten = False
                        patientid = row[0].split('/')[2].split('patient')[1]
                        dicomfilename = row[1]
                        if(dicomfilename.endswith('.dcm')):
                            dicom = pydicom.dcmread(dicomfilename)   
                            out = dicom.pixel_array
                            age, sex = getAgeSex(dicom)
                            name = folderSuffix + '_' + str(count)
                            if(notwritten):
                                notwritten = False
                                if(row[0].startswith('test')):
                                    info_test.writerow([name, age, sex])                                  
                                elif(row[0].startswith('train')):                                 
                                    info_train.writerow([name, age, sex])
                                else:                                
                                    info_valid.writerow([name, age, sex])                                      

def testptdr():
    mdr = 'valid/RAID_ELBOW_5/patient4/xdxdxd'
    patientid = mdr.split('/')[2].split('patient')[1]
    print(patientid)
    
test = 1.0
validation = 0.0

normalize = True
resize = True
imagesize = (1024,1024)

random.seed(123)
cwd = getcwd()
label = 0

#testptdr()
getDataset()
#getMURALikeDataset()
#getRSNALikeDataset()                    
