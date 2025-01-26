from os import getcwd, mkdir
from shutil import rmtree
from pydicom import dcmread
from torchvision import transforms
import cv2
import numpy as np
from os import scandir

def checkFileNames(filenames):
    extensions = ['.jpg', '.jpeg', '.png', '.dcm']
    filenamesCheck = []
    bool = True
    if len(filenames) == 0:
        bool = False
    for filename in filenames:
        if filename.endswith(extensions[0]) or filename.endswith(extensions[1]) or filename.endswith(extensions[2]) or filename.endswith(extensions[3]):
            filenamesCheck.append(True)
        else:
            filenamesCheck.append(False)
    for check in filenamesCheck:
        bool = bool * check
    return bool  
   
def getAgeSex(dicom):
    age = dicom.PatientAge
    if(age[3] == 'Y'):
        age = age[0:3]
    elif(age[3] == 'M'):
        age = '00' + str(math.floor(int(age[0:3])/12+0.5))
    sex = dicom.PatientSex == 'M'
    return age, sex

def isFileXray(filename):
    try:
        dicom = dcmread(filename)
        img = dicom.pixel_array
        return True
    except:
        return False
   
def getImagesFromDicom(filenames):
    xraynum = 0
    cwd = getcwd()
    newFilenames = []
    try:
        mkdir(cwd + '/xrays_temp')
    except FileExistsError:
        rmtree(cwd + '/xrays_temp')
        mkdir(cwd + '/xrays_temp')
    for filename in filenames:
        xraynum += 1
        if filename.endswith('.dcm') or not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
            dicom = dcmread(filename)       
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
            out = cv2.resize(out,(1024,1024))  
            output_filename = 'image' + str(xraynum) + '.png'
            output_path = cwd + '/xrays_temp/' + output_filename
            cv2.imwrite(output_path, out)    
            reout = cv2.imread(output_path)    
            cv2.imwrite(output_path, reout)            
            newFilenames.append(output_path) 
        else:
            newFilenames.append(filename)
    return newFilenames
    
def getXraysFromFolder(folder):
    filenames = []
    for study in [f.path for f in scandir(folder) if f.is_dir()]:
        for s1 in [f.path for f in scandir(study) if f.is_file()]:
            if isFileXray(s1):
                filenames.append(s1)
        for serie in [f.path for f in scandir(study) if f.is_dir()]:
            for s2 in [f.path for f in scandir(serie) if f.is_file()]:
                if isFileXray(s2):
                    filenames.append(s2)
            for object in [f.path for f in scandir(serie) if f.is_dir()]:
                for xray in [f.path for f in scandir(object) if f.is_file()]:
                    if isFileXray(xray):
                        filenames.append(xray)    
    return filenames
    
 
def predPct(pred):
    if pred <= 0.5:
        txt = 'Negative '
        pct = (pred - 0.5) * 2 * (-1)
        pct = format(pct*100, '.2f')
    else:
        txt = 'Positive '
        pct = (pred - 0.5) * 2 
        pct = format(pct*100, '.2f')
    txt += pct + ' %'     
    return txt


def TensorToPIL(tensor):
    PILimg = transforms.ToPILImage()(tensor)
    return PILimg
    
def TensorToCV2(tensor):
    CV2img = tensor[0].numpy()
    return CV2img
    
def default_transforms():
    return transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
 
#def default_transforms_raid():
#    return transforms.Compose([
#                transforms.Resize((1024, 1024)),
#                transforms.ToTensor(), 
#                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                ])      
                
def reverse_normalize(image):
    reverse = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
    return reverse(image) 

def _is_iterable(variable):
    return isinstance(variable, list) or isinstance(variable, tuple)  
    