import os 
import glob

def main():   
    cwd = os.getcwd()
    i = 0
    for xmlname in glob.glob('elbow/*.xml'): 
        i = i + 1
        xmlnameDst = 'elbow/image' + str(i) + '.xml'  
        
        pngname = xmlname.split('.')[0] + '.png'        
        pngnameDst = 'elbow/image' + str(i) + '.png'     
        
        os.rename(xmlname, xmlnameDst) 
        print('xml renamed')
        
        os.rename(pngname, pngnameDst) 
        print('png renamed')
  
if __name__ == '__main__': 
    main() 