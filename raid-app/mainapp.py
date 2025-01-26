import tkinter.filedialog
import tkinter as tk 
from tkinter import messagebox
import ui
from model import DensenetModel, DetectionModel, RaidModel
import utils, vis
from PIL import Image, ImageTk, ImageGrab
import torch
import cv2
import math
from functools import partial

   
def loadImages():
    filenames = tk.filedialog.askopenfilenames(title = "Select images",filetypes = (("all files","*.*"),("dicom files","*.dcm"),("jpg files","*.jpg")))
    if len(filenames) > 0:     
        UI.reset()
        UI.openFilenames(filenames)
        UI.openImages()
        
def loadImagesFolder():
    folder = tk.filedialog.askdirectory(title = "Select folder")
    filenames = utils.getXraysFromFolder(folder)
    if len(filenames) > 0:     
        UI.reset()
        UI.openFilenames(filenames)
        UI.openImages()
    
def loadClipboardImage(event):
    UI.loadClipboardImage()    

def getPrediction(): 
    if(not UI.computed and len(UI.images) > 0 and (raid512On.get() == 1 or raid1024On.get() == 1 or crop512On.get() == 1)):
        predsFull = []
        imagesCrop = []
        predsCrop = []
        
        if raid512On.get() == 1:    
            predsFull.append((raidModel.fullModels[0].name, raidModel.fullModels[0].predict_for_study(UI.images)))
        if raid1024On.get() == 1:            
            predsFull.append((raidModel.fullModels[1].name, raidModel.fullModels[1].predict_for_study(UI.images)))
        if crop512On.get() == 1:
            imagesCrop = raidModel.detectionModel.getCropImages(UI.images)
            predsCrop.append((raidModel.cropModels[0].name, raidModel.cropModels[0].predict_for_study(imagesCrop)))
        
        predsResults = []
        
        gradResults = []
        gradFilenames = []
        
        normGradResults = []
        normGradFilenames = []
        
        for preds in [predsFull, predsCrop]:
            for pred in preds:
                predsResults.append(pred)
                
        for j, model in enumerate(raidModel.models):
            if 'Crop512' in model.name and crop512On.get() == 1:
                gradimgs = model.gradcam(imagesCrop)
                normalGradimgs = model.normGradcam(imagesCrop)
            elif 'Raid512' in model.name and raid512On.get() == 1:
                gradimgs = model.gradcam(UI.images)
                normalGradimgs = model.normGradcam(UI.images)
            elif 'Raid1024' in model.name and raid1024On.get() == 1:
                gradimgs = model.gradcam(UI.images)
                normalGradimgs = model.normGradcam(UI.images)
            else:
                gradimgs = []
                normalGradimgs = []
            for i, gradcam in enumerate(gradimgs):                
                gradResults.append(gradcam)
                outFilename = 'xrays_temp/gradcam_' + str(j) + '_' + str(i) + '.png'
                cv2.imwrite(outFilename, gradcam)
                gradFilenames.append(outFilename)
            for i, gradcam in enumerate(normalGradimgs):                
                normGradResults.append(gradcam)
                outFilename = 'xrays_temp/normgradcam_' + str(j) + '_' + str(i) + '.png'
                cv2.imwrite(outFilename, gradcam)
                normGradFilenames.append(outFilename)
                
        UI.setPreds(predsResults)
        UI.setGradfilenames(gradFilenames)
        UI.setNormGradfilenames(normGradFilenames)
        UI.computed = True
        
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()    
        
def showGradcams():
    if UI.computed:
        UI.showGradcams(raid512On, raid1024On, crop512On)
    
def showNormGradcams():
    if UI.computed:
        UI.showNormGradcams(raid512On, raid1024On, crop512On)

def showXrays():
    if UI.computed:
        UI.openImages() 
        
def exportResults():
    if UI.computed and not UI.exported:
        UI.exportResults()
        
def unCompute():
    UI.computed = False
    
def wheel(event):
    UI.zoomImage(event)

def moveFrom(event):
    UI.canvas.scan_mark(event.x, event.y)

def moveTo(event):
    UI.canvas.scan_dragto(event.x, event.y, gain=1)
    UI.showImage()
        
modelPaths = [
                    'models/crop2-fix-512-7_acc.pth', 
                    'models/raid5-512-8_acc.pth', 
                    'models/raid5-fix-1024-12_loss.pth', 
                    'models/frcnn_elbow.pth'
            ]

modelCrop512 = DensenetModel.load(modelPaths[0], 'Crop512')
modelRaid512 = DensenetModel.load(modelPaths[1], 'Raid512')
modelRaid1024 = DensenetModel.load(modelPaths[2], 'Raid1024')
modelDetection = DetectionModel.load(modelPaths[3])     

raidModel = RaidModel([modelRaid512, modelRaid1024], [modelCrop512], modelDetection)
    
## ----------------------------------------------------------------------------------------------------------------------------------------


gui = tk.Tk(className='radiology AI demonstrator')
w,h = 1000,1000
gui.minsize(width=w, height=h)
gui.maxsize(width=w, height=h)

gui.bind('<Control-v>', loadClipboardImage)

UI = ui.UI(gui, w, h)
#UI.canvas.bind("<MouseWheel>", wheel)
#UI.canvas.bind("<Button-4>", wheel)
#UI.canvas.bind("<Button-5>", wheel)
#UI.canvas.bind('<ButtonPress-1>', moveFrom)
#UI.canvas.bind('<B1-Motion>', moveTo)        

loadImagesButton = tk.Button(text="Load images", width=20, height=2, command=loadImages)
loadImagesButton.place(x=math.floor(w/2)-350,y=h-100)

loadImageFolderButton = tk.Button(text="Load folder", width=20, height=2, command=loadImagesFolder)
loadImageFolderButton.place(x=math.floor(w/2)-350,y=h-70)

computeButton = tk.Button(text="Compute", width=20, height=4, command=getPrediction)
computeButton.place(x=math.floor(w/2)-200,y=h-100)

showXraysButton = tk.Button(text="Show xrays", width=20, height=4, command=showXrays)
showXraysButton.place(x=math.floor(w/2)-50,y=h-100)

showGradcamsButton = tk.Button(text="Show gradcams", width=20, height=2, command=showGradcams)
showGradcamsButton.place(x=math.floor(w/2)+100,y=h-100)

showNormGradcamsButton = tk.Button(text="Show norm. gradcams", width=20, height=2, command=showNormGradcams)
showNormGradcamsButton.place(x=math.floor(w/2)+100,y=h-70)

exportButton = tk.Button(text="Export results", width=20, height=4, command=exportResults)
exportButton.place(x=math.floor(w/2)+250,y=h-100)

raid512On = tk.BooleanVar()
raid512On.set(1)
raid512Button = tk.Checkbutton(text="Raid512", variable=raid512On, command=unCompute)
raid512Button.place(x=10,y=1)

raid1024On = tk.BooleanVar()
raid1024On.set(1)   
raid1024Button = tk.Checkbutton(text="Raid1024", variable=raid1024On, command=unCompute)
raid1024Button.place(x=100,y=1)

crop512On = tk.BooleanVar()
crop512On.set(1)
crop512Button = tk.Checkbutton(text="Crop512", variable=crop512On, command=unCompute)
crop512Button.place(x=200,y=1)

gui.mainloop()