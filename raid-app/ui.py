import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import utils
import math
import csv

class UI:
    def __init__(self, gui, w, h):
        self.w = w
        self.h = h
        self.imw = 3/4*self.w
        self.imh = 3/4*self.h        
        self.canvas = tk.Canvas(gui, width=self.imw, height=self.imh)
        self.canvas.place(x=math.floor(w/10), y=75)
        self.image = Image.open("raid.png")
        self.canvasimage = ImageTk.PhotoImage(self.image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvasimage)
        self.imscale = 1.0
        self.delta = 1.3
        self.container = self.canvas.create_rectangle(0, 0, self.imw, self.imh, width=0)
        self.showImage() 
        self.predtext = tk.Label(gui)
        self.predtext.place(x=math.floor(w/2-w/10), y=10)      
        self.imagetitle = tk.Label(gui)
        self.imagetitle.place(x=math.floor(w/2)-75, y=h-130)     
        self.imagecounttext = tk.Label(gui)
        self.imagecounttext.place(x=math.floor(w/2)-75, y=h-150)
        self.showPreviousButton = tk.Button(text="Previous", width=6, height=2, command=self.showPreviousImage)
        self.showPreviousButton.place(x=math.floor(w/5),y=h-150)        
        self.showNextButton = tk.Button(text="Next", width=6, height=2, command=self.showNextImage)
        self.showNextButton.place(x=math.floor(w/5)+75,y=h-150)
        self.deleteCurrentImageButton = tk.Button(text="X", width=6, height=2, command=self.deleteCurrentImage)
        self.deleteCurrentImageButton.place(x=math.floor(3*w/5),y=h-150)
        self.currentImage = 0
        self.images = []
        self.preds = []
        self.filenames = []
        self.gradfilenames = []
        self.normgradfilenames = []
        self.oldfilenames = []
        self.titles = []
        self.showingXrays = True
        self.computed = False
        self.exported = False
        
    def loadClipboardImage(self):
        try:
            temp_path = 'clipboard.png'
            im = ImageGrab.grabclipboard()
            im.save(temp_path)
            self.reset()
            self.openFilenames([temp_path])
            self.openImages()
        except:
            pass

    def setImages(self, images):
        if len(images) > 0:
            self.images = images#.copy()
            self.displayImage(0)
        
    def setPreds(self, preds):
        self.preds = preds
        self.showPreds()
        
    def setTitles(self, titles):
        self.titles = titles
    
    def setFilenames(self, filenames):
        self.filenames = filenames
    
    def setGradfilenames(self, gradfilenames):
        self.gradfilenames = gradfilenames
        
    def setNormGradfilenames(self, normgradfilenames):
        self.normgradfilenames = normgradfilenames
        
    def setOldfilenames(self, oldfilenames):
        self.oldfilenames = oldfilenames
                
    def reset(self):
        self.setImages([])
        self.setPreds([])
        self.setPredsText('')
        self.setTitles([])
        self.setFilenames([])
        self.setGradfilenames([])
        self.setNormGradfilenames([])
        self.setOldfilenames([])
        
    def openImages(self):
        self.showingXrays = True
        images = []
        try:
            for filename in self.filenames:
                image = Image.open(filename)
                images.append(image)            
            self.setTitles(self.filenames)
        except Exception as e: 
            print(e)
        self.setImages(images)
     
    def openFilenames(self, filenames):
        self.reset()
        self.setTitles(filenames)
        self.setOldfilenames(filenames)
        self.setFilenames(utils.getImagesFromDicom(filenames))
        self.computed = False
        self.exported = False
    
    def showNextImage(self):
        if len(self.images) != 0:
            if self.currentImage == len(self.images) - 1:
                nextImage = 0
            else:            
                nextImage = self.currentImage +  1
            self.displayImage(nextImage)
            
    def showPreviousImage(self):    
        if len(self.images) != 0:            
            if self.currentImage == 0:
                nextImage = len(self.images) - 1
            else:        
                nextImage = self.currentImage -  1
            self.displayImage(nextImage)
            
    def deleteCurrentImage(self):
        if self.showingXrays and not self.computed and len(self.images) > 1:            
            self.images.pop(self.currentImage)            
            self.titles.pop(self.currentImage)            
            self.oldfilenames = list(self.oldfilenames)
            self.oldfilenames.pop(self.currentImage)            
            #self.filenames = list(self.filenames)
            #self.filenames.pop(self.currentImage)            
            if self.currentImage > 0:
                self.displayImage(self.currentImage - 1)
            else:
                self.displayImage(self.currentImage)
            
    def changeCountText(self, i):
        txt = str(i+1) + '/' + str(len(self.images))
        self.imagecounttext.configure(text=txt)
        self.imagecounttext.text = txt
            
    def changeTitleText(self, i):
        txt = self.titles[i].split('/')
        txt = txt[len(txt)-1]
        self.imagetitle.configure(text=txt)
        self.imagetitle.text = txt
    
    def displayImage(self, i):
        self.displayPilImage(self.images[i])
        self.changeCountText(i)
        self.changeTitleText(i)
        self.currentImage = i
        
    def showPreds(self):
        if len(self.preds) > 0 :
            txt = ''
            for pred in self.preds:
                predPct = utils.predPct(pred[1])
                txt += str(pred[0]) + ' : ' + str(format(pred[1]*100,'.2f')) + ' -> ' + str(predPct) + '\n'
            self.setPredsText(txt)           
            
    def setPredsText(self, txt):
        self.predtext.configure(text=txt)
        self.predtext.text = txt
        
    def displayPilImage(self, pilImg):
        if isinstance(pilImg, np.ndarray):
            imgIn = Image.fromarray(pilImg)
        else:
            imgIn = pilImg
        self.image = imgIn.resize((math.floor(self.imw),math.floor(self.imh)))
        self.canvasimage = ImageTk.PhotoImage(self.image)   
        self.canvas.itemconfig(self.image_on_canvas, image=self.canvasimage)
        
    def zoomImage(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass
        else: return
        scale = 1.0
        if event.num == 5 or event.delta == -120:  
            i = min(self.imw, self.imh)
            if int(i * self.imscale) < 30: return
            self.imscale /= self.delta
            scale        /= self.delta
        if event.num == 4 or event.delta == 120:  
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return
            self.imscale *= self.delta
            scale        *= self.delta
        self.canvas.scale('all', x, y, scale, scale) 
        self.showImage()
        
    def showImage(self, event=None):
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region
        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            x = min(int(x2 / self.imscale), self.imw)   # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.imh)  # ...and sometimes not
            image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection 
       
    def gradfilenamesToTitles(self, raid512, raid1024, crop512):
        titles = []
        totalModels = raid512.get() + raid1024.get() + crop512.get()
        modelTypes = []
        if raid512.get():
            modelTypes.append('Raid512')
        if raid1024.get():
            modelTypes.append('Raid1024')
        if crop512.get():
            modelTypes.append('Crop512')
        for i in range(1, totalModels+1):
            lgt = len(self.oldfilenames)
            j = 0
            type = modelTypes[i-1]
            while j < lgt:
                txt = self.oldfilenames[j].split('/')
                txt = txt[len(txt)-1]
                txt = 'Gradcam ' + type + ' ' + txt
                titles.append(txt)
                j = j + 1      
        self.setTitles(titles)       
        
    def exportResults(self):
        if self.computed:
            try:
                f = open('results.csv')
                f.close()
                with open('results.csv', 'a+') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    if len(self.preds) == 3:
                        writer.writerow([len(self.images), self.oldfilenames, self.preds[0], self.preds[1], self.preds[2]])
                    elif len(self.preds) == 2:                   
                        writer.writerow([len(self.images), self.oldfilenames, self.preds[0], self.preds[1], 0])
                    elif len(self.preds) == 1:                   
                        writer.writerow([len(self.images), self.oldfilenames, self.preds[0], 0, 0])       
            except FileNotFoundError:
                with open('results.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(["Number", "Links", "Results1", "Results2", "Results3"])
                    if len(self.preds) == 3:
                        writer.writerow([len(self.images), self.oldfilenames, self.preds[0], self.preds[1], self.preds[2]])
                    elif len(self.preds) == 2:                   
                        writer.writerow([len(self.images), self.oldfilenames, self.preds[0], self.preds[1], 0])
                    elif len(self.preds) == 1:                   
                        writer.writerow([len(self.images), self.oldfilenames, self.preds[0], 0, 0])
        self.exported = True
        
    def normGradfilenamesToTitles(self, raid512, raid1024, crop512):
        titles = []
        totalModels = raid512.get() + raid1024.get() + crop512.get()
        modelTypes = []
        if raid512.get():
            modelTypes.append('Raid512')
        if raid1024.get():
            modelTypes.append('Raid1024')
        if crop512.get():
            modelTypes.append('Crop512')
        for i in range(1, totalModels+1):
            lgt = len(self.oldfilenames)
            j = 0
            type = modelTypes[i-1]
            while j < lgt:
                txt = self.oldfilenames[j].split('/')
                txt = txt[len(txt)-1]
                txt = 'NormGradcam ' + type + ' ' + txt
                titles.append(txt)
                j = j + 1      
        self.setTitles(titles)
      
    def showGradcams(self, raid512, raid1024, crop512):
        if self.computed:
            self.showingXrays = False
            gradcamImages = []
            if len(self.gradfilenames) > 0:
                self.gradfilenamesToTitles(raid512, raid1024, crop512)
                for filename in self.gradfilenames:
                    image = Image.open(filename)
                    gradcamImages.append(image)
            self.setImages(gradcamImages)
        
        
    def showNormGradcams(self, raid512, raid1024, crop512):
        if self.computed:
            self.showingXrays = False
            normGradcamImages = []
            if len(self.normgradfilenames) > 0:
                self.normGradfilenamesToTitles(raid512, raid1024, crop512)
                for filename in self.normgradfilenames:
                    image = Image.open(filename)
                    normGradcamImages.append(image)
            self.setImages(normGradcamImages)
    