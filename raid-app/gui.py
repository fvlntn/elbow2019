class ImageWindow:
    def __init__(self):
        self.label = tk.Label(gui)
        self.label.place(x=75, y=150)
        self.showPreviousButton = tk.Button(text="Previous", width=6, height=2, command=showPreviousImage)
        self.showPreviousButton.place(x=100,y=460)        
        self.showNextButton = tk.Button(text="Next", width=6, height=2, command=showNextImage)
        self.showNextButton.place(x=500,y=460)
        self.currentImage = 0
        self.images = []    

    def setImages(self, images):
        self.images = images.copy()
        
    def showNextImage(self):
        if len(self.images) != 0:
            if self.currentImage == len(self.images) - 1:
                nextImage = 0
            else:            
                nextImage = self.currentImage +  1
            self.displayImagesOnGUI(images[nextImage])
            self.currentImage = nextImage
            
    def showPreviousImage(self):    
        if len(self.images) != 0:            
            if self.currentImage == 0:
                nextImage = len(self.images) - 1
            else:        
                nextImage = self.currentImage -  1
            displayImagesOnGUI(images[nextImage])
            self.self.self.currentImage = nextImage
            
    def displayImagesOnGUI(pilImg):
        img = ImageTk.PhotoImage(pilImg.resize((500,300)))
        self.label.configure(image=img)
        self.label.image = img