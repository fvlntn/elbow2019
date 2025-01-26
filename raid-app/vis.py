from PIL import Image, ImageDraw
import math

def concatImage(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
    
def getConcatImage(filenames):
    size = math.floor(600/(len(filenames)+1))
    im = Image.open(filenames[0]).resize((300,size))
    for i in range(len(filenames)):
        if i > 0:
            nextIm = Image.open(filenames[i]).resize((300,size))
            im = concatImage(im, nextIm)
    return im    
    
def cropImage(image, i, box):    
    cropImage = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3]))).resize((512,512)) 
    cropImage.save('xrays_temp/crop_' + str(i) + '.png')
    return cropImage
    
def rectImage(image, i, box):   
    rectImage = image.copy()
    drawing = ImageDraw.Draw(rectImage)
    drawing.rectangle([(box[0], box[1]), (box[2], box[3])])   
    rectImage.save('xrays_temp/rect_' + str(i) + '.png')
    return rectImage  
    
def resultImage(image, predsFull, predsCrop, fullModels, cropModels):
    resultImage = image.copy()
    drawing = ImageDraw.Draw(resultImage)
    models = []
    for predModels in [fullModels, cropModels]:
        for predModel in predModels:
            models.append(predModel)
    for preds in [predsFull, predsCrop]:    
        for i, pred in enumerate(preds):
            drawing.text((10,300-20*i), "Preds " + str(models[i].name) + ': ' + str(pred), fill=(255,255,255,255))
    return resultImage