import torch
from utils import default_transforms, _is_iterable
from torchvision import models
import vis
import numpy as np
import cv2

class DensenetModel:
    def __init__(self, name):
        self._model = models.densenet169(pretrained=True)
        num_features = self._model.classifier.in_features
        self._model.classifier = torch.nn.Linear(num_features, 1)
        self._model = torch.nn.DataParallel(self._model)
        self.name = name
        self._features_blobs = []
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
        
    def load(file, name):
        model = DensenetModel(name)
        model._model.load_state_dict(torch.load(file, map_location=model._device))
        return model
     
    def predict_for_study(self, images):
        if len(images) != 0:           
            self._model.eval()
            with torch.no_grad():
                if not _is_iterable(images):
                    images = [images]                
                if not isinstance(images[0], torch.Tensor):
                    images = [default_transforms()(img) for img in images]
                images = [1 - img for img in images]
                preds = self._model(torch.stack(images))
                studyPred = torch.sigmoid(torch.max(preds.data)).item()            
                return studyPred
                
    def _hook_feature(self, module, input, output):
        self._features_blobs.append(output.data.cpu().numpy())
        
    def _checkMin(self, image, min):
        if(np.min(image) < min):
            return False
        else:       
            return True
            
    def _checkMax(self, image, max):
        if(np.max(image) > max):
            return False
        else:       
            return True
            
    def _writeNewMin(self, min):
        with open('minmax', 'r') as file:
            data = file.readlines()
        if 'Raid512' in self.name:
            data[0] = str(min) + '\n'
        elif 'Raid1024' in self.name:
            data[2] = str(min) + '\n'
        elif 'Crop512' in self.name:
            data[4] = str(min) + '\n'
        with open('minmax', 'w') as file:
            file.writelines(data)
        
    def _writeNewMax(self, max):
        with open('minmax', 'r') as file:
            data = file.readlines()
        if 'Raid512' in self.name:
            data[1] = str(max) + '\n'
        elif 'Raid1024' in self.name:
            data[3] = str(max) + '\n'
        elif 'Crop512' in self.name:
            data[5] = str(max) + '\n'
        with open('minmax', 'w') as file:
            file.writelines(data) 
    
    def _returnCAM_normalize(self, feature_conv, weight_softmax, class_idx):
        with open('minmax', 'r') as minmax:
            data = minmax.readlines()
        if 'Raid512' in self.name:
            min = float(data[0])
            max = float(data[1])
        elif 'Raid1024' in self.name:
            min = float(data[2])
            max = float(data[3])
        elif 'Crop512' in self.name:
            min = float(data[4])
            max = float(data[5])
        size_upsample = (1024, 1024)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)            
            if 'Raid512' in self.name or 'Raid1024' in self.name:
                cam = cam[3:30, 3:30]
            if self._checkMin(cam, min) and self._checkMax(cam, max):                
                cam = cam - min
                cam_img = cam / ( max - min )
            elif not self._checkMin(cam, min) and self._checkMax(cam, max):
                newMin = np.min(cam)
                self._writeNewMin(newMin)
                cam = cam - newMin
                cam_img = cam / ( max - newMin )   
            elif self._checkMin(cam, min) and not self._checkMax(cam, max):
                newMax = np.max(cam)
                self._writeNewMax(newMax)
                cam = cam - min
                cam_img = cam / ( newMax - min )  
            else:
                newMin = np.min(cam)
                self._writeNewMin(newMin)
                newMax = np.max(cam)
                self._writeNewMax(newMax)
                cam = cam - newMin
                cam_img = cam / ( newMax - newMin ) 
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam    

    def _returnCAM(self, feature_conv, weight_softmax, class_idx):
        size_upsample = (1024, 1024)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            if 'Raid512' in self.name or 'Raid1024' in self.name:                
                cam = cam[3:30, 3:30]
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam
    
    def gradcam(self, images):
        if len(images) != 0:   
            self._model.eval()   
            with torch.no_grad():
                imagesTensor = images.copy()
                if not _is_iterable(imagesTensor):
                    imagesTensor = [imagesTensor]                
                if not isinstance(imagesTensor[0], torch.Tensor):
                    imagesTensor = [default_transforms()(img) for img in imagesTensor]
                gradcams = []
                for i, image in enumerate(imagesTensor):
                    self._features_blobs = []
                    a = list(list(self._model._modules.items())[0][1]._modules.items())[0][1]
                    a.register_forward_hook(self._hook_feature)
                    params = list(self._model.parameters())
                    weight_softmax = params[-2].data.cpu().numpy()
                    img_variable = torch.stack([image])
                    logit = self._model(img_variable)
                    h_x = torch.sigmoid(torch.t(logit)[0].data)#.type(torch.FloatTensor)
                    probs, idx = h_x.sort(0, True)
                    probs = probs.cpu().numpy()
                    idx = idx.cpu().numpy()
                    CAMs = self._returnCAM(self._features_blobs[0], weight_softmax, [idx])
                    width, height = images[i].size
                    cvimage = np.array(images[i])
                    cvimage = cvimage[:, :, ::-1].copy()
                    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
                    gradcam = heatmap * 0.3 + cvimage * 0.5
                    gradcams.append(gradcam)
                return gradcams
                
    def normGradcam(self, images):
        if len(images) != 0:   
            self._model.eval()   
            with torch.no_grad():
                imagesTensor = images.copy()
                if not _is_iterable(imagesTensor):
                    imagesTensor = [imagesTensor]                
                if not isinstance(imagesTensor[0], torch.Tensor):
                    imagesTensor = [default_transforms()(img) for img in imagesTensor]
                gradcams = []
                for i, image in enumerate(imagesTensor):
                    self._features_blobs = []
                    a = list(list(self._model._modules.items())[0][1]._modules.items())[0][1]
                    a.register_forward_hook(self._hook_feature)
                    params = list(self._model.parameters())
                    weight_softmax = params[-2].data.cpu().numpy()
                    img_variable = torch.stack([image])
                    logit = self._model(img_variable)
                    h_x = torch.sigmoid(torch.t(logit)[0].data)#.type(torch.FloatTensor)
                    probs, idx = h_x.sort(0, True)
                    probs = probs.cpu().numpy()
                    idx = idx.cpu().numpy()
                    CAMs = self._returnCAM_normalize(self._features_blobs[0], weight_softmax, [idx])
                    width, height = images[i].size
                    cvimage = np.array(images[i])
                    cvimage = cvimage[:, :, ::-1].copy()
                    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
                    gradcam = heatmap * 0.3 + cvimage * 0.5
                    gradcams.append(gradcam)
                return gradcams       
        
class DetectionModel:
    def __init__(self):
        self._model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        classes = ['Elbow']
        in_features = self._model.roi_heads.box_predictor.cls_score.in_features
        self._model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, len(classes) + 1)
        self._classes = ['__background__'] + classes
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
    def load(file):
        model = DetectionModel()
        model._model.load_state_dict(torch.load(file, map_location=model._device))
        return model
            
    def predict(self, images):
        if len(images) != 0:  
            images = [images] if not _is_iterable(images) else images            
            self._model.eval()        
            with torch.no_grad():              
                if not isinstance(images[0], torch.Tensor):
                    images = [default_transforms()(img) for img in images]                
                preds = self._model(images)
                preds = [{k: v.to(torch.device('cpu')) for k, v in p.items()} for p in preds]                 
            results = []        
            for pred in preds:
                result = ([self._classes[val] for val in pred['labels']], pred['boxes'], pred['scores'])
                results.append(result)            
            return results
            
    def getCropImages(self, images):  
        imagesCrop = []      
        predsDetection = self.predict(images)      
        for i, image in enumerate(images): 
            xmin = predsDetection[i][1][0][0].item()
            ymin = predsDetection[i][1][0][1].item()
            xmax = predsDetection[i][1][0][2].item()
            ymax = predsDetection[i][1][0][3].item()    
            box = [xmin, ymin, xmax, ymax]
            crop = vis.cropImage(image, i, box)
            rect = vis.rectImage(image, i, box)
            imagesCrop.append(crop)
        return imagesCrop
            
class RaidModel:
    def __init__(self, fullModels, cropModels, detectionModel):
        self.fullModels = fullModels
        self.cropModels = cropModels
        self.detectionModel = detectionModel
        allModels = []
        for models in [fullModels, cropModels]:
            for model in models:
                allModels.append(model)
        self.models = allModels