from core import Model
import utils, visualize

model = Model.load('model_weights.pth', ['Elbow'])

images = [
    utils.read_image('dataset_val/image1901.png'), 
    utils.read_image('dataset_val/image1902.png'), 
    utils.read_image('dataset_val/image1903.png'), 
    utils.read_image('dataset_val/image1904.png')
    ]
    
predictions = model.predict(images)  # Get all predictions on an image

for i, image in enumerate(images):
    labels, boxes, scores = predictions[int(i)]
    print(labels, boxes, scores)
    
visualize.plot_prediction_grid(model, images, dim=(2, 2), figsize=(8, 8))
