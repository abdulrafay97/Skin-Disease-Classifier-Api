import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO
from helper import bytes_to_img

class PythonPredictor:
    def  __init__(self, config):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        
        self.data_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                self.normalize
            ])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.efficientnet_b2(pretrained=False).to(device)
        in_features = 1024
        self.model._fc = nn.Sequential(
            nn.BatchNorm1d(num_features=in_features),    
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(0.4),
            nn.Linear(128, 31),).to(device)

        self.model.load_state_dict(torch.load('enet.h5' , map_location=torch.device('cpu')))
        self.model.eval()

    def predictLabel(self, file):
        image_size = 224
        img = torch.reshape(file , (1, 3, image_size, image_size))
        return self.model(img)

    def predict(self, payload):
        allClasses = ['Actinic Keratosis', 'Basal Cell Carcinoma',
	      'Dariers', 'Dermatofibroma',
	      'Epidermolysis Bullosa Pruriginosa', 'Hailey-Hailey Disease',
	      'Herpes Simplex', 'Impetigo', 'Larva Migrans', 'Leprosy Borderline',
	      'Leprosy Lepromatous', 'Leprosy Tuberculoid', 'Lichen Planus',
	      'Lupus Erythematosus Chronicus Discoides', 'Melanoma',
	      'Molluscum Contagiosum', 'Mycosis Fungoides', 'Neurofibromatosis',
	      'Nevus', 'Papilomatosis Confluentes And Reticulate', 'Pediculosis Capitis',
	      'Pigmented Benign Keratosis', 'Pityriasis Rosea', 'Porokeratosis Actinic',
	      'Psoriasis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Tinea Corporis',
	      'Tinea Nigra', 'Tungiasis', 'Vascular Lesion']
        image_str = payload["url"]
        img_pil = bytes_to_img(image_str)
        img_tensor = self.data_transforms(img_pil[0])
        img_tensor.unsqueeze_(0)
        out = self.predictLabel(img_tensor)
        _, predicted = torch.max(out.data, 1)
        allClasses.sort()
        labelPred = allClasses[predicted]
        return labelPred
