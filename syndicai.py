import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

class PythonPredictor:
    def  __init__(self):
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
            nn.Linear(128, 24),).to(device)

        self.model.load_state_dict(torch.load('D:\FYP-ALL DATA\API\\enet.h5' , map_location=torch.device('cpu')))
        self.model.eval()

    def processImage(self, file):
        image_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        data_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize
            ])
        image = Image.open(file)
        img = data_transforms(image)
        img = torch.reshape(img , (1, 3, image_size, image_size))
        return self.model(img)

    def predict(self, img):
        allClasses = ['Basal Cell Carcinoma','Dariers', 'Hailey-Hailey Disease', 'Impetigo', 'Larva Migrans',        
            'Leprosy Borderline', 'Leprosy Lepromatous', 'Leprosy Tuberculoid', 'Lichen Planus',
            'Lupus Erythematosus Chronicus Discoides', 'Melanoma', 'Molluscum Contagiosum',
            'Mycosis Fungoides', 'Pityriasis Rosea', 'Porokeratosis Actinic' , 'Psoriasis',
            'Tinea Corporis', 'Tinea Nigra', 'Tungiasis', 'Epidermolysis Bullosa Pruriginosa',
            'Herpes Simplex', 'Neurofibromatosis', 'Papilomatosis Confluentes And Reticulate',
            'Pediculosis Capitis']
        out = self.processImage(img)
        _, predicted = torch.max(out.data, 1)
        allClasses.sort()
        labelPred = allClasses[predicted]
        return labelPred
        