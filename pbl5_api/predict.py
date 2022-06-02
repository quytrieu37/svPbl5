from PIL import Image
import torch
from .model import ResNet , Bottleneck
from collections import namedtuple
import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])

device = torch.device('cpu')
model = ResNet(config=resnet50_config, output_dim=12)
model.load_state_dict(torch.load("./model_leaf_plant_disease_detection_model_v1.pt", map_location = device))


model = model.to(device)
classes = ['chili healthy','chili leaf curl', 'chili leaf spot', 'chili whitefy','chili yellowish', 'Corn(maize) Cercospora_leaf_spot Gray_leaf_spot',
'Corn(maize) Common rust_', 'Corn(maize) healthy','Corn(maize) Northern Leaf Blight','Potato Early_blight','Potato healthy','Potato Late_blight']


pretrained_means = [0.4581, 0.5027, 0.3906]
pretrained_stds = [0.1798, 0.1671, 0.1852]
test_transforms = transforms.Compose([
                                      transforms.Resize(size=255),
                                      transforms.CenterCrop(size=244),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean = pretrained_means, 
                                          std = pretrained_stds)
                                      ])

def single_prediction(image_path):
    path = '.' + image_path;  
    image = Image.open(path)

    # get normalized image
    img_normalized = test_transforms(image).float()
    img_normalized = img_normalized.unsqueeze_(0)
    with torch.no_grad():
      model.eval()
      output =model(img_normalized)
      index = output.data.cpu().numpy().argmax()
    #   print("Original : ", image_path[22:-4])
      return classes[index]

    # output = output.detach().numpy()
    # index = np.argmax(output)
    # return index
    # pred_csv = data["disease_name"][index]
    # print(pred_csv)



