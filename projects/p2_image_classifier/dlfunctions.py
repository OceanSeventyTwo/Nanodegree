import torch
import torchvision

import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

def load_train_data(train_dir, bs=64, Prob_flip=0, Rot_deg=0):
    train_transforms = transforms.Compose([transforms.CenterCrop(255),
                                     transforms.Resize(224),
                                     transforms.RandomHorizontalFlip(Prob_flip),
                                     transforms.RandomRotation(Rot_deg),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)

    return trainloader, train_data #Return trainloader, train_data

def load_vt_data(test_dir,val_dir,bs=64):
    vt_transforms = transforms.Compose([transforms.CenterCrop(255),
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_data = datasets.ImageFolder(test_dir , transform=vt_transforms)
    val_data = datasets.ImageFolder(val_dir , transform=vt_transforms)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=bs)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=bs)

    return testloader, valloader #Return testloader, valloader

def add_FF_Classifier(model, arch, o_layer,h_layers,p_drop=0,smax_dim=1):
    """ add_FF_Classifier intializes a classifier based on the model, hidden layers, output layer, and dropout
    Args:
        model (torchvision pretrained model object)
        arch (str):

        arch (str): (options)
            'resnet18'[default],'alexnet','vgg16','squeezenet','densenet','inception'
    Returns:

    """
    from collections import OrderedDict

    class_name = ''  # Initializes class name to idenitfy type of classifer.


    if arch == 'alexnet':
        i_layer = 9216
        class_name = 'classifier'
    elif arch == 'vgg16':
        i_layer = 25088
        class_name = 'classifier'
    elif arch ==  'squeezenet':
        i_layer = 512
        class_name = 'classifier'
    elif arch == 'densenet':
        i_layer = 2208
        class_name = 'classifier'
    elif arch == 'inception':
        i_layer = 2048
        class_name = 'fc'
    else:  #RESNET
        i_layer = 512
        class_name = 'fc'


    hidden_layer = []
    hidden_layer.extend(h_layers)
    hidden_layer.insert(0,i_layer)
    hidden_layer.append(o_layer)

    io_pairs = zip(hidden_layer[:-1],hidden_layer[1:])
    l_dict = OrderedDict()

    l = 0
    hl = len(hidden_layer)-2
    for a,b in io_pairs:
        l_dict['drop'+str(l)] = nn.Dropout(p_drop)
        l_dict['fc'+str(l)] = nn.Linear(a,b)
        if l < hl:
            l_dict['relu'+str(l)] = nn.ReLU()
        l += 1

    l_dict['softmax'] = nn.LogSoftmax(dim=smax_dim)


    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(l_dict)

    if class_name == 'fc':
        model.fc = classifier
        for param in model.fc.parameters():
            param.requires_grad = True

    elif class_name == 'classifier':
        model.classifier = classifier
        for param in model.classifier.parameters():
            param.requires_grad = True

    else:
        print('Error: custom classifier not loaded into model')

    return 

def init_model(arch=''): 
    """Used to intialize the pretrained torchvision model

    Args:
        arch (str): (options)
            'resnet18'[default],'alexnet','vgg16','squeezenet','densenet','inception'
    Returns:
        model (torchvision model object): Returns the pretrained model
        arch (str): returns arch (provides output - for DEFAULT case)


    'resnet18' = models.resnet18()  # [DEFAULT]
    'alexnet' = models.alexnet()
    'vgg16' = models.vgg16()
    'squeezenet' = models.squeezenet1_0()
    'densenet' = models.densenet161()
    'inception' = models.inception_v3()
    """


    import torchvision.models as models

    model = []


    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch ==  'squeezenet':
        model = models.squeezenet1_0(pretrained=True)
    elif arch == 'densenet':
        model = models.densenet161(pretrained=True)
    elif arch == 'inception':
        model = models.inception_v3(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)
        arch = 'resnet18'

    return model,arch #Returns model,arch

def validation(model, vtloader, crit, dev):
    model = model.to(dev)
    model = model.eval()

    vt_loss = 0
    accuracy = 0
    for jj, (images, labels) in enumerate(vtloader):

        images = images.to(dev)
        labels = labels.to(dev)

        output = model.forward(images)
        vt_loss += crit(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return vt_loss, accuracy

def train_mod(model, trn_ldr, v_ldr, dev, opt, crit, epochs=1, print_every=10):
    steps = 0
    for e in range(epochs):
        model = model.to(dev)
        model.train()
        running_loss = 0

        for ii, (images, labels) in enumerate(trn_ldr):
            images = images.to(dev)
            labels = labels.to(dev)

            steps += 1

            opt.zero_grad()

            output = model.forward(images)
            loss = crit(output, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()


            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    val_loss, accuracy = validation(model, v_ldr, crit, dev)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Cross_val Loss: {:.3f}.. ".format(val_loss/len(v_ldr)),
                      "Val Accuracy: {:.3f}".format(accuracy/len(v_ldr)))

                running_loss = 0

                # Make sure training is back on
                model.train()

    model.eval()
    return e

def save_mod_v1(filename, model,arch,n_classes,h_layers,lr, p_drop,opt,last_epoch):
    ''' Use to save the state of a trained torchvision model for future training or evaluation

    Args:
        filename (str): Model filename (*.pth) that saves the state of a torchvision model
        model (torchvision model object): torchvision model that is being saved
        arch (str):  pretrained model archetecture
        n_classes (int):  number of classes that are being evaluated
        h_layers (list of int): list of sizes(int) of the hidden layers (e.g. [200, 100, 50]; three hidden layers of 200, 100 and 50)
        p_drop : probability of dropout
        opt: optimizer state (Currently only Adam optmizer with predefined)
        last_epoch: last epoch trained

    '''

    checkpoint = {'output_size': n_classes, #output
                  'hidden_layers': h_layers, #list of hidden layers
                  'p_drop' : p_drop, #dropout rate
                  'lr' : lr, #learn rate
                  'state_dict': model.state_dict(),
                  'tvision model name': arch,
                  'opt_state_dict' : opt.state_dict(),
                  'last epoch' : last_epoch}
    torch.save(checkpoint, filename)
    print('Checkpoint saved: ', filename)

    return

def load_model_v1(filename):
    ''' Load model saved using save_mod_v1

        Args:
            filename (str):  loads *.pth file.  use .pth in the filename call

        Return:
            model (pretrained torchvision model)
            optimizer (Adam optimizer ) : returns optimizer with saved state
            last epoch (int):  returns last epoch trained
            cp (dict): returns saved dict from save_mod_v1
    '''
    model = 0
    arch = ''
    cp = torch.load(filename,map_location=lambda storage, loc: storage)  # loadcheck point

    #model = models.resnet18(pretrained=True)
    ### LOAD MODEL ###
    model,arch = init_model(cp['tvision model name'])
    add_FF_Classifier(model,arch,cp['output_size'],cp['hidden_layers'],p_drop=cp['p_drop'])
    model.load_state_dict(cp['state_dict'])

    ### LOAD OPTIMIZER

    if (arch == 'alexnet') or (arch == 'vgg16') or (arch ==  'squeezenet') or (arch == 'densenet'):
        opt = optim.Adam(model.classifier.parameters(), cp['lr'])
    else:
        opt = optim.Adam(model.fc.parameters(), cp['lr'])

    opt.load_state_dict(cp['opt_state_dict'])

    return model, opt, cp



    #resize to 256 on long dimention
    im.thumbnail(size)
    
    #center crop to 224 x 224
    width, height = im.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height +224)/2

    im = im.crop((left, top, right, bottom))
    
    #Create np array
    np_img = np.array(im)
    np_img = np_img/255         # convert 0-255 -> 0-1
    np_img = (np_img - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225] #normalize
    np_img = np.transpose(np_img,[2,0,1]) #move color channel to the first dimention
    
    imshow(np_img)
    
    return np_img

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
        
    size = 256,256
    csize = 224
      
    im = Image.open(image)
    
    #resize to 256 on long dimention
    im.thumbnail(size)
    
    #center crop to 224 x 224
    width, height = im.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height +224)/2

    im = im.crop((left, top, right, bottom))
    
    #Create np array
    np_img = np.array(im)
    np_img = np_img/255         # convert 0-255 -> 0-1
    np_img = (np_img - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225] #normalize
    np_img = np.transpose(np_img,[2,0,1]) #move color channel to the first dimention
    
    #imshow(np_img)
    
    return np_img
    
def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval()    
    np_img = process_image(image_path)
    t_img = torch.from_numpy(np_img).float().to(device)
    output = model.forward(t_img[None,:,:,:])
    
    p,c = output.topk(topk)
    
    probs = p[0].cpu().detach().numpy()
    classes = c[0].cpu().detach().numpy()

    return np.exp(probs),(classes+1)  