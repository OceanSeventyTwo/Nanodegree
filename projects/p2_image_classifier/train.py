import argparse
from dlfunctions import *

last_epoch = 0
parser = argparse.ArgumentParser(prog='train.py')

parser.add_argument("PATH",help='Directory where images are stored')
parser.add_argument('-a','--arch', default='resnet',help='Pretrained torch model arch')
parser.add_argument('-hl','--hidden_layer', type=int, nargs='+',help='list of hidden layers (e.g [512,256,128])',default=[])
parser.add_argument('-lr','--learn_rate', type=float, help='learning rate', default=0.001)
parser.add_argument('-bz','--batch_size', type=int, help='batch size', default=64)
parser.add_argument('-dp','--dropout', type=float ,default = 0,help='probability of dropout in dropout layers')
parser.add_argument('-e','--epochs',type=int, default=1, help='number of epochs to train')
parser.add_argument('-g','--GPU',action='store_true',help='use GPU for training')
parser.add_argument('-sd','--save_dir',help='Directory where to save checkpoint')
parser.add_argument('-pt','--model_ckpt',default='model.pth',help='checkpoint filename')

args = parser.parse_args()
#print(args.__dict__)

# If GPU is requested AND availible: 
if args.GPU == True:
    if torch.cuda.is_available() == True:
        device = torch.device('cuda:0')
        print('Device: ',device)
    else:
        device = torch.device('cpu')
        print('GPU not availible... Device:', device)
else:
    device = torch.device('cpu')
    print('Device: ',device)

#Initialize Dataloaders

train_dir = args.PATH + '/train'
val_dir = args.PATH + '/valid'
test_dir = args.PATH + '/test'

bz = args.batch_size
trainloader,train_data = load_train_data(train_dir,bz,0,0)
testloader,valloader = load_vt_data(test_dir,val_dir,bz)
n_classes = len(train_data.classes)   

# Initialize pretrained model and add specified classifier
arch = args.arch
model,arch = init_model(arch)
add_FF_Classifier(model, arch, n_classes,args.hidden_layer,p_drop=args.dropout,smax_dim=1)

# optimizer and criterion
if arch == 'alexnet' or arch == 'vgg16' or arch == 'squeezenet' or arch == 'densenet':
    optimizer = optim.Adam(model.classifier.parameters(),args.learn_rate)
    print(model.classifier)
elif arch == 'resnet18' or arch == 'inception':
    optimizer = optim.Adam(model.fc.parameters(),args.learn_rate)
    print(model.fc)

criterion = nn.NLLLoss()

#Train model per parameters
last_epoch = train_mod(model, trainloader, valloader, device, optimizer, criterion, epochs=args.epochs, print_every=20)
# Save model checkpoint
   ## concatenate save file and dir
if args.save_dir == None:
    save_dir = ''
else:
    save_dir = args.save_dir
filename = save_dir + args.model_ckpt
  ## save checkpoint
save_mod_v1(filename, model,arch ,n_classes,args.hidden_layer,args.learn_rate, args.dropout ,optimizer,last_epoch)
#print(test statement)