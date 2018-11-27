import argparse
import json
from dlfunctions import *

parser = argparse.ArgumentParser(prog='predict.py')

parser.add_argument("im_PATH",help='Directory/file where image is stored')
parser.add_argument("md_PATH",help='Directory/file where checkpoint is stored')
parser.add_argument('-K','--top_K', type=int, default='1',help='Top # of classes to be displayed ')
parser.add_argument('-C','--category_names',type=str,default=None,help='*.json file')
parser.add_argument('-g','--GPU',action='store_true',help='use GPU for predicting')

args = parser.parse_args()

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


#LOAD MODEL
model, opt, cp = load_model_v1(args.md_PATH)
model = model.to(device)

#LOAD JSON File 
if args.category_names != None:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
##

#Load/Process Image
np_img = process_image(args.im_PATH)

#predict classes
probs,classes = predict(args.im_PATH,model,device,args.top_K)


# If JSON file is provided

if  args.category_names != None:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    cnames = [cat_to_name[str(x)] for x in (classes)]
    for x,y in zip(cnames,probs):
        print("Prediction: {:20}  |...| ".format(x), "Probability: {:.2%} ".format(y))
else:
    for x,y in zip(classes,probs):
        print("Prediction: {:4}   |...| ".format(x), "Probability: {:.2%} ".format(y))