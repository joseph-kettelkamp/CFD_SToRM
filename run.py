import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from dataAndOperators import dop as dataAndOperators
from generator import generator
from optimize_generator import optimize_generator
from latentVariable import latentVariable
from ptflops import get_model_complexity_info

gpu = torch.device('cuda:0')

params = {'name': 'parameters',
          'directory': '',
          'device': gpu,
          'filename': "",
          'dtype': torch.float,

          'im_size': (340, 340),  # imaege size
          'nintlPerFrame': 6,  # interleaves per frame
          'nFramesDesired': 100,  # number of frames in the reconstruction
          'slice': 3,  # slice of the series to process, note that it begins with 0 in python
          'factor': 1,  # scale image by 1/factor to save compute time

          'gen_base_size': 40,  # base number of filters
          'gen_reg': 0.0005,  # regularization penalty on generator

          'siz_l': 8}  # number of latent parameters

params['directory'] = 'data/results/'
params['filename'] = 'data/'

# %% Level 1 training

nFramesDesired = 100
nScale = 15
params['nintlPerFrame'] = int(6 * nScale)
params['nFramesDesired'] = int(nFramesDesired/nScale)
params['lr_g'] = 1e-4
params['lr_z'] = 1e-4

# Reading and pre-processing the data and parameters
dop = dataAndOperators(params)

# Initializaition of the generator
G = generator(params)
G.weight_init()
G.cuda(gpu)

# Initialization of th elatent variables
alpha = np.zeros([8])*0.1
alpha = torch.FloatTensor(alpha).to(gpu)
z = latentVariable(params, init='ones', alpha=alpha)
# Training
G, z, train_hist, SER1 = optimize_generator(dop, G, z, params, lambda0=8750, train_epoch=1000, proj_flag=False)

#%% Level 2 training

G_old = G.state_dict()
z_old = z.z_

nFramesDesired = 100
nScale = 3

params['nintlPerFrame'] = int(6*nScale)
params['nFramesDesired'] = int(nFramesDesired/nScale)
params['lr_g'] = 1e-5
params['lr_z'] = 5e-3


# Change the number of frames
dop.changeNumFrames(params)

#Initialize the latent variables

z.z_ = z_old
z = latentVariable(params,z_in=z,alpha=alpha)
G.load_state_dict(G_old)
#Training
G,z1,train_hist,SER2 = optimize_generator(dop,G,z,params, lambda0=1000,train_epoch=600,proj_flag=True)

#%% Final training

G_old = G.state_dict()
z_old = z1.z_
params['nFramesDesired'] = nFramesDesired
params['nintlPerFrame'] = 6
params['lr_g'] = 1e-4
params['lr_z'] = 5e-3

dop.changeNumFrames(params)

z.z_ = z_old
z = latentVariable(params,z_in=z,alpha=alpha)
G.load_state_dict(G_old)

G,z,train_hist,SER3 = optimize_generator(dop,G,z,params, lambda0 = 100,train_epoch=700,proj_flag=True)

#Initialize the latent variables
#%% Final training

G_old = G.state_dict()
z_old = z1.z_
params['nFramesDesired'] = nFramesDesired
params['nintlPerFrame'] = 6
params['lr_g'] = 1e-4
params['lr_z'] = 5e-3

dop.changeNumFrames(params)

z.z_ = z_old
z = latentVariable(params,z_in=z,alpha=alpha)
G.load_state_dict(G_old)

G,z,train_hist,SER3 = optimize_generator(dop,G,z,params, lambda0 = 100,train_epoch=2000,proj_flag=True)


# %% Display the results
ztemp = z.z_.data.detach()

# %% Save the results

zs = z.z_.data.squeeze().cpu().numpy()
torch.save(G.state_dict(), "generator_param.pkl")
np.save('zs.npy', zs)

# %% Saving data to file
import imageio
from matplotlib.transforms import Bbox
import os

pad = nn.ReplicationPad2d((1, 0, 1, 0))
blur = nn.MaxPool2d(2, stride=1)

images = []
imagesU = []
imagesV = []
imagesP = []
my_dpi = 100  # Good default - doesn't really matter
h = params['im_size'][0]
w = params['im_size'][1]

dirname = params['filename'].replace('.mat', '/results')
dirname = dirname + str(params['slice'])
if not (os.path.exists(dirname)):
    os.makedirs(dirname)

for k in range(nFramesDesired):
    test_image = ztemp[k, :, :, :].unsqueeze(0)
    test_image = blur(pad(G(test_image)))
    U, V, P = torch.split(test_image, 1, dim=1)
    test_image1 = torch.sqrt(torch.square(U) + torch.square(V)).squeeze().data.cpu().numpy()
    U = U.squeeze().data.cpu().numpy()
    V = V.squeeze().data.cpu().numpy()
    P = P.squeeze().data.cpu().numpy()
    #test_image1 = test_image1[0, :, :] + test_image1[1, :, :] * 1j
    fig, ax = plt.subplots(1, figsize=(w / my_dpi, h / my_dpi), dpi=my_dpi)
    ax.set_position([0, 0, 1, 1])

    plt.imshow(P, cmap='gray')
    ax.axis('off')
    img_name = dirname + '/frame_P_' + str(k) + '.png'
    fig.savefig(img_name, bbox_inches=Bbox([[0, 0], [w / my_dpi, h / my_dpi]]), dpi=my_dpi)
    imagesP.append(imageio.imread(img_name))

    plt.imshow(U, cmap='gray')
    ax.axis('off')
    img_name = dirname + '/frame_U_' + str(k) + '.png'
    fig.savefig(img_name, bbox_inches=Bbox([[0, 0], [w / my_dpi, h / my_dpi]]), dpi=my_dpi)
    imagesU.append(imageio.imread(img_name))

    plt.imshow(V, cmap='gray')
    ax.axis('off')
    img_name = dirname + '/frame_V_' + str(k) + '.png'
    fig.savefig(img_name, bbox_inches=Bbox([[0, 0], [w / my_dpi, h / my_dpi]]), dpi=my_dpi)
    imagesV.append(imageio.imread(img_name))

    plt.imshow((abs(test_image1)), cmap='gray')
    ax.axis('off')
    img_name = dirname + '/frame_' + str(k) + '.png'
    fig.savefig(img_name, bbox_inches=Bbox([[0, 0], [w / my_dpi, h / my_dpi]]), dpi=my_dpi)
    plt.close()
    images.append(imageio.imread(img_name))
imageio.mimsave(dirname + '.gif', images, fps=10)
imageio.mimsave(dirname + 'U.gif', imagesU, fps=10)
imageio.mimsave(dirname + 'V.gif', imagesV, fps=10)
imageio.mimsave(dirname + 'P.gif', imagesP, fps=10)