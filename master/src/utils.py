
import os
import sys
import cv2
import copy
import h5py
import time
import torch
import pandas
import random
import scipy.io

import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
# import matplotlib.pyplot as plt
import torch.functional as func
import torchvision.datasets as dset
import torch.nn.functional as nnfunc
import torchvision.models as models
import scipy.spatial.distance as spd
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms

from torch.autograd import Variable
from itertools import cycle, islice
from sklearn import cluster, datasets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.append('../models/')

from models import *

cfg = {
    'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

def DataLoaderCustom(hyper_para, required_data):

	normMean = [0.49139968, 0.48215827, 0.44653124]
	normStd  = [0.24703233, 0.24348505, 0.26158768]
		
	## Data loading
	if(required_data=='train' or required_data=='trainval' or required_data=='all' or required_data=='traintest'):
		train_data  = GetData(hyper_para.dataset_path+hyper_para.dataset_name+'/'+'train_data.mat', hyper_para.dataset_file_format)['trainData']
		train_label = GetData(hyper_para.dataset_path+hyper_para.dataset_name+'/'+'train_label.mat', hyper_para.dataset_file_format)['trainLabel']

		if(hyper_para.dataset_file_format=='hdf5'):
			train_data = np.swapaxes(np.swapaxes(np.swapaxes(train_data,2,3),1,2),0,1)
			train_label = np.swapaxes(train_label,0,1)

		train_label = np.asarray(train_label, dtype=float)
		if(hyper_para.tanh_flag==True):
			train_data  = 2.0*train_data-1.0                                                      # make it between -1 to 1
		# for i in range(3):
		# 		train_data[:,i,:,:] = (train_data[:,i,:,:]-normMean[i])/normStd[i]
		temp_id       = np.where( np.in1d(train_label, hyper_para.kwn+1) == True )
		train_data    = train_data[temp_id[0],:,:,:]
		train_label   = train_label[temp_id[0]]
		no_train_data = np.shape(train_data)[0]

		for i in range(np.shape(hyper_para.kwn)[0]):
			train_label[train_label==(hyper_para.kwn[i]+1)] = i+1

		train_label = train_label[:,0]-1
		rand_id     = np.random.permutation(no_train_data)
		
		train_data  = train_data[rand_id,:,:,:]
		train_label = train_label[rand_id]

		train_data  = torch.from_numpy(train_data)
		train_label = torch.from_numpy(train_label)

	if(required_data=='validation' or required_data=='trainval' or required_data=='testval' or required_data=='all'):
		validation_data  = GetData(hyper_para.dataset_path+hyper_para.dataset_name+'/'+'validation_data.mat', hyper_para.dataset_file_format)['validationData']
		validation_label = GetData(hyper_para.dataset_path+hyper_para.dataset_name+'/'+'validation_label.mat', hyper_para.dataset_file_format)['validationLabel']

		if(hyper_para.dataset_file_format=='hdf5'):
			validation_data = np.swapaxes(np.swapaxes(np.swapaxes(validation_data,2,3),1,2),0,1)
			validation_label = np.swapaxes(validation_label,0,1)

		validation_label = np.asarray(validation_label, dtype=float)
		if(hyper_para.tanh_flag==True):
			# validation_data  = 2.0*validation_data-1.0                                                      # make it between -1 to 1
			for i in range(3):
				validation_data[:,i,:,:] = (validation_data[:,i,:,:]-normMean[i])/normStd[i]

		temp_id       = np.where( np.in1d(validation_label, hyper_para.kwn+1) == True )
		validation_data    = validation_data[temp_id[0],:,:,:]
		validation_label   = validation_label[temp_id[0]]
		no_validation_data = np.shape(validation_data)[0]

		for i in range(np.shape(hyper_para.kwn)[0]):
			validation_label[validation_label==(hyper_para.kwn[i]+1)] = i+1

		validation_label = validation_label[:,0]-1
		rand_id     = np.random.permutation(no_validation_data)
		
		validation_data  = validation_data[rand_id,:,:,:]
		validation_label = validation_label[rand_id]

		validation_data  = torch.from_numpy(validation_data)
		validation_label = torch.from_numpy(validation_label)
	
	if(required_data=='test' or required_data=='testval' or required_data=='traintest' or required_data=='all'):
		test_data  = GetData(hyper_para.dataset_path+hyper_para.dataset_name+'/'+'test_data.mat', hyper_para.dataset_file_format)['testData']
		test_label = GetData(hyper_para.dataset_path+hyper_para.dataset_name+'/'+'test_label.mat', hyper_para.dataset_file_format)['testLabel']

		if(hyper_para.dataset_file_format=='hdf5'):
			test_data = np.swapaxes(np.swapaxes(np.swapaxes(test_data,2,3),1,2),0,1)
			test_label = np.swapaxes(test_label,0,1)

		test_label = np.asarray(test_label, dtype=float)
		if(hyper_para.tanh_flag==True):
			test_data  = 2.0*test_data-1.0                                                        # make it between -1 to 1
		# 	for i in range(3):
		# 		# test_data[:,2-i,:,:] = (test_data[:,2-i,:,:]-normMean[2-i])/normStd[2-i]
		# 		test_data[:,i,:,:] = (test_data[:,i,:,:]-normMean[i])/normStd[i]

		temp_id          = np.where( np.in1d(test_label, hyper_para.unk) == True )
		test_data_unk    = test_data[temp_id[0],:,:,:]
		test_label_unk   = test_label[temp_id[0]]
		no_test_data_unk = np.shape(test_data_unk)[0]

		temp_id      = np.where( np.in1d(test_label, hyper_para.kwn) == True )
		test_data    = test_data[temp_id[0],:,:,:]
		test_label   = test_label[temp_id[0]]
		no_test_data = np.shape(test_data)[0]

		for i in range(np.shape(hyper_para.kwn)[0]):
			test_label[test_label==(hyper_para.kwn[i]+1)] = i+1

		test_label = test_label[:,0]-1
		rand_id    = np.random.permutation(no_test_data)
		test_data  = test_data[rand_id,:,:,:]
		test_label = test_label[rand_id]

		test_data  = torch.from_numpy(test_data)
		test_label = torch.from_numpy(test_label)

		test_label_unk = test_label_unk[:,0]-1
		rand_id        = np.random.permutation(no_test_data_unk)
		test_data_unk  = test_data_unk[rand_id,:,:,:]
		test_label_unk = test_label_unk[rand_id]
		
		test_data_unk  = torch.from_numpy(test_data_unk)
		test_label_unk = torch.from_numpy(test_label_unk)

	if(required_data=='train'):
		return train_data, train_label, no_train_data
	if(required_data=='validation'):
		return validation_data, validation_label, no_validation_data
	if(required_data=='test'):
		return test_data, test_label, test_data_unk, test_label_unk, no_test_data, no_test_data_unk
	if(required_data=='trainval'):
		return train_data, train_label, validation_data, validation_label, no_train_data, no_validation_data
	if(required_data=='traintest'):
		return train_data, train_label, test_data, test_label, test_data_unk, test_label_unk, no_train_data, no_test_data, no_test_data_unk
	if(required_data=='all'):
		return train_data, train_label,\
			   validation_data, validation_label,\
			   test_data, test_label,\
			   test_data_unk, test_label_unk,\
			   no_train_data, no_validation_data,\
			   no_test_data, no_test_data_unk

def GetData(data_file_path, file_type):

	if(file_type=='hdf5'):
		data = h5py.File(data_file_path, 'r')
	elif(file_type=='.mat'):
		data  = scipy.io.loadmat(data_file_path)
	else:
		print('ERROR: File type not supported in this version of code.')
	return data

def OneHot(labels, n_classes):
	onehot = torch.FloatTensor(labels.size()[0], n_classes)
	labels = labels.data
	if labels.is_cuda:
		onehot = onehot.cuda()
	onehot.zero_()
	onehot.scatter_(1, labels.view(-1, 1), 1)
	return onehot

def GetDistance(input1, input2, hyper_para):

	dist = None
	if(hyper_para.dist_type == 'L1'):
		dist = torch.mean(torch.abs((input1-input2)))
	elif(hyper_para.dist_type == 'L2'):
		dist = torch.sqrt(torch.mean((input1-input2)*(input1-input2)))
	elif(hyper_para.dist_type == 'D'):
		D = torch.load('../../temp_folder/D.pth')
		dist = D(input1)
	else:
		print('ERROR: Unidentified distance type')

	return dist

def AddNoise(inputs, sigma):

	noise_shape = np.shape(inputs)
	
	noise = np.random.normal(0, sigma, noise_shape)
	noise = torch.from_numpy(noise)
	noise = Variable(noise).float()

	if(inputs.is_cuda):
		outputs = inputs + noise.cuda()
	else:
		outputs = inputs + noise

	return outputs

def SetupImageFolders(hyper_para):

	os.system('rm -rf ../../save_folder/results/'   + hyper_para.dataset_name + '/encoded_images/')
	os.system('rm -rf '+'../../save_folder/results/'+ hyper_para.dataset_name +'/test_features_wsvm/')
	os.system('rm -rf '+'../../save_folder/results/'+ hyper_para.dataset_name +'/train_features_wsvm/')

	os.system('mkdir ../../save_folder/results/'  + hyper_para.dataset_name + '/encoded_images/')
	os.system('mkdir ../../save_folder/results/'  + hyper_para.dataset_name + '/encoded_images/'+'/kwn/')
	os.system('mkdir ../../save_folder/results/'  + hyper_para.dataset_name + '/encoded_images/'+'/unk/')
	os.system('mkdir ../../save_folder/results/'  + hyper_para.dataset_name + '/test_features_wsvm/')
	os.system('mkdir ../../save_folder/results/'  + hyper_para.dataset_name + '/train_features_wsvm/')

def TestAccuracyMLOSR(hyper_para, cuda_flag=True):

	# load models
	if(hyper_para.mlosr_model=='densenet'):
		E = DenseNet10(growthRate=hyper_para.growth_rate, depth=hyper_para.depth, reduction=0.5, bottleneck=True, nClasses=10)
		C = DenseClassifier10(latent_size=hyper_para.latent_size)
		G = generatorM(latent_size=hyper_para.latent_size, batch_size=1, n=hyper_para.no_closed)
	elif(hyper_para.mlosr_model=='vggnet'):
		E = VGG(make_layers_vgg(cfg['A'], batch_norm=True))
		new_classifier = nn.Sequential(*list(E.classifier.children())[:-2])
		E.classifier = new_classifier
		C = DenseClassifier10(latent_size=4096)
		G = generatorM(latent_size=4096, batch_size=1, n=hyper_para.no_closed)

	E.load_state_dict(torch.load('../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'encoder'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth'))
	C.load_state_dict(torch.load('../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'classifier'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth'))
	
	if(cuda_flag):
		E.cuda()
		C.cuda()
	
	E.eval()
	C.eval()

	normMean = [0.49139968, 0.48215827, 0.44653124]
	normStd = [0.24703233, 0.24348505, 0.26158768]
	normTransform = transforms.Normalize(normMean, normStd)

	testTransform = transforms.Compose([
	    transforms.ToTensor(),
	    normTransform
	])
	
	testLoader = DataLoader(dset.CIFAR10(root='../../datasets/cifar10/', train=False, download=True, transform=testTransform), batch_size=1, shuffle=False)

	test_loss = 0
	incorrect = 0
	for data, target in testLoader:
		data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		output = C(E(data))
		pred = output.view(1, 10).data.max(1)[1] # get the index of the max log-probability
		incorrect += pred.ne(target.data).cpu().sum()

	nTotal = len(testLoader.dataset)
	err = 100.*incorrect/nTotal
	print('\n Error: {}/{} ({:.0f}%)\n'.format(incorrect, nTotal, err))

def MLOSR(hyper_para):

	# load models
	if(hyper_para.mlosr_model=='densenet'):
		E = DenseNet10(growthRate=hyper_para.growth_rate, depth=hyper_para.depth, reduction=0.5, bottleneck=True, nClasses=10)
		C = DenseClassifier10(latent_size=hyper_para.latent_size)
		G = generatorM(latent_size=hyper_para.latent_size, batch_size=hyper_para.batch_size, n=hyper_para.no_closed)
	elif(hyper_para.mlosr_model=='vggnet'):
		E = VGG(make_layers_vgg(cfg['A'], batch_norm=True))
		new_classifier = nn.Sequential(*list(E.classifier.children())[:-2])
		E.classifier = new_classifier
		C = DenseClassifier10(latent_size=4096)
		G = generatorM(latent_size=4096, batch_size=hyper_para.batch_size, n=hyper_para.no_closed)
	
	mse_criterion = nn.L1Loss()
	ce_criterion  = nn.CrossEntropyLoss()
	ac_scale = 1

	optimizer_e = optim.Adam(E.parameters(), lr=hyper_para.lr, betas=(0.5, 0.999))
	optimizer_g = optim.Adam(G.parameters(), lr=hyper_para.lr, betas=(0.5, 0.999))
	optimizer_c = optim.Adam(C.parameters(), lr=hyper_para.lr, betas=(0.5, 0.999))
	
	E.train(mode=True)
	G.train(mode=True)
	C.train(mode=True)
	
	np.random.seed(int(time.time()))
	random.seed(int(time.time()))

	running_tl = 0.0
	running_cc = 0.0
	running_rc = 0.0
	running_ri = 0.0

	if hyper_para.gpu:
		E.cuda()
		G.cuda()
		C.cuda()

	normMean = [0.49139968, 0.48215827, 0.44653124]
	normStd  = [0.24703233, 0.24348505, 0.26158768]
	normTransform = transforms.Normalize(normMean, normStd)

	# train_data, train_label, no_train_data = DataLoaderCustom(hyper_para, required_data='train')
	# single = train_data[train_label==0]
	# for i in range(3):
	# 	single[:,i,:,:] = (single[:,i,:,:]-normMean[i])/normStd[i]

	
	trainTransform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normTransform])
	
	trainLoader = DataLoader(dset.CIFAR10(root='../../datasets/cifar10/', train=True, download=True,transform=trainTransform), batch_size=hyper_para.batch_size, shuffle=True)

	print('  + Number of params: {}'.format(sum([p.data.nelement() for p in E.parameters()])))

	for i in range(int(hyper_para.iterations)):
		t1 = time.time()
		for batch_idx, (data, target) in enumerate(trainLoader):

			inputs, labels = Variable(data), Variable(target)

			inputs = Variable(inputs).float()
			labels = Variable(labels).long()

			if hyper_para.gpu:
				inputs					 = inputs.cuda()
				labels 					 = labels.cuda()
			m5 = E(inputs)

			temp = Variable(torch.zeros(64, 3, 32, 32)).cuda().float()
			loss_rc = mse_criterion(temp, temp)
			if(np.shape(m5)[0]==64):
				inputs_hat = G(m5, ac_scale)
				inputs_hat = 0.5*(inputs_hat+1)
				for l in range(3):
					inputs_hat[:,l,:,:] = (inputs_hat[:,l,:,:]-normMean[l])/normStd[l]
				loss_rc = mse_criterion(inputs_hat, inputs)

			loss_cc = ce_criterion(C(m5), labels)
			
			loss_tl = hyper_para.alpha * loss_rc + (1 - hyper_para.alpha) * loss_cc
			
			# Minimize loss
			optimizer_e.zero_grad()
			optimizer_c.zero_grad()
			optimizer_g.zero_grad()
			
			loss_tl.backward()

			optimizer_e.step()
			optimizer_c.step()
			optimizer_g.step()
			
			running_rc  += loss_rc.data
			running_cc  += loss_cc.data

		torch.save(E.state_dict(), '../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'encoder'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth')
		torch.save(C.state_dict(), '../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'classifier'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth')
		torch.save(G.state_dict(), '../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'decoder'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth')
		
		t2 = time.time()
		line =  hyper_para.BLUE   + '[' + str(format(i+1, '8d')) + '/'+ str(format(int(hyper_para.iterations), '8d')) + ']' + hyper_para.ENDC + \
				hyper_para.GREEN  + ' loss_rc: '     + hyper_para.ENDC + str(format(running_rc/hyper_para.stats_frequency, '1.8f'))  + \
				hyper_para.GREEN  + ' loss_cc: '     + hyper_para.ENDC + str(format(running_cc/hyper_para.stats_frequency, '1.8f')) + \
				hyper_para.YELLOW + ' time (min): '  + hyper_para.ENDC + str(int((t2-t1)*20.0))
		if(hyper_para.verbose):
			print(line)
		running_rc = 0.0
		running_cc = 0.0

	E.eval()
	C.eval()
	G.eval()

	if(hyper_para.gpu):
		E.cpu()
		C.cpu()
		G.cpu()
	
	torch.save(E.state_dict(), '../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'encoder'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth')
	torch.save(C.state_dict(), '../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'classifier'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth')
	torch.save(G.state_dict(), '../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'decoder'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth')

	TestAccuracyMLOSR(hyper_para)

def MLOSR_ablation(hyper_para):

	train_data, train_label, no_train_data = DataLoaderCustom(hyper_para, required_data='train')

	E_C = DCCA_Encoder()
	E_G = DCCA_Encoder()
	C = DCCA_Label_Classifier()
	G = DCCA_Decoder()
	
	mse_criterion = nn.L1Loss()
	ce_criterion  = nn.CrossEntropyLoss()

	optimizer_ec = optim.Adam(E_C.parameters(), lr=hyper_para.lr)
	optimizer_eg = optim.Adam(E_G.parameters(), lr=hyper_para.lr)
	optimizer_g = optim.Adam(G.parameters(), lr=hyper_para.lr)
	optimizer_c = optim.Adam(C.parameters(), lr=hyper_para.lr)
		
	E_C.train(mode=True)
	E_G.train(mode=True)
	C.train(mode=True)
	G.train(mode=True)
	
	np.random.seed(int(time.time()))
	random.seed(int(time.time()))

	running_cc = 0.0
	running_rc = 0.0
	running_ri = 0.0

	if(hyper_para.gpu):
		E_C.cuda()
		E_G.cuda()
		C.cuda()
		G.cuda()

	for i in range(int(hyper_para.iterations)):

		t1 = time.time()
		
		rand_id = np.asarray(random.sample(range(no_train_data), hyper_para.batch_size))
		rand_id = torch.from_numpy(rand_id)

		inputs = Variable(train_data[rand_id]).float()
		labels = Variable(train_label[rand_id]).long()

		if(hyper_para.gpu):
			inputs = inputs.cuda()
			labels = labels.cuda()
		
		if(hyper_para.separate_flag):
			print np.shape(G(E_C(inputs)))
			print jmd
			loss_cc = ce_criterion(C(E_C(inputs)), labels)
			loss_rc = mse_criterion(G(E_G(inputs)), inputs)
			
			optimizer_ec.zero_grad()
			optimizer_c.zero_grad()
			loss_cc.backward(retain_graph=True)
			optimizer_ec.step()
			optimizer_c.step()

			optimizer_eg.zero_grad()
			optimizer_g.zero_grad()
			loss_rc.backward(retain_graph=True)
			optimizer_eg.step()
			optimizer_g.step()
		else:
			loss_cc = ce_criterion(C(E_C(inputs)), labels)
			loss_rc = mse_criterion(G(E_C(inputs)), inputs)
			
			loss_t = loss_cc + loss_rc

			optimizer_ec.zero_grad()
			optimizer_c.zero_grad()
			optimizer_g.zero_grad()
			loss_t.backward(retain_graph=True)
			optimizer_ec.step()
			optimizer_c.step()
			optimizer_g.step()

		running_cc  += loss_cc.data[0]
		running_rc += loss_rc.data[0]
		
		if ((i%(hyper_para.stats_frequency)==(hyper_para.stats_frequency-1))):
			t2 = time.time()
			line =  hyper_para.BLUE   + '[' + str(format(i+1, '8d')) + '/'+ str(format(int(hyper_para.iterations), '8d')) + ']' + hyper_para.ENDC + \
					hyper_para.GREEN  + ' loss_rc: '    + hyper_para.ENDC + str(format(running_rc/hyper_para.stats_frequency, '1.8f')) + \
					hyper_para.GREEN  + ' loss_cc: '     + hyper_para.ENDC + str(format(running_cc/hyper_para.stats_frequency, '1.8f'))  + \
					hyper_para.YELLOW + ' time (min): ' + hyper_para.ENDC + str(int((t2-t1)*20.0))
			print(line)
			running_cc = 0.0
			running_rc = 0.0
			running_ri = 0.0
	
	E_C.eval()
	E_G.eval()
	C.eval()
	G.eval()
	
	if(hyper_para.gpu):
		E_C.cpu()
		E_G.cpu()
		C.cpu()
		G.cpu()

	torch.save(E_C.state_dict(), '../../temp_folder/E_C.pth')
	torch.save(E_G.state_dict(), '../../temp_folder/E_G.pth')
	torch.save(G.state_dict(), '../../temp_folder/G.pth')
	torch.save(C.state_dict(), '../../temp_folder/C.pth')

def MLOSR_test(hyper_para):

	SetupImageFolders(hyper_para)

	# load models
	if(hyper_para.mlosr_model=='densenet'):
		E = DenseNet10(growthRate=hyper_para.growth_rate, depth=hyper_para.depth, reduction=0.5, bottleneck=True, nClasses=10)
		C = DenseClassifier10(latent_size=hyper_para.latent_size)
		G = generatorM(latent_size=hyper_para.latent_size, batch_size=1, n=hyper_para.no_closed)
	elif(hyper_para.mlosr_model=='vggnet'):
		E = VGG(make_layers_vgg(cfg['A'], batch_norm=True))
		new_classifier = nn.Sequential(*list(E.classifier.children())[:-2])
		E.classifier = new_classifier
		C = DenseClassifier10(latent_size=4096)
		G = generatorM(latent_size=4096, batch_size=1, n=hyper_para.no_closed)
	
	ac_scale=1
	
	## load models
	E.load_state_dict(torch.load('../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'encoder'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth'))
	C.load_state_dict(torch.load('../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'classifier'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth'))
	G.load_state_dict(torch.load('../../save_folder/models/'+hyper_para.dataset_name+'/'+hyper_para.method+'/'+'decoder'+'_'+hyper_para.mlosr_model+'_'+str(hyper_para.iterations)+'.pth'))
	
	E.eval()
	C.eval()
	G.eval()

	E.cuda()
	C.cuda()
	G.cuda()

	normMean = [0.49139968, 0.48215827, 0.44653124]
	normStd = [0.24703233, 0.24348505, 0.26158768]
	normTransform = transforms.Normalize(normMean, normStd)

	testTransform = transforms.Compose([
	    transforms.ToTensor(),
	    normTransform
	])

	tensor2pil = transforms.ToPILImage()
	pil2tensor = transforms.ToTensor()
	
	testLoader = DataLoader(dset.CIFAR10(root='../../datasets/cifar10/', train=False, download=True, transform=testTransform), batch_size=1, shuffle=False)

	_, _, test_data_unk, test_label_unk, _, no_test_data_unk = DataLoaderCustom(hyper_para, required_data='test')
	
	incorrect=0
	for data, target in testLoader:
		data, target = data, target
		data, target = Variable(data).cuda(), Variable(target).cuda()
		output = C(E(data))
		pred = output.view(1, 10).data.max(1)[1] # get the index of the max log-probability
		incorrect += pred.ne(target.data).cpu().sum()

	nTotal = len(testLoader.dataset)
	err = 100.*incorrect/nTotal
	print('\n Error: {}/{} ({:.0f}%)\n'.format(incorrect, nTotal, err))

	nTotal = len(testLoader.dataset)
	kwn_mse = np.zeros((nTotal,))
	all_test_mse = np.zeros((nTotal+no_test_data_unk,))
	all_test_label = np.zeros((nTotal+no_test_data_unk,))
	all_test_score = np.zeros((nTotal+no_test_data_unk,hyper_para.no_closed))
	k=0
	i=0
	incorrect=0
	pred_label_test_g = np.zeros((nTotal,))
	tTotal = 0
	
	for data, target in testLoader:
		data, target = data, target
		data, target = Variable(data).cuda(), Variable(target)
		
		a = datetime.now()
		temp_img = G(E(data), ac_scale)

		temp_scores = C(E(data))
		temp_mse  = GetDistance(data, temp_img, hyper_para)
		temp_mse = torch.exp(-temp_mse/10.0+10.0)
		pred = temp_scores.view(1, 10).data.max(1)[1] # get the index of the max log-probability
		
		pred = temp_scores.view(1, 10).data.max(1)[1] # get the index of the max log-probability
		incorrect += pred.ne(target.cuda().data).cpu().sum()
		
		temp_img = 0.5*temp_img.view(hyper_para.image_channel,hyper_para.image_size,hyper_para.image_size) + 0.5
		
		output = tensor2pil(temp_img.data.cpu())
		output = testTransform(output)
		output = output.view(1, hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size)
		
		temp_mse  = GetDistance(data.cpu(), Variable(output), hyper_para)
		temp_mse  = temp_mse.data*torch.ones(1,1)
		temp_mse  = temp_mse.numpy()

		for l in range(3):
			data[:,l,:,:] = data[:,l,:,:]*normStd[l]+normMean[l]

		if(k<101):
			cv2.imwrite('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/kwn/'+str(i)+'_real_'+str(int(target.data.numpy()))+'.jpg', np.reshape(np.transpose(data.data.cpu().numpy())*255,(hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel)))
		if(i<101):
			cv2.imwrite('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/kwn/'+str(i)+'_zfake_'+str(int(target.data.numpy()))+'_'+str(int(51))+'.jpg', np.reshape(np.transpose(temp_img.data.cpu().numpy())*255,(hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel)))
			
		pred_label_test_g[i] = temp_mse
		kwn_mse[i] = temp_mse
		all_test_mse[k]   = temp_mse
		all_test_label[k] = target.data.numpy()
		all_test_score[k] = temp_scores.data.cpu().numpy()


		k += 1
		i += 1

	err = 100.*incorrect/nTotal
	print('\n Error: {}/{} ({:.0f}%)\n'.format(incorrect, nTotal, err))

	
	unk_unk_mse = np.zeros((no_test_data_unk,))
	
	for i in range(no_test_data_unk):
		
		temp = Variable(test_data_unk[i]).float()
		
		real_data = temp.view(hyper_para.image_channel,hyper_para.image_size,hyper_para.image_size)
		real_data = tensor2pil(real_data)
		real_data = testTransform(real_data)
		real_data = real_data.view(1, hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size)
		
		m5 = E(real_data.cuda())
		temp_scores = C(m5).data.cpu().numpy()
		oimg = np.transpose(temp.data.numpy())
		
		if(i<101):
			cv2.imwrite('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/unk/'+str(i)+'_zfake_'+str(int(test_label_unk[i]))+'_'+str(int(51))+'.jpg', np.reshape((oimg)*255,(hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel)))
		
		temp_img = G(m5, ac_scale).data.cpu()
		temp_data = 0.5*(temp_img+1)

		temp_data = temp_data.view(hyper_para.image_channel,hyper_para.image_size,hyper_para.image_size)
		temp_data = tensor2pil(temp_data)
		temp_data = testTransform(temp_data)
		temp_data = temp_data.view(1, hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size)

		temp_mse = GetDistance(temp_data, temp.data, hyper_para)
		temp_mse = temp_mse.data*torch.ones(1,1)
		temp_mse = temp_mse.numpy()

		mse = temp_mse

		temp_img = np.transpose(temp_img.data.numpy())
		
		for l in range(3):
			temp_data[:,l,:,:] = temp_data[:,l,:,:]*normStd[l]+normMean[l]
		
		if(i<101):
			cv2.imwrite('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/unk/'+str(i)+'_real_'+str(int(test_label_unk[i]))+'.jpg', np.reshape((temp_img)*255,(hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel)))
		
		unk_unk_mse[i]    = mse
		all_test_mse[k]   = mse
		all_test_label[k] = -1
		all_test_score[k] = temp_scores
		k += 1
	print('')

	# saving all the files
	scipy.io.savemat('../../save_folder/results/'+hyper_para.dataset_name+'/'+'mlosr_mse.mat',    {'mlosr_mse':all_test_mse})
	scipy.io.savemat('../../save_folder/results/'+hyper_para.dataset_name+'/'+'mlosr_scores.mat', {'mlosr_scores':all_test_score})
	scipy.io.savemat('../../save_folder/results/'+hyper_para.dataset_name+'/'+'label.mat',       {'label':all_test_label})
	scipy.io.savemat('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/kwn.mat',{'kwn':kwn_mse})
	scipy.io.savemat('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/unk_unk.mat',{'unk_unk':unk_unk_mse})

def MLOSR_test_ablation(hyper_para):

	SetupImageFolders(hyper_para)
	print hyper_para.kwn
	print hyper_para.unk

	# load models
	E_C = DCCA_Encoder()
	E_G = DCCA_Encoder()
	C = DCCA_Label_Classifier()
	G = DCCA_Decoder()
	
	ac_scale=1
	## load models
	if(hyper_para.separate_flag):
		E_C.load_state_dict(torch.load('../../temp_folder/E_C.pth'))
		E_G.load_state_dict(torch.load('../../temp_folder/E_G.pth'))
		C.load_state_dict(torch.load('../../temp_folder/C.pth'))
		G.load_state_dict(torch.load('../../temp_folder/G.pth'))
	else:
		E_C.load_state_dict(torch.load('../../temp_folder/E_C.pth'))
		E_G.load_state_dict(torch.load('../../temp_folder/E_C.pth'))
		C.load_state_dict(torch.load('../../temp_folder/C.pth'))
		G.load_state_dict(torch.load('../../temp_folder/G.pth'))
	
	E_C.eval()
	E_G.eval()
	C.eval()
	G.eval()

	E_C.cuda()
	E_G.cuda()
	C.cuda()
	G.cuda()

	test_data, test_label, test_data_unk, test_label_unk, no_test_data, no_test_data_unk = DataLoaderCustom(hyper_para, required_data='test')
	
	kwn_mse = np.zeros((no_test_data,))
	all_test_mse = np.zeros((no_test_data+no_test_data_unk,))
	all_test_label = np.zeros((no_test_data+no_test_data_unk,))
	all_test_score = np.zeros((no_test_data+no_test_data_unk,hyper_para.no_closed))
	k=0
	i=0
	incorrect=0
	pred_label_test_g = np.zeros((no_test_data,))
	
	for i in range(no_test_data):
		
		real_data = Variable(test_data[i]).float()
		real_data = real_data.view(1, hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size).cuda()
		
		m5 = E_C(real_data)
		temp_scores = C(m5).data.cpu().numpy()
		oimg = np.transpose(real_data.data.cpu().numpy())
		
		if(i<101):
			cv2.imwrite('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/kwn/'+str(i)+'_zfake_'+str(int(test_label_unk[i]))+'_'+str(int(51))+'.jpg', np.reshape((oimg)*255,(hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel)))
		
		temp_data = G(E_G(real_data)).data.cpu()
		temp_img = 0.5*(temp_data+1)

		temp_data = temp_data.view(1, hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size)

		temp_mse = GetDistance(temp_data, real_data.data.cpu(), hyper_para)
		temp_mse = temp_mse.data*torch.ones(1,1)
		temp_mse = temp_mse.numpy()

		mse = temp_mse

		temp_img = np.transpose(temp_img.data.numpy())
		
		if(i<101):
			cv2.imwrite('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/kwn/'+str(i)+'_real_'+str(int(test_label_unk[i]))+'.jpg', np.reshape((temp_img)*255,(hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel)))
		
		kwn_mse[i]    = mse
		all_test_mse[k]   = mse
		all_test_label[k] = test_label[i]
		all_test_score[k] = temp_scores
		k += 1
	print('')

		
	unk_unk_mse = np.zeros((no_test_data_unk,))
	
	for i in range(no_test_data_unk):
		
		real_data = Variable(test_data_unk[i]).float()
		real_data = real_data.view(1, hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size).cuda()
		
		m5 = E_C(real_data)
		temp_scores = C(m5).data.cpu().numpy()
		oimg = np.transpose(real_data.data.cpu().numpy())
		
		if(i<101):
			cv2.imwrite('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/unk/'+str(i)+'_zfake_'+str(int(test_label_unk[i]))+'_'+str(int(51))+'.jpg', np.reshape((oimg)*255,(hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel)))
		
		temp_data = G(E_G(real_data)).data.cpu()
		temp_img = 0.5*(temp_data+1)

		temp_data = temp_data.view(1, hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size)

		temp_mse = GetDistance(temp_data, real_data.data.cpu(), hyper_para)
		temp_mse = temp_mse.data*torch.ones(1,1)
		temp_mse = temp_mse.numpy()

		mse = temp_mse

		temp_img = np.transpose(temp_img.data.numpy())
		
		if(i<101):
			cv2.imwrite('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/unk/'+str(i)+'_real_'+str(int(test_label_unk[i]))+'.jpg', np.reshape((temp_img)*255,(hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel)))
		
		unk_unk_mse[i]    = mse
		all_test_mse[k]   = mse
		all_test_label[k] = -1
		all_test_score[k] = temp_scores
		k += 1
	print('')

	# saving all the files
	scipy.io.savemat('../../save_folder/results/'+hyper_para.dataset_name+'/'+'mlosr_mse.mat',            {'mlosr_mse':all_test_mse})
	scipy.io.savemat('../../save_folder/results/'+hyper_para.dataset_name+'/'+'mlosr_scores.mat',         {'mlosr_scores':all_test_score})
	scipy.io.savemat('../../save_folder/results/'+hyper_para.dataset_name+'/'+'label.mat',               {'label':all_test_label})
	scipy.io.savemat('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/kwn.mat',     {'kwn':kwn_mse})
	scipy.io.savemat('../../save_folder/results/'+hyper_para.dataset_name+'/encoded_images/unk_unk.mat', {'unk_unk':unk_unk_mse})
