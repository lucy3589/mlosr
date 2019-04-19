
import random
import numpy as np


### Setting hyperparameters
class Hyperparameters():
	def __init__(self):
		#
		self.batch_size					 = 64
		self.iterations					 = 500  #epochs
		self.lr							 = 3e-4
		self.alpha						 = 0.5
		self.sigma						 = 0.000000000001
		self.gain						 = 1.00
		self.no_init_filters			 = 64
		self.latent_size				 = 600
		self.betas						 = 0.5
		self.depth 						 = 92
		self.growth_rate				 = 24

		#
		self.stats_frequency			 = 1000
		self.image_channel				 = 3
		self.image_size					 = 32
		
		#
		self.gpu	 					 = True
		self.verbose 					 = True

		#
		self.dataset_name				 = 'cifar10'
		self.model_mode                  = 'train'
		self.mlosr_model				 = 'vggnet'
		self.dataset_path 				 = '../../dataset/'
		self.method						 = 'mlosr'
		self.dataset_file_format		 = 'hdf5'
		self.dist_type					 = 'L1'
		
		#
		self.no_total					 = 11
		self.no_closed					 = 10
		self.no_open 					 =  1

		#
		self.kwn, self.unk = GetKwnUnkClasses(self.no_total, self.no_closed, self.no_open, 'sequential')

		#
		self.tail_size 					 = 20
		self.open_alpha					 = self.no_closed
		self.dist_measure				 = 'euclidean'
		self.labels_name 				 = ['zero','one','two','three','four','five','six','seven','eight','nine','zero1']


		#
		self.HEADER		= '\033[95m'
		self.BLUE		= '\033[94m'
		self.GREEN		= '\033[92m'
		self.YELLOW		= '\033[93m'
		self.FAIL		= '\033[91m'
		self.ENDC		= '\033[0m'
		self.BOLD		= '\033[1m'
		self.UNDERLINE	= '\033[4m'

def GetKwnUnkClasses(no_total, no_closed, no_open, magic_word):

	if(magic_word=='sequential'):
		kwn = np.asarray(range(no_closed))
		unk = no_closed+np.asarray(range(np.min((no_open,no_total-no_closed))))
		print kwn
		print unk
	elif(magic_word=='random'):
		rand_id  = np.asarray(random.sample(range(no_total-no_closed),no_open))
		kwn = np.sort(np.asarray(random.sample(range(no_total),no_closed)))
		unk = np.asarray(np.where(np.in1d(np.asarray(range(no_total)),kwn)==False))[0,rand_id[0:no_open]]
		print kwn
		print unk
	elif(magic_word=='manual'):
		kwn = np.asarray([9])
		unk = np.asarray([9])
		print kwn
		print unk
	else:
		print('ERROR: known unknown split type not available')

	return kwn, unk

hyper_para = Hyperparameters()