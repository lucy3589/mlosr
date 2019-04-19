
### importing important libraries
import sys
import argparse

from utils import *



def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def str2float(frac_str):
	try:
		return float(frac_str)
	except ValueError:
		num, denom = frac_str.split('/')
		try:
			leading, num = num.split(' ')
			whole = float(leading)
		except ValueError:
			whole = 0
		frac = float(num) / float(denom)
		return whole - frac if whole < 0 else whole + frac

def main():

	parser = argparse.ArgumentParser()

	# optional arguments
	parser.add_argument("--batch_size"          , default=64                  , type=int      , help="batch size")
	parser.add_argument("--iterations"          , default=500                 , type=float    , help="desired iterations for training (not to be confused with epoch)")
	parser.add_argument("--alpha"               , default=0.5                 , type=float    , help="alpha parameter")
	parser.add_argument("--sigma"               , default=1e-12               , type=float    , help="sigma parameter")
	parser.add_argument("--gain"                , default=1e-0                , type=float    , help="gain parameter")
	parser.add_argument("--no_init_filters"     , default=64                  , type=int      , help="number of initial filters")
	parser.add_argument("--latent_size"         , default=600                 , type=int      , help="latent size")
	parser.add_argument("--betas"               , default=0.5                 , type=float    , help="betas parameter")
	parser.add_argument("--depth"               , default=92                  , type=int      , help="depth of densenet")
	parser.add_argument("--growth_rate"         , default=24                  , type=int      , help="growth of densenet")
	parser.add_argument("--stats_frequency"     , default=1000                , type=int      , help="frequency of display (not in use as of now)")
	parser.add_argument("--image_channel"       , default=3                   , type=int      , help="image channels")
	parser.add_argument("--image_size"          , default=32                  , type=int      , help="expected image size")
	parser.add_argument("--gpu"                 , default=True                , type=str2bool , help="change to False if you want to train on CPU (Seriously??)")
	parser.add_argument("--verbose"             , default=True                , type=str2bool , help="message control flag, set to False if not interested in code status messages")
	parser.add_argument("--dataset_path"        , default='../../datasets/'   , type=str      , help="folder name that has train and test mat-file")
	parser.add_argument("--dataset_name"        , default='cifar10'           , type=str      , help="dataset name")
	parser.add_argument("--model_mode"          , default='train'             , type=str      , help="valid arguments eval and train")
	parser.add_argument("--method"              , default='mlosr'             , type=str      , help="mlosr or mlosr_ablation")
	parser.add_argument("--dataset_file_format" , default='hdf5'              , type=str      , help="select file format")
	parser.add_argument("--mlosr_model"         , default='vggnet'            , type=str      , help="select vggnet or densenet")
	parser.add_argument("--dist_type"           , default='L1'                , type=str      , help="select distance measure type")
	parser.add_argument("--lr"                  , default=1e-4                , type=float    , help="set learning rate (default recommended)")
	parser.add_argument("--no_closed"           , default=10                  , type=int      , help="no known classes can't be greater than total classes")
	parser.add_argument("--no_open"             , default=1                   , type=int      , help="no unknown classes can't be greater than (total classes-no closed classes)")
	parser.add_argument("--no_total"            , default=11                  , type=int      , help="no total classes")
	parser.add_argument("--tail_size"           , default=20                  , type=int      , help="tail size for evt model (not in use now)")
	parser.add_argument("--default"             , default=True                , type=str2bool , help="change True if validation set required")
	parser.add_argument("--tanh_flag"           , default=True                , type=str2bool , help="change True if used tanh as final decoder layer")
	parser.add_argument("--separate_flag"       , default=True                , type=str2bool , help="change True if used separate is true")
	
	args = parser.parse_args()

	# change the arguments from default to optional
	sys.path.append('../parameters/'+args.dataset_name+'/')
	from parameters import hyper_para
	
	if not args.default:
		hyper_para.dataset_path				= args.dataset_path
		hyper_para.dataset_name				= args.dataset_name
		hyper_para.model_mode				= args.model_mode
		hyper_para.method 					= args.method
		hyper_para.dataset_file_format		= args.dataset_file_format
		hyper_para.gpu						= args.gpu
		hyper_para.verbose					= args.verbose
		hyper_para.default					= args.default
		hyper_para.tanh_flag				= args.tanh_flag
		hyper_para.lr 						= args.lr
		hyper_para.iterations				= args.iterations
		hyper_para.mlosr_model  			= args.mlosr_model
		hyper_para.separate_flag  			= args.separate_flag

		
	if hyper_para.model_mode == 'train':
		if hyper_para.method == 'mlosr':
			MLOSR(hyper_para)
		elif hyper_para.method == 'mlosr_ablation':
			MLOSR_ablation(hyper_para)
		else:
			print('ERROR: method type not valid!!')
	elif hyper_para.model_mode == 'eval':
		if hyper_para.method   == 'mlosr':
			MLOSR_test(hyper_para)
		elif hyper_para.method   == 'mlosr_ablation':
			MLOSR_test_ablation(hyper_para)
		else:
			print('ERROR: method type not valid!!')
	else:
		print('ERROR: Input valid model_mode argument!! valid model_mode are eval or train')

if __name__ == "__main__":
	main()