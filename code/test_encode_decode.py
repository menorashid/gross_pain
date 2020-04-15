import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations
import numpy as np
import numpy.linalg as la

from rhodin.python.utils import io as rhodin_utils_io
from rhodin.python.utils import datasets as rhodin_utils_datasets
from rhodin.python.utils import plotting as rhodin_utils_plt
from rhodin.python.utils import skeleton as rhodin_utils_skel

import train_encode_decode
from rhodin.python.ignite._utils import convert_tensor
from rhodin.python.ignite.engine import Events

from matplotlib.widgets import Slider, Button

# load data
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class IgniteTestNVS(train_encode_decode.IgniteTrainNVS):
    def run(self, config_dict_file, config_dict):
        config_dict['n_hidden_to3Dpose'] = config_dict.get('n_hidden_to3Dpose', 2)


        if 0: # load small example data
            import pickle
            data_loader = pickle.load(open('../examples/test_set.pickl',"rb"))
        else:
            data_loader = self.load_data_test(config_dict)
            # save example data
            if 0:
                import pickle
                data_iterator = iter(data_loader)
                data_cach = [next(data_iterator) for i in range(10)]
                data_cach = tuple(data_cach)
                pickle.dump(data_cach, open('../examples/test_set.pickl', "wb"))

        # load model
        model = self.load_network(config_dict)
        model = model.to(device)

        def tensor_to_npimg(torch_array):
            return np.swapaxes(np.swapaxes(torch_array.numpy(), 0, 2), 0, 1)

        def denormalize(np_array):
            return np_array * np.array(config_dict['img_std']) + np.array(config_dict['img_mean'])

        # extract image
        def tensor_to_img(output_tensor):
            output_img = tensor_to_npimg(output_tensor)
            output_img = denormalize(output_img)
            output_img = np.clip(output_img, 0, 1)
            return output_img

        # get next image
        input_dict, label_dict = None, None
        data_iterator = iter(data_loader)
        def nextImage():
            nonlocal input_dict, label_dict
            input_dict, label_dict = next(data_iterator)

            input_dict['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().to(device)
            input_dict['img_crop']=input_dict['bg_crop']
        nextImage()

        print (list(input_dict.keys()))
        for k in label_dict.keys():
            print (k)
            # print (type(label_dict[k]))

        # apply model on images
        output_dict = None
        def predict():
            nonlocal output_dict
            model.eval()
            with torch.no_grad():
                input_dict_cuda, label_dict_cuda = rhodin_utils_datasets.nestedDictToDevice((input_dict, label_dict), device=device)
                output_dict_cuda = model(input_dict_cuda)
                output_dict = rhodin_utils_datasets.nestedDictToDevice(output_dict_cuda, device='cpu')
        predict()
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    args = train_encode_decode.parse_arguments(sys.argv[1:])
    print(args)
    train_subjects = re.split('/', args.train_subjects)
    test_subjects = re.split('/',args.test_subjects)
    config_dict_module = rhodin_utils_io.loadModule(args.config_file)
    config_dict = config_dict_module.config_dict
    config_dict['job_identifier'] = args.job_identifier
    config_dict['train_subjects'] = train_subjects
    config_dict['test_subjects'] = test_subjects
    ignite = IgniteTestNVS()
    ignite.run(config_dict_module.__file__, config_dict)
