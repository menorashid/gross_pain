import numpy as np
import IPython
import pickle
import wandb
import torch
import sys
import os

from rhodin.python.utils import datasets as utils_data
from rhodin.python.utils import plot_dict_batch as utils_plot_batch
from rhodin.python.ignite.engine.engine import Engine, State, Events

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# optimization function
def create_supervised_trainer(model, optimizer, loss_fn, device=None):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = utils_data.nestedDictToDevice(batch, device=device) # make it work for dict input too
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred
    engine = Engine(_update)
    return engine

def create_supervised_evaluator(model, metrics={}, device=None):
    def _inference(engine, batch):  
        # now compute error
        model.eval()
        with torch.no_grad():
            x, y = utils_data.nestedDictToDevice(batch, device=device) # make it work for dict input too
            y_pred = model(x)
            
        return y_pred, y        

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def save_training_error(save_path, engine):
    # log training error
    iteration = engine.state.iteration - 1
    loss, _ = engine.state.output
    wandb.log({'train loss': loss})
    
    print("Epoch[{}] Iteration[{}] Batch Loss: {:.2f}".format(engine.state.epoch, iteration, loss))
    title="Training error"
    # also save as .txt for plotting
    log_name = os.path.join(save_path, 'debug_log_training.txt')
    if iteration ==0:
        with open(log_name, 'w') as the_file: # overwrite exiting file
            the_file.write('#iteration,loss\n')     
    with open(log_name, 'a') as the_file:
        the_file.write('{},{}\n'.format(iteration, loss))
    # if iteration<100:
    plot_loss(log_name, log_name.replace('.txt','.jpg'),'Training Loss')

 
def plot_loss(in_file, out_file, title):
    with open(in_file,'r') as f:
        lines=f.readlines()
    lines=[line.strip('\n') for line in lines]
    [xlabel,ylabel] = lines[0].split(',')[:2]
    x = []
    y = []
    for line in lines[1:]:
        line = line.split(',')
        x.append(int(line[0]))
        y.append(float(line[1]))

    plt.figure()
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.plot(np.array(x),np.array(y))
    plt.savefig(out_file)
    plt.close()
    


def save_testing_error(save_path, trainer, evaluator,
                       dataset_str, save_extension=None):
    # The trainer is only given here to get the current iteration and epoch.
    iteration = trainer.state.iteration
    epoch = trainer.state.epoch

    metrics = evaluator.state.metrics
    print("{} Results - Epoch: {}  AccumulatedLoss: {}".format(dataset_str, epoch, metrics))
    metric_values = []
    for key in metrics.keys():
        title="Testing metric: {}".format(key)
        metric_value = metrics[key]
        metric_values.append(metric_value)
        name_for_log = dataset_str + ' ' + key
        wandb.log({name_for_log: metric_value})

    # also save as .txt for plotting
    log_name = os.path.join(save_path, save_extension)
    if epoch == 1: #assumes you always eval at end of 1st epoch
        with open(log_name, 'w') as the_file: # overwrite exiting file
            the_file.write('#iteration,loss1,loss2,...\n')     
    with open(log_name, 'a') as the_file:
        the_file.write('{},{}\n'.format(iteration, ",".join(map(str, metric_values)) ))
    plot_loss(log_name, log_name.replace('.txt','.jpg'), 'Testing Loss')
    return metrics['AccumulatedLoss']


def save_training_example(save_path, engine, config_dict):
    # print training examples
    iteration = engine.state.iteration - 1
    loss, output = engine.state.output
    inputs, labels = engine.state.batch
    
    mode='training'
    img_name = os.path.join(save_path, 'debug_images_{}_{:06d}.jpg'.format(mode, iteration))
    utils_plot_batch.plot_iol(inputs, labels, output, config_dict, mode, img_name)
    #img = misc.imread(img_name)

    # log_name = os.path.join(save_path, 'debug_log_training.txt')
    # if os.path.exists(log_name):
    #     plot_loss(log_name, log_name.replace('.txt','.jpg'),'Training Loss')

def save_test_example(save_path, trainer, evaluator, config_dict):
    iteration_global = trainer.state.iteration
    iteration = evaluator.state.iteration - 1
    inputs, labels = evaluator.state.batch
    output, gt = evaluator.state.output # Note, comes in a different order as for training
    mode='testing_{}'.format(iteration_global)
    img_name = os.path.join(save_path, 'debug_images_{}_{:06d}.jpg'.format(mode,iteration))
    utils_plot_batch.plot_iol(inputs, labels, output, config_dict, mode, img_name)               
    
def load_model_state(save_path, model, optimizer, state):
    model.load_state_dict(torch.load(os.path.join(save_path,"network_best_val_t1.pth")))
    optimizer.load_state_dict(torch.load(os.path.join(save_path,"optimizer_best_val_t1.pth")))
    sate_variables = pickle.load(open(os.path.join(save_path,"state_last_best_val_t1.pickle"),'rb'))
    for key, value in sate_variables.items(): setattr(state, key, value)
    print('Loaded ',sate_variables)


def save_model_state(save_path, engine, current_loss, model, optimizer, state):
    # Update the best value, 99999999 if first time
    best_val = engine.state.metrics.get('best_val', 99999999)
    engine.state.metrics['best_val'] = np.minimum(current_loss, best_val)
    
    print("Saving last model")
    model_path = os.path.join(save_path,"models/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path,"network_last_val.pth"))
    torch.save(optimizer.state_dict(), os.path.join(model_path,"optimizer_last_val.pth"))
    state_variables = {key:value for key, value in engine.state.__dict__.items() if key in ['iteration','metrics']}
    pickle.dump(state_variables, open(os.path.join(model_path,"state_last_val.pickle"),'wb'))
    
    if current_loss==engine.state.metrics['best_val']:
        print("Saving best model (previous best_loss={} > current_loss={})".format(best_val, current_loss))
        
        torch.save(model.state_dict(), os.path.join(model_path,"network_best_val_t1.pth"))
        torch.save(optimizer.state_dict(), os.path.join(model_path,"optimizer_best_val_t1.pth"))
        state_variables = {key:value for key, value in engine.state.__dict__.items() if key in ['iteration','metrics']}
        pickle.dump(state_variables, open(os.path.join(model_path,"state_best_val_t1.pickle"),'wb'))


def save_model_state_iter(save_path, engine, model, optimizer, state):

    print("Saving model at epoch", state.epoch, "iter", state.iteration)
    model_path = os.path.join(save_path,"models/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    str_file_name = '%03d'%state.epoch
    torch.save(model.state_dict(), os.path.join(model_path,"network_"+str_file_name+".pth"))
    torch.save(optimizer.state_dict(), os.path.join(model_path,"optimizer_"+str_file_name+".pth"))
    state_variables = {key:value for key, value in engine.state.__dict__.items() if key in ['iteration','metrics']}
    pickle.dump(state_variables, open(os.path.join(model_path,"state_"+str_file_name+".pickle"),'wb'))
    

# Fix of original Ignite Loss to not depend on single tensor output but to accept dictionaries
from rhodin.python.ignite.metrics import Metric
class AccumulatedLoss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.
    `loss_fn` must return the average loss over all observations in the batch.
    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(AccumulatedLoss, self).__init__(output_transform)
        self._loss_fn = loss_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        average_loss = self._loss_fn(y_pred, y)
        assert len(average_loss.shape) == 0, '`loss_fn` did not return the average loss'
        self._sum += average_loss.item() * 1 # HELGE: Changed here from original version
        self._num_examples += 1 # count in number of batches

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples
    
    
def transfer_partial_weights(state_dict_other, obj, submodule=0, prefix=None, add_prefix=''):
    print('Transferring weights...')
    
    if 0:
        print('\nStates source\n')
        for name, param in state_dict_other.items():
            print(name)
        print('\nStates target\n')
        for name, param in obj.state_dict().items():
            print(name)
        
    own_state = obj.state_dict()
    copyCount = 0
    skipCount = 0
    paramCount = len(own_state)

    for name_raw, param in state_dict_other.items():
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if prefix is not None and not name_raw.startswith(prefix):
            #print("skipping {} because of prefix {}".format(name_raw, prefix))
            continue
        
        # remove the path of the submodule from which we load
        name = add_prefix+".".join(name_raw.split('.')[submodule:])

        if name in own_state:
            if hasattr(own_state[name],'copy_'): #isinstance(own_state[name], torch.Tensor):
                #print('copy_ ',name)
                if own_state[name].size() == param.size():
                    own_state[name].copy_(param)
                    copyCount += 1
                else:
                    print('Invalid param size(own={} vs. source={}), skipping {}'.format(own_state[name].size(), param.size(), name))
                    skipCount += 1
            
            elif hasattr(own_state[name],'copy'):
                own_state[name] = param.copy()
                copyCount += 1
            else:
                print('training.utils: Warning, unhandled element type for name={}, name_raw={}'.format(name,name_raw))
                print(type(own_state[name]))
                skipCount += 1
                IPython.embed()
        else:
            skipCount += 1
            print('Warning, no match for {}, ignoring'.format(name))
            #print(' since own_state.keys() = ',own_state.keys())
            
    print('Copied {} elements, {} skipped, and {} target params without source'.format(copyCount, skipCount, paramCount-copyCount))
