import numpy as np;
import ignite
import pickle
import torch
import scipy
import subprocess;
import os;
# import importlib
# import importlib.util
import collections

def get_image_name(subject, interval_ind, interval, view, frame, data_dir_path):
    frame_id = '_'.join([subject[:2], '%02d'%interval_ind,
                             str(view), '%06d'%frame])
    return os.path.join(data_dir_path,'{}/{}/{}/{}.jpg'.format(subject,
                                                        interval,
                                                        view,
                                                        frame_id))

def getFilesInFolder(folder,ext):
    list_files=[os.path.join(folder,file_curr) for file_curr in os.listdir(folder) if file_curr.endswith(ext)];
    return list_files;     
    
def getStartingFiles(dir_curr,img_name):
    files=[file_curr for file_curr in os.listdir(dir_curr) if file_curr.startswith(img_name)];
    return files;

def getEndingFiles(dir_curr,img_name):
    files=[file_curr for file_curr in os.listdir(dir_curr) if file_curr.endswith(img_name)];
    return files;


def getFileNames(file_paths,ext=True):
    just_files=[file_curr[file_curr.rindex('/')+1:] for file_curr in file_paths];
    file_names=[];
    if ext:
        file_names=just_files;
    else:
        file_names=[file_curr[:file_curr.rindex('.')] for file_curr in just_files];
    return file_names;

def getRelPath(file_curr,replace_str='/disk2'):
    count=file_curr.count('/');
    str_replace='../'*count
    rel_str=file_curr.replace(replace_str,str_replace);
    return rel_str;

def mkdir(dir_curr):
    if not os.path.exists(dir_curr):
        os.mkdir(dir_curr);

def makedirs(dir_curr):
    if not os.path.exists(dir_curr):
        os.makedirs(dir_curr);


def getIndexingArray(big_array,small_array):
    small_array=np.array(small_array);
    big_array=np.array(big_array);
    assert np.all(np.in1d(small_array,big_array))

    big_sort_idx= np.argsort(big_array)
    small_sort_idx= np.searchsorted(big_array[big_sort_idx],small_array)
    index_arr = big_sort_idx[small_sort_idx]
    return index_arr

def getIdxRange(num_files,batch_size):
    idx_range=range(0,num_files+1,batch_size);
    if idx_range[-1]!=num_files:
        idx_range.append(num_files);
    return idx_range;

def readLinesFromFile(file_name):
    with open(file_name,'r') as f:
        lines=f.readlines();
    lines=[line.strip('\n') for line in lines];
    return lines

def normalize(matrix,gpuFlag=False):
    if gpuFlag==True:
        import cudarray as ca
        norm=ca.sqrt(ca.sum(ca.power(matrix,2),1,keepdims=True));
        matrix_n=matrix/norm
    else:
        norm=np.sqrt(np.sum(np.square(matrix),1,keepdims=True));
        matrix_n=matrix/norm
    
    return matrix_n

def getHammingDistance(indices,indices_hash):
    ham_dist_all=np.zeros((indices_hash.shape[0],));
    for row in range(indices_hash.shape[0]):
        ham_dist_all[row]=scipy.spatial.distance.hamming(indices[row],indices_hash[row])
    return ham_dist_all    

def product(arr):
    p=1;
    for l in arr:
        p *= l
    return p;

def getIOU(box_1,box_2):
    box_1=np.array(box_1);
    box_2=np.array(box_2);
    minx_t=min(box_1[0],box_2[0]);
    miny_t=min(box_1[1],box_2[1]);
    min_vals=np.array([minx_t,miny_t,minx_t,miny_t]);
    box_1=box_1-min_vals;
    box_2=box_2-min_vals;
    # print box_1,box_2
    maxx_t=max(box_1[2],box_2[2]);
    maxy_t=max(box_1[3],box_2[3]);
    img=np.zeros(shape=(maxx_t,maxy_t));
    img[box_1[0]:box_1[2],box_1[1]:box_1[3]]=1;
    img[box_2[0]:box_2[2],box_2[1]:box_2[3]]=img[box_2[0]:box_2[2],box_2[1]:box_2[3]]+1;
    # print np.min(img),np.max(img)
    count_union=np.sum(img>0);
    count_int=np.sum(img==2);
    # print count_union,count_int
    # plt.figure();
    # plt.imshow(img,vmin=0,vmax=10);
    # plt.show();
    iou=count_int/float(count_union);
    return iou

def escapeString(string):
    special_chars='!"&\'()*,:;<=>?@[]`{|}';
    for special_char in special_chars:
        string=string.replace(special_char,'\\'+special_char);
    return string

def replaceSpecialChar(string,replace_with):
    special_chars='!"&\'()*,:;<=>?@[]`{|}';
    for special_char in special_chars:
        string=string.replace(special_char,replace_with);
    return string

def writeFile(file_name,list_to_write):
    with open(file_name,'w') as f:
        for string in list_to_write:
            f.write(string+'\n');

def getAllSubDirectories(meta_dir):
    meta_dir=escapeString(meta_dir);
    command='find '+meta_dir+' -type d';
    sub_dirs=subprocess.check_output(command,shell=True)
    sub_dirs=sub_dirs.split('\n');
    sub_dirs=[dir_curr for dir_curr in sub_dirs if dir_curr];
    return sub_dirs

def get_class_weights_au(train_files, n_classes = None):
    
    if n_classes is None:
        idx_start = 1
    else:
        idx_start = -1*n_classes

    arr = []
    for line_curr in train_files:
        arr.append([int(val) for val in line_curr.split(' ')[idx_start:]] )
    arr = np.array(arr)
    
    arr[arr>0]=1
    counts = np.sum(arr,0)
    counts = counts/float(np.sum(counts))
    counts = np.array([1./val  if val>0 else 0 for val in counts])
    counts = counts/float(np.sum(counts))  
        
    return counts

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_class_weights(train_files,au=False):
    classes = [int(line_curr.split(' ')[1]) for line_curr in train_files]
    classes_uni = np.unique(classes)
    # print classes_uni
    counts = np.array([classes.count(val) for val in classes_uni])
    # print counts
    counts = counts/float(np.sum(counts))
    counts = 1./counts
    counts = counts/float(np.sum(counts))
    counts_class = counts
    to_return = []
    to_return.append(counts_class)

    if au:
        arr = []
        for line_curr in train_files:
            arr.append([int(val) for val in line_curr.split(' ')[2:]] )
        arr = np.array(arr)
        # print arr.shape
        arr_keep = arr[arr[:,0]>0,1:]
        # print arr_keep.shape
        counts = np.sum(arr_keep,0)
        # print counts
        counts = counts/float(np.sum(counts))
        counts = 1./counts
        counts = counts/float(np.sum(counts))    
        # print counts
        # print counts.shape
        to_return.append(counts)
    
    if len(to_return)>1:
        return tuple(to_return)
    else:
        return to_return[0]


def save_training_error(save_path, engine, vis, vis_windows):
    # log training error
    iteration = engine.state.iteration - 1
    loss, _ = engine.state.output
    print("Epoch[{}] Iteration[{}] Batch Loss: {:.2f}".format(engine.state.epoch, iteration, loss))
    title="Training error"
    if vis is not None:
        vis_windows[title] = vis.line(X=np.array([engine.state.iteration]), Y=np.array([loss]),
                 update='append' if title in vis_windows else None,
                 win=vis_windows.get(title, None),
                 opts=dict(xlabel="# iteration", ylabel="loss", title=title))
    # also save as .txt for plotting
    log_name = os.path.join(save_path, 'debug_log_training.txt')
    if iteration ==0:
        with open(log_name, 'w') as the_file: # overwrite exiting file
            the_file.write('#iteration,loss\n')     
    with open(log_name, 'a') as the_file:
        the_file.write('{},{}\n'.format(iteration, loss))


def save_testing_error(save_path, trainer, evaluator, vis, vis_windows,
                       dataset_str, save_extension=None):
    # The trainer is only given here to get the current iteration and epoch.
    iteration = trainer.state.iteration
    epoch = trainer.state.epoch

    metrics = evaluator.state.metrics
    import ipdb; ipdb.set_trace()
    print("{} Results - Epoch: {}  AccumulatedLoss: {}".format(dataset_str, epoch, metrics))
    metric_values = []
    for key in metrics.keys():
        title="Testing metric: {}".format(key)
        metric_value = metrics[key]
        metric_values.append(metric_value)
        if vis is not None:
            vis_windows[title] = vis.line(X=np.array([iteration]), Y=np.array([metric_value]),
                         update='append' if title in vis_windows else None,
                         win=vis_windows.get(title, None),
                         opts=dict(xlabel="# iteration", ylabel="value", title=title))

    # also save as .txt for plotting
    log_name = os.path.join(save_path, save_extension)
    if iteration ==0:
        with open(log_name, 'w') as the_file: # overwrite exiting file
            the_file.write('#iteration,loss1,loss2,...\n')     
    with open(log_name, 'a') as the_file:
        the_file.write('{},{}\n'.format(iteration, ",".join(map(str, metric_values)) ))
    return sum(metric_values)

        
def save_model_state(save_path, engine, current_loss, model, optimizer):
    # update the best value
    best_val = engine.state.metrics.get('best_val', 99999999)
    engine.state.metrics['best_val'] = np.minimum(current_loss, best_val)
    
    print("Saving last model")
    model_path = os.path.join(save_path,"models/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path,"network_last_val.pth"))
    torch.save(optimizer.state_dict(), os.path.join(model_path,"optimizer_last_val.pth"))
    import ipdb; ipdb.set_trace()
    state_variables = {key:value for key, value in engine.state.__dict__.items() if key in ['iteration','metrics']}
    pickle.dump(state_variables, open(os.path.join(model_path,"state_last_val.pickle"),'wb'))
    
    if current_loss==engine.state.metrics['best_val']:
        print("Saving best model (previous best_loss={} > current_loss={})".format(best_val, current_loss))
        
        torch.save(model.state_dict(), os.path.join(model_path,"network_best_val_t1.pth"))
        torch.save(optimizer.state_dict(), os.path.join(model_path,"optimizer_best_val_t1.pth"))
        state_variables = {key:value for key, value in engine.state.__dict__.items() if key in ['iteration','metrics']}
        pickle.dump(state_variables, open(os.path.join(model_path,"state_best_val_t1.pickle"),'wb'))


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
        print("accloss is being updated")
        y_pred, y = output
        average_loss = self._loss_fn(y_pred, y)
        assert len(average_loss.shape) == 0, '`loss_fn` did not return the average loss'
        self._sum += average_loss.item() * 1 # HELGE: Changed here from original version
        self._num_examples += 1 # count in number of batches

    def compute(self):
        if self._num_examples == 0:
            raise ignite.exceptions.NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples
    
    
