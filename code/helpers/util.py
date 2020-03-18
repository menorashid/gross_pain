import numpy as np;
import torch
import scipy
import subprocess;
import os;
import importlib
import importlib.util



################ Code adapted from Rhodin below ##########################################
def default_collate_with_string(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    _use_shared_memory = False
    numpy_type_map = {
        'float64': torch.DoubleTensor,
        'float32': torch.FloatTensor,
        'float16': torch.HalfTensor,
        'int64': torch.LongTensor,
        'int32': torch.IntTensor,
        'int16': torch.ShortTensor,
        'int8': torch.CharTensor,
        'uint8': torch.ByteTensor,
    }
    string_classes = (str, bytes)
    if torch.is_tensor(batch[0]):
        #print("IN","torch.is_tensor(batch[0])")
        #IPython.embed()
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        #print("batch:",[e.numpy().shape for e in batch])
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        #print("IN", "type(batch[0]).__module__ == 'numpy'")
        #IPython.embed()
        if type(elem).__name__ == 'ndarray':
            if elem.dtype.kind in {'U', 'S'}:
                return np.stack(batch, 0)
            else:
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate_with_string([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate_with_string(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


def load_module(path):
    # if contained in module it would be a oneliner: 
    # config_dict_module = importlib.import_module(dict_module_name) 
    module_child_name = path.split('/')[-1].replace('.py','')
    spec = importlib.util.spec_from_file_location(module_child_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

###############################################################################

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
