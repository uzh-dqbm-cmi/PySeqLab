'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
import os
import pickle
import shutil
from datetime import datetime
from copy import deepcopy
from itertools import combinations
import warnings
import numpy

warnings.filterwarnings('error')

class SequenceStruct():
    def __init__(self, X, Y, seg_other_symbol = None):
        """class for representing each sequence in the data
           Y: list containing the sequence of states (i.e. ['P','O','O','L','L'])
           X: list containing dictionary elements of observation sequences and/or features of the input
              (i.e. [{'w':Michael', 'shape':'Xx+'}, 
                     {'w':'is', 'shape':'x+'}, 
                     {'w':':'in', 'shape':'x+'}, 
                     {'w':'New', 'shape':'Xx+'},
                     {'w':'Haven', 'shape':'Xx+'}]
               where w is the word as input and shape is the collapsed shape of the word (Upper or lower case))
           seg_other_symbol: If it is specified, then the task is a segmentation problem 
                              (in this case we need to specify the non-entity/other element)
                              else if it is None (default), then it is considered as sequence labeling problem
 
        """
        self.seg_attr = {}
        self.X = X
        self.Y = (Y, seg_other_symbol)

    @property
    def X(self):
        return(self._X)
    @X.setter
    def X(self, l):
        """input is a list of elements (i.e. X =  [{'w':'Michael'}, {'w':'is'}, {'w':'in'}, {'w':'New'}, {'w':'Haven'}])
           output is a dict  {1:{'w':'Michael'},
                             2:{'w':'is'}, 
                             3:{'w':'in'}, 
                             4:{'w':'New'},
                             5:{'w':'Haven'}
                            }
        """
        self._X = {}
        for i in range(len(l)):
            self._X[i+1] = l[i]

        # new assignment clear seg_attr
        if(self.seg_attr):
            self.seg_attr.clear()
        self.T = len(self._X)
        
    @property
    def Y(self):
        return(self._Y)
    @Y.setter
    def Y(self, elmtup):
        """input is a tuple consisting of :
                Y: a list of elements (i.e. Y = ['P','O','O','L','L']) that represents the labels of the elements in X
                non_entity_symbol: is the label which represents the Other category (i.e. non entity element which is 'O' in above example)
           output is {(1, 1): 'P', (2,2): 'O', (3, 3): 'O', (4, 5): 'L'}
        """
        try:
            Y_ref, non_entity_symb = elmtup
        except ValueError:
            raise ValueError("tuple containing Y and non-entity symbol must be passed")
        else:
            self._Y = {}
            # length of longest entity in a segment
            L = 1
            if(non_entity_symb):
                label_indices = {}
                for i in range(len(Y_ref)):
                    label = Y_ref[i]
                    if(label in label_indices):
                        label_indices[label].append(i+1)
                    else:
                        label_indices[label] = [i+1]
 
                for label, indices_list in label_indices.items():
                    if(label == non_entity_symb or len(indices_list) == 1):
                        for indx in indices_list:
                            boundary = (indx, indx)
                            self._Y[boundary] = label
                    else:
                        indx_stack = []
                        for indx in indices_list:
                            if(not indx_stack):
                                indx_stack.append(indx)
                            else:
                                diff = indx - indx_stack[-1]
                                if(diff > 1):
                                    boundary = (indx_stack[0], indx_stack[-1])
                                    self._Y[boundary] = label
                                    l = indx_stack[-1] - indx_stack[0]
                                    if(l > L):
                                        L = l
                                    indx_stack = [indx]
                                else:
                                    indx_stack.append(indx)
                        if(indx_stack):
                            boundary = (indx_stack[0], indx_stack[-1])
                            self._Y[boundary] = label
                            l = indx_stack[-1] - indx_stack[0]
                            if(l > L):
                                L = l
                            indx_stack = [indx]
    
            else:
                for i in range(len(Y_ref)):
                    label = Y_ref[i]
                    boundary = (i+1, i+1)
                    self._Y[boundary] = label
            # store the length of longest entity
            self.L = L
            # keep a copy of Y in as flat list (i.e. ['P','O','O','L','L'])
            self.flat_y = Y_ref
            
    def update_boundaries(self):
        self.y_boundaries = self.get_y_boundaries()
        self.x_boundaries = self.get_x_boundaries()

    def flatten_y(self, Y):
        """ input Y is {(1, 1): 'P', (2,2): 'O', (3, 3): 'O', (4, 5): 'L'}
            output is  ['P','O','O','L','L']
        """
        s_boundaries = sorted(Y)
        flat_y = []
        for b in s_boundaries:
            for _ in range(b[0], b[-1]+1):
                flat_y.append(Y[b])
        return(flat_y)
  
    def get_y_boundaries(self):
        return(sorted(self.Y.keys()))
    
    def get_x_boundaries(self):
        boundaries = []
        for elmkey in self.X:
            boundaries.append((elmkey, elmkey))
        return(boundaries)
    
class DataFileParser():
    """ class to parse a data file that includes the training data consisting of:
        label sequences Y
        Observation sequences X (each type of observation sequences is represented in a separate column)
        IMPORTANT: label sequences are the LAST column in the file (i.e. X_a X_b Y)
    """
    def __init__(self):
        # list that will hold the sequences, each is an instance of SequenceStruct() class
        self.seqs = []
        self.header = []
        
    def read_file(self, file_path, header, y_ref = True, seg_other_symbol = None, column_sep = " "):
        """ header: specifying how the header is reported in the file containing the sequences
                       'main' -> one header in the beginning of the file
                       'per_sequence' -> a header for every sequence
                       list of keywords as header (i.e. ['w', 'part_of_speech'])
        """
        if(y_ref):
            update_seq = self.update_XY
        else:
            update_seq = self.update_X
            
        with open(file_path) as file_obj:
            counter = 0
            X = []
            Y = []
            for line in file_obj:
                counter += 1
                line = line.rstrip("\n")
#                 print(line)
                if line:
#                     print(line)
                    if(y_ref):
                        *x_arg, y = line.split(column_sep)
                        self._xarg = x_arg
                        self._y = y
                    else:
                        x_arg = line.split(column_sep)
                        self._xarg = x_arg

#                     print(x_arg)
                    # first line of a sequence
                    if(counter == 1):
                        if(header == "main"):
                            if(self.header):
                                update_seq(X, Y)
#                                 X.append(self.parse_line(x_arg))
#                                 Y.append(y)
                            else:
                                self.parse_header(x_arg)
                                
                        elif(header == "per_sequence"):
                            if(not self.header):
                                self.parse_header(x_arg)
                        else:
                            if(self.header):
                                update_seq(X, Y)
#                                 X.append(self.parse_line(x_arg))
#                                 Y.append(y)
                            else:   
                                self.parse_header(header)
                                update_seq(X, Y)
#                                 X.append(self.parse_line(x_arg))
#                                 Y.append(y)
                    else:
                        update_seq(X, Y)
#                         X.append(self.parse_line(x_arg))
#                         Y.append(y)

                else:
                    seq = SequenceStruct(X, Y, seg_other_symbol)
                    self.seqs.append(seq)
                    # reset counter for filling new sequence
                    counter = 0
                    X = []
                    Y = []
                    self._xarg = None
                    self._y = None
                    
            if(X and Y):
                seq = SequenceStruct(X, Y, seg_other_symbol)
                self.seqs.append(seq)
                # reset counter for filling new sequence
                counter = 0
                X = []
                Y = []
                self._xarg = None
                self._y = None
        

    def update_XY(self, X, Y):
        X.append(self.parse_line(self._xarg))
        Y.append(self._y)
    
    def update_X(self, X, Y):
        X.append(self.parse_line(self._xarg))
                
    def parse_line(self, x_arg):
        # fill the sequences X and Y with observations and tags respectively
        header = self.header
        x = {}
        for i in range(len(x_arg)):
            x[header[i]] = x_arg[i]
        return(x)

    def parse_header(self, x_arg):
        seq_header = [input_src for input_src in x_arg]
        self.header = seq_header

    def print_seqs(self):
        seqs = self.seqs
        for i in range(len(seqs)):
            seq = seqs[i]
            print("Y sequence: \n {}".format(seq.Y))
            print("X sequence: \n {}".format(seq.X))
            print("-" * 40)
            
    def clear_seqs(self):
        self.seqs = []

class ReaderWriter(object):
    def __init__(self):
        pass
    @staticmethod
    def dump_data(data, file_name, mode = "wb"):
        with open(file_name, mode) as f:
            pickle.dump(data, f, protocol = 4) 
    @staticmethod  
    def read_data(file_name, mode = "rb"):
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)
    @staticmethod
    def log_progress(line, outfile, mode="a"):
        with open(outfile, mode) as f:
            f.write(line)



##########################
# A* searcher to be used with viterbi algorithm to generate k-decoded list
########################
import heapq

class AStarNode(object):
    def __init__(self, cost, position, pi_c, label, frwdlink):
        self.cost = cost
        self.position = position
        self.pi_c = pi_c
        self.label = label
        self.frwdlink = frwdlink
        
    def print_node(self):
        statement = "cost: {}, position: {}, label: {}, ".format(self.cost, self.position, self.label)
        if(self.frwdlink):
            statement += "forward_link: {}".format(self.frwdlink)
        else:
            statement += "forward_link: None"
        print(statement)
        
class AStarAgenda(object):
    def __init__(self):
        self.qagenda = []
        self.entry_count = 0

    def push(self, astar_node, cost):
        heapq.heappush(self.qagenda, (-cost, self.entry_count, astar_node))
        self.entry_count += 1

    def pop(self):
        astar_node = heapq.heappop(self.qagenda)[-1]
        return(astar_node)
    
class AStarSearcher(object):
    def __init__(self, P_codebook, pi_elems):
        self.P_codebook = P_codebook
        self.P_codebook_rev = {code:state for state, code in P_codebook.items()}
        self.pi_elems = pi_elems
        
    def get_node_label(self, pi_code):
        pi = self.P_codebook_rev[pi_code]
        y =  self.pi_elems[pi][-1]
        return(y)

    def infer_labels(self, top_node, back_track):
        P_codebook = self.P_codebook
        P_codebook_rev = self.P_codebook_rev
        # decoding the sequence
        #print("we are decoding")
        #top_node.print_node()
        y = top_node.label
        pi_c = top_node.pi_c
        pos = top_node.position
        Y_decoded = []
        Y_decoded.append((P_codebook_rev[pi_c], y))
        #print("t={}, p_T_code={}, p_T={}, y_T ={}".format(T, p_T_code, p_T, y_T))
        t = pos - 1
        while t>0:
            p_tplus1 = Y_decoded[-1][0]
            p_t, y_t = back_track[(t+1, P_codebook[p_tplus1])]
            #print("t={}, (t+1, p_t_code)=({}, {})->({},{})".format(t, t+1, P_codebook[p_tplus1], p_t, y_t))
            Y_decoded.append((p_t, y_t))
            t -= 1
        Y_decoded.reverse()
        
        while(top_node.frwdlink):
            pi_c = top_node.frwdlink.pi_c
            y = top_node.frwdlink.label
            Y_decoded.append((P_codebook_rev[pi_c], y))
            top_node = top_node.frwdlink
#         print(Y_decoded)
        return([y for (pi, y) in Y_decoded])
    
    def search(self, alpha, back_track, T, K):
        # push the best astar nodes to the queue (i.e. the pi's at time T)
        q = AStarAgenda()
        r = set()
        c = 0
        P_codebook_rev = self.P_codebook_rev
        P_codebook = self.P_codebook
        # create nodes from the pi's at time T
        for pi_c in P_codebook_rev:
            cost = alpha[T, pi_c]
            pos = T
            frwdlink = None
            label = self.get_node_label(pi_c)
            node = AStarNode(cost, pos, pi_c, label, frwdlink)
#             node.print_node()
            q.push(node, cost)
            
        track = []
        topk_list = []
        try:
            while c < K:
                #print("heap size ", len(q.qagenda))
                top_node = q.pop()
                track.append(top_node)
        
#                 for n in q.qagenda:
#                     print(n)
            
                for i in reversed(range(2, top_node.position+1)):
                    prev_pi, y = back_track[i, top_node.pi_c]
                    prev_pi_c = P_codebook[prev_pi]
                    pos = i - 1
                    for curr_pi_c in P_codebook_rev:
                        # create a new astar node
                        if(curr_pi_c != prev_pi_c):
                            label = self.get_node_label(curr_pi_c)
                            cost = alpha[pos, curr_pi_c]
                            s = AStarNode(cost, pos, curr_pi_c, label, top_node)
                            q.push(s, cost)
                    # create the backlink of the top_node 
                    cost = alpha[pos, prev_pi_c]
                    top_node = AStarNode(cost, pos, prev_pi_c, y, top_node)
                    
                # decode and check if it is not saved already in topk list
                y_labels = self.infer_labels(track[-1], back_track)
#                 print(y_labels)
                sig = "".join(y_labels)
                if(sig not in r):
                    r.add(sig)
                    topk_list.append(y_labels)
                    c += 1
                    track.pop()
        except (KeyError, IndexError) as e:
            # consider logging the error
            print(e)
    
        finally:
            #print('r ', r)
            #print('topk ', topk_list)
            return(topk_list)

#######################
# template generating utility functions
#######################
class TemplateGenerator(object):
    def __init__(self):
        pass
    
    def generate_template_XY(self, attr_name, x_spec, y_spec, template):
        ngram_options, wsize = x_spec
        templateX = self._traverse_x(attr_name, ngram_options, wsize)
        templateY = self.generate_template_Y(y_spec)
        templateXY = self._mix_template_XY(templateX, templateY)
        #update the template we are building
        for attr_name in templateXY:
            if(attr_name in template):
                for x_offset in templateXY[attr_name]:
                    template[attr_name][x_offset] = templateXY[attr_name][x_offset]
            else:
                template[attr_name] = templateXY[attr_name]
                        
    def _traverse_x(self, attr_name, ngram_options, wsize):
        options = ngram_options.split(":")
        l = list(wsize)
        template = {attr_name:{}}
        for option in options:
            n = int(option.split("-")[0])
            ngram_list = self.generate_ngram(l, n)
            for offset in ngram_list:
                template[attr_name][offset] = None
        return(template)
    
    def generate_template_Y(self, ngram_options):
        template = {'Y':[]}
        options = ngram_options.split(":")
        for option in options:
            max_order = int(option.split("-")[0])
            template['Y'] += self._traverse_y(max_order, accumulative = False)['Y']
        return(template)
    
    @staticmethod
    def _traverse_y(max_order, accumulative = True):
        attr_name = 'Y'
        template = {attr_name:[]}
        if(accumulative):
            for j in range(max_order):
                offsets_y = [-i for i in range(j+1)]
                offsets_y = tuple(reversed(offsets_y))
                template[attr_name].append(offsets_y)
        else:
            offsets_y = [-i for i in range(max_order)]
            offsets_y = tuple(reversed(offsets_y))
            template[attr_name].append(offsets_y) 
    
        return(template)
    
    @staticmethod
    def _mix_template_XY(templateX, templateY):
        template_XY = deepcopy(templateX)
        for attr_name in template_XY:
            for offset_x in template_XY[attr_name]:
                template_XY[attr_name][offset_x] = tuple(templateY['Y'])
        return(template_XY)
    @staticmethod
    def generate_ngram(l, n):
        ngram_list = []
        for i in range(0, len(l)):
            elem = tuple(l[i:i+n])
            if(len(elem) != n):
                break
            ngram_list.append(elem)
            
        return(ngram_list)
    
    def generate_combinations(self):
        pass
    
def filter_templates(ngram_template, condition, operator):
    f_ngram_template = {}
    if(operator == "="):
        f_ngram_template = {condition:ngram_template[condition]}
    elif(operator == "!="):
        f_ngram_template = deepcopy(ngram_template)
        del(f_ngram_template[condition])         
    elif(operator == "in"):
        f_ngram_template = deepcopy(ngram_template)
        for option in ngram_template:
            if(condition not in option):
                del(f_ngram_template[option])
    return(f_ngram_template)

def filter_templates_by_attrname(f_ngram_template, attr_name):
    filtered_template = {attr_name:{}}
    for option, template in f_ngram_template.items():
        if(attr_name in template):
            filtered_template[attr_name].update(template[attr_name])
    if(not filtered_template[attr_name]):
        del filtered_template[attr_name]
    return(filtered_template)

def generate_templates(attr_names, window, y_options, x_options):

    ngram_templateY = ngram_options_y(y_options)
    ngram_templateX = ngram_options_x(attr_names, window, x_options)
    ngram_templateXY = ngram_options_xy(ngram_templateX, ngram_templateY)
    return((ngram_templateY, ngram_templateXY))

def ngram_combinations(n,accumulative = True, comb = True):
    option_names = []
    if(accumulative):
        start = 1
    else:
        start = n
    for i in range(start, n+1):
        option_names.append("{}-gram".format(i))
        
    config = {}
    if(comb):
        start = 1
    else:
        start = n
    for i in range(start, n+1):
        config[i] = list(combinations(option_names, i))
        
    config_combinations = {}
    for c_list in config.values():
        for c_tup in c_list:
            key_name = "_".join(c_tup)
            option= {}
            for elem in c_tup:
                option.update({elem: True})
            config_combinations[key_name] = option
            
    return(config_combinations)

def ngram_options_y(options):
    # get the y options 
    n = options['n']
    cummulative = options['cummulative']
    comb = options['comb']
    config_combinations = ngram_combinations(n, cummulative, comb)
    ngram_templateY = {}
    for comb, options in config_combinations.items():
        templateY = {'Y':[]}
        for option in options:
            max_order = int(option.split("-")[0])
            templateY['Y'] += generate_templateY(max_order, accumulative = False)['Y']
        ngram_templateY[comb] = templateY
    return(ngram_templateY)
        
def ngram_options_x(attr_names, l, options):
    # get the x options 
    n = options['n']
    cummulative = options['cummulative']
    comb = options['comb']
    if(len(l) < n):
        # setting up the maximum order based on the provided window
        n = len(l)
    config_combinations = ngram_combinations(n, cummulative, comb)
    ngram_templateX = {}
    for comb, options in config_combinations.items():
        print("comb {}".format(comb))
        ngram_templateX[comb] = generate_templateX(attr_names, l, options)     
        print("ngram_templateX[comb] = {}".format(ngram_templateX[comb]))   
    return(ngram_templateX)

def ngram_options_xy(ngram_templateX, ngram_templateY):
    ngram_templateXY = {}
    for option_y, templateY in ngram_templateY.items():
        for option_x, templateX in ngram_templateX.items():
            ngram_templateXY["{}:{}".format(option_x, option_y)] = mix_templateXY(templateX, templateY)
    return(ngram_templateXY)

def generate_templateX(attr_names, l, options):
    template = {}
    for attr_name in attr_names:
        template[attr_name] = {}
        ngram_list = []
        for option in options:
            n = int(option.split("-")[0])
            ngram_list = generate_ngram(l, n)
            for offset in ngram_list:
                template[attr_name][offset] = None
    return(template)

def generate_templateY(max_order, accumulative = True):
    attr_name = 'Y'
    template = {attr_name:[]}
    temp = []
    if(accumulative):
        for j in range(max_order):
            offsets_y = [-i for i in range(j+1)]
            offsets_y = tuple(reversed(offsets_y))
            temp.append(offsets_y)
    else:
        offsets_y = [-i for i in range(max_order)]
        offsets_y = tuple(reversed(offsets_y))
        temp.append(offsets_y) 
         
    template[attr_name] = temp     
    return(template)

def mix_templateXY(templateX, templateY):
    template_X = deepcopy(templateX)
    for attr_name in template_X:
        for offset_x in template_X[attr_name]:
            template_X[attr_name][offset_x] = tuple(templateY['Y'])
    return(template_X)
    
def generate_ngram(l, n):
    ngram_list = []
    for i in range(0, len(l)):
        elem = tuple(l[i:i+n])
        if(len(elem) != n):
            break
        ngram_list.append(elem)
        
    return(ngram_list)


def delete_directory(directory):
    if(os.path.isdir(directory)):
        shutil.rmtree(directory)
        
def delete_file(filepath):
    check = os.path.isfile(filepath)
    if(check):
        os.remove(filepath)
                
def create_directory(folder_name, directory = "current"):
    """ function to create directory/folder if it does not exist and returns the path of the directory"""
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)
    else:
        path_current_dir = directory
    path_new_dir = os.path.join(path_current_dir, folder_name)
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)

def generate_datetime_str():
    datetime_now = datetime.now()
    datetime_str = "{}_{}_{}-{}_{}_{}_{}".format(datetime_now.year,
                                                 datetime_now.month,
                                                 datetime_now.day,
                                                 datetime_now.hour,
                                                 datetime_now.minute,
                                                 datetime_now.second,
                                                 datetime_now.microsecond)
    return(datetime_str)



def vectorized_logsumexp(vec):
    with numpy.errstate(invalid='warn'):
        max_a = numpy.max(vec)
        try:
            res = max_a + numpy.log(numpy.sum(numpy.exp(vec - max_a)))
        except Warning:
            res = max_a
    return(res)
    
##################
# utility functions for splitting, grouping dataset
#################
def split_data(seqs_id, options):
    N = len(seqs_id)
    data_split = {}
    method = options.get('method')
    if(method == None):
        method = 'cross_validation'
    if(method == "cross_validation"):
        k_fold = options.get("k_fold")
        if(type(k_fold) != int):
            # use 10 fold cross validation
            k_fold = 10
        elif(k_fold <= 0):
            k_fold = 10
        batch_size = int(numpy.ceil(N/k_fold))
        test_seqs = seqs_id.copy()
        seqs_len = len(test_seqs)
        numpy.random.shuffle(test_seqs)
        indx = numpy.arange(0, seqs_len + 1, batch_size)
        if(indx[-1] < seqs_len):
            indx = numpy.append(indx, [seqs_len])
            
        for i in range(len(indx)-1):
            data_split[i] = {}
            current_test_seqs = test_seqs[indx[i]:indx[i+1]]
            data_split[i]["test"] = current_test_seqs
            data_split[i]["train"] = list(set(seqs_id)-set(current_test_seqs))

    elif(method == "random"):
        num_splits = options.get("num_splits")
        if(type(num_splits) != int):
            num_splits = 5
        trainset_size = options.get("trainset_size")
        if(type(trainset_size) != int):
            # 80% of the data set is training and 20% for testing
            trainset_size = 80
        elif(trainset_size <= 0 or trainset_size >=100):
            trainset_size = 80
        for i in range(num_splits):
            data_split[i] = {}
            current_train_seqs = numpy.random.choice(seqs_id, int(N*trainset_size/100), replace = False)
            data_split[i]["train"] = list(current_train_seqs)
            data_split[i]["test"] = list(set(seqs_id)-set(current_train_seqs))
            
    return(data_split)

#########################################
## split data based on seqs length -
## we need to execute the three functions in order:
## (1) group_seqs_by_length, (2) weighted_sample, (3) aggregate_weightedsample
#########################################
def group_seqs_by_length(seqs_info):
    grouped_seqs = {}
    for seq_id, seq_info in seqs_info.items():
        T = seq_info["T"]
        if(T in grouped_seqs):
            grouped_seqs[T].append(seq_id)
        else:
            grouped_seqs[T] = [seq_id]
    # loop to regroup single sequences
    singelton = [T for T, seqs_id in grouped_seqs.items() if len(seqs_id) == 1]
    singelton_seqs = []
    for T in singelton:
        singelton_seqs += grouped_seqs[T]
        del grouped_seqs[T]

    grouped_seqs["singleton"] = singelton_seqs
    return(grouped_seqs)
    
def weighted_sample(grouped_seqs, trainset_size):
    options = {'method':'random', 'num_splits':1, 'trainset_size':trainset_size}
    wsample = {}
    for group_var, seqs_id in grouped_seqs.items():
#         quota = trainset_size*count_seqs[group_var]/total
        data_split = split_data(seqs_id, options)
        wsample[group_var] = data_split[0]
    return(wsample)

def aggregate_weightedsample(w_sample):
    wdata_split= {"train":[],
                  "test": []}
    for grouping_var in w_sample:
        for data_cat in w_sample[grouping_var]:
            wdata_split[data_cat] += w_sample[grouping_var][data_cat]
    return({0:wdata_split})
##################################

def nested_cv(seqs_id, outer_kfold, inner_kfold):
    outer_split = split_data(seqs_id, "cross_validation", k_fold = outer_kfold)
    cv_hierarchy = {}
    for outerfold, outer_datasplit in outer_split.items():
        cv_hierarchy["{}_{}".format("outer", outerfold)] = outer_datasplit
        curr_train_seqs = outer_datasplit['train']
        inner_split = split_data(curr_train_seqs, "cross_validation", k_fold = inner_kfold) 
        for innerfold, inner_datasplit in inner_split.items():
            cv_hierarchy["{}_{}_{}_{}".format("outer", outerfold, "inner", innerfold)] = inner_datasplit
    return(cv_hierarchy)

def get_conll00():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(current_dir)
    parser = DataFileParser()
    files_info = {'train_short_main.txt':('main', True, " "), 
                  'train_short_none.txt':(('w','pos'), True, " "),
                  'train_short_per_sequence.txt':('per_sequence', True, " "),
                  'train_short_noref.txt':(('w', 'pos'), False, "\t")
                  }
    for file_name in files_info:
        file_path = os.path.join(root_dir, "tests", "dataset","conll00",file_name)
        parser.read_file(file_path, header=files_info[file_name][0], y_ref = files_info[file_name][1], column_sep=files_info[file_name][2])
        parser.print_seqs()
        print("*"*40)
        parser.seqs = []
        parser.header = []

    
if __name__ == "__main__":
    pass
#     get_conll00()

