'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
import os
import pickle
import shutil
from datetime import datetime
from copy import deepcopy
from itertools import combinations
import numpy
from attributes_extraction import SequenceStruct


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
        
    def read_file(self, file_path, header, seg_other_symbol = None, column_sep = " "):
        """ header: specifying how the header is reported in the file containing the sequences
                       'main' -> one header in the beginning of the file
                       'per_sequence' -> a header for every sequence
                       list of keywords as header (i.e. ['w', 'part_of_speech'])
        """
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
                    *x_arg, y = line.split(column_sep)
                    # first line of a sequence
                    if(counter == 1):
                        if(header == "main"):
                            if(self.header):
                                X.append(self.parse_line(x_arg))
                                Y.append(y)
                            else:
                                self.parse_header(x_arg)
                        elif(header == "per_sequence"):
                            if(self.header):
                                continue
                            else:
                                self.parse_header(x_arg)
                                X.append(self.parse_line(x_arg))
                                Y.append(y)
                        else:
                            self.parse_header(header)
                            X.append(self.parse_line(x_arg))
                            Y.append(y)   
                    else:
                        X.append(self.parse_line(x_arg))
                        Y.append(y)

                else:
                    seq = SequenceStruct(X, Y, seg_other_symbol)
                    self.seqs.append(seq)
                    # reset counter for filling new sequence
                    counter = 0
                    X = []
                    Y = []
            if(X and Y):
                seq = SequenceStruct(X, Y, seg_other_symbol)
                self.seqs.append(seq)
                # reset counter for filling new sequence
                counter = 0
                X = []
                Y = []
            
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
            
#######################
# template generating utility functions
#######################

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
                    
def generate_templates(attr_names, window, n_y, n_x):
    ngram_templateY = ngram_options_y(n_y)
    ngram_templateX = ngram_options_x(attr_names, window, n_x)
    ngram_templateXY = ngram_options_xy(ngram_templateX, ngram_templateY)
    return((ngram_templateY, ngram_templateXY))

def ngram_combinations(n):
    option_names = []
    for i in range(1, n+1):
        option_names.append("{}-gram".format(i))
        
    config = {}
    for i in range(1, n+1):
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

def ngram_options_y(n):
    config_combinations = ngram_combinations(n)
    ngram_templateY = {}
    for comb, options in config_combinations.items():
        templateY = {'Y':[]}
        for option in options:
            max_order = int(option.split("-")[0])
            templateY['Y'] += generate_templateY(max_order, accumulative = False)['Y']
        ngram_templateY[comb] = templateY
    return(ngram_templateY)
        
def ngram_options_x(attr_names, l, n):
    if(len(l) < n):
        # setting up the maximum order based on the provided window
        n = len(l)
    config_combinations = ngram_combinations(n)
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
    max_a = numpy.max(vec)
    res = max_a + numpy.log(numpy.sum(numpy.exp(vec - max_a)))
    return(res)
    
##################
# utility functions for splitting, grouping dataset
#################
def split_data(seqs_id, method, **kwargs):
    N = len(seqs_id)
    data_split = {}

    if(method == "cross_validation"):
        k_fold = kwargs.get("k_fold")
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
#             print("t = {} -- evaluating y_t = {}\n".format(t, y_tminus1))
    elif(method == "random"):
        num_splits = kwargs.get("num_splits")
        if(type(num_splits) != int):
            num_splits = 5
        trainset_size = kwargs.get("trainset_size")
        if(type(trainset_size) != int):
            # 80% of the data set is training and 20% for testing
            trainset_size = 80
        elif(trainset_size <= 0 or trainset_size >=100):
            trainset_size = 80
        for i in range(num_splits):
            data_split[i] = {}
            current_train_seqs = numpy.random.choice(seqs_id, N*trainset_size/100, replace = False)
            data_split[i]["train"] = list(current_train_seqs)
            data_split[i]["test"] = list(set(seqs_id)-set(current_train_seqs))
            
    return(data_split)

def group_seqs_by_length(seqs_info):
    grouped_seqs = {}
    for seq_id, seq_info in seqs_info.items():
        T = seq_info["T"]
        if(T in grouped_seqs):
            grouped_seqs[T].append(seq_id)
        else:
            grouped_seqs[T] = [seq_id]
    return(grouped_seqs)

    
def weighted_sample(grouped_seqs, trainset_size):
#     count_seqs = {}
#     total = 0
#     for group_var, seqs_id in grouped_seqs.items():
#         num_seqs = len(seqs_id)
#         count_seqs[group_var] = num_seqs
#         total += num_seqs
    wsample = {}
    for group_var, seqs_id in grouped_seqs.items():
#         quota = trainset_size*count_seqs[group_var]/total
        data_split = split_data(seqs_id, method = "random", num_splits = 1, trainset_size = trainset_size)
        wsample[group_var] = data_split
    return(wsample)

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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "dataset","conll00","train_short.txt")
    parser = DataFileParser()
    parser.read_file(file_path, header="main")
    parser.print_seqs()
    
if __name__ == "__main__":
    get_conll00()
