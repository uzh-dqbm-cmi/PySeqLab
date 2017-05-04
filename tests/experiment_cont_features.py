'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''    
import os
from copy import deepcopy
from pyseqlab.features_extraction import FOFeatureExtractor, HOFeatureExtractor, SeqsRepresenter
from pyseqlab.ho_crf_ad import HOCRFAD, HOCRFADModelRepresentation
from pyseqlab.fo_crf import FirstOrderCRF, FirstOrderCRFModelRepresentation
from pyseqlab.workflow import TrainingWorkflow
from pyseqlab.utilities import ReaderWriter, SequenceStruct, TemplateGenerator, create_directory, generate_updated_model
from pyseqlab.attributes_extraction import AttributeScaler
from pyseqlab.crf_learning import Learner
import numpy as np

SEQ_LEN = 100
NUM_LABELS = 3
NUM_FEATURES = 100
NUM_SEQS = 5
SCALING = "rescaling"
PERC_CATEGORICAL=0
# NUM_CONT_FEATURES = int(np.ceil(NUM_FEATURES*(100-PERC_CATEGORICAL)/100))
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))


def generate_data(seq_len, num_labels, num_features, percent_categorical=PERC_CATEGORICAL):
    len_perlabel = int(np.ceil(seq_len/num_labels))
    num_cont_features = num_features
    if(percent_categorical):
        num_cont_features = int(np.ceil(num_features*(100-percent_categorical)/100))
    step_size = 50
    labels = []
    for j in range(num_labels):
        labels += [j]*len_perlabel
    # generate continuous features
    features = []
    for i in range(num_cont_features):
        step_min = np.random.randint(1, 100, 1)
        step_max = step_min + step_size
        feature = np.array([])
        for j in range(num_labels):
            feature = np.concatenate([feature,
                                      np.random.randint(step_min,step_max, len_perlabel) + np.random.randn(len_perlabel)])
            step_min = step_max + step_size + np.random.randint(1, step_size, 1)
            step_max = step_min + step_size
        features.append(feature.tolist())
    # generate categorical features (if applicable)
    for i in range(num_features-num_cont_features):
        step_min = np.random.randint(1, 100, 1)
        step_max = step_min + 10
        feature = np.array([], dtype='int32')
        for j in range(num_labels):
            feature = np.concatenate([feature, np.random.randint(step_min,step_max, len_perlabel)])
            step_min = step_max + step_size + np.random.randint(1, step_size, 1)
            step_max = step_min + 10
        features.append(feature.tolist())
    return(features, labels)

def prepare_data(features, labels):
    num_features = len(features)
    X = []
    flag = False
#     print("features ", features)
    for i in range(num_features):
        feat_name = 'f_{}'.format(i)
#         print("feat_name ", feat_name)
        for j, elem in enumerate(features[i]):
#             print("j ", j)
#             print("elem ", elem)
            if(flag):
                X[j][feat_name] = str(elem)
            else:
                X.append({feat_name:str(elem)})
        flag = True
    labels = [str(elem) for elem in labels]
    return(X, labels)

def generate_seqs(seq_len, num_labels, num_features, num_seqs, percent_categorical=PERC_CATEGORICAL):
    seqs = []
    for __ in range(num_seqs):
        features, labels= generate_data(seq_len, num_labels, num_features)
        X, Y = prepare_data(features, labels)
        seq = SequenceStruct(X, Y)
        seqs.append(seq)
    return(seqs)

class AttributeExtractor(object):
    """class implementing observation functions that generates attributes from observations"""

    def __init__(self):
        self.num_cont_features = int(np.ceil(NUM_FEATURES*(100-PERC_CATEGORICAL)/100))
        self.track_names = ["f_{}".format(i) for i in range(NUM_FEATURES)]
        self.attr_desc = self.generate_attributes_desc()
        self.seg_attr = {}
    
    def generate_attributes_desc(self):
        attr_desc = {}
        for i in range(self.num_cont_features):
            track_attr_name = "f_{}".format(i)
            attr_desc[track_attr_name] = {'description': '{} track'.format(track_attr_name),
                                          'encoding':'continuous'}
        for j in range(self.num_cont_features, NUM_FEATURES):
            track_attr_name = "f_{}".format(j)
            attr_desc[track_attr_name] = {'description': '{} track'.format(track_attr_name),
                                          'encoding':'categorical'}  
        for attr in attr_desc:
            if(attr_desc[attr]['encoding'] == 'categorical'):
                attr_desc[attr]['repr_func'] = self._represent_categorical_attr
            else:
                attr_desc[attr]['repr_func'] = self._represent_continuous_attr
        print(attr_desc)
        return(attr_desc)
    
    def group_attributes(self):
        """function to group attributes based on the encoding type (i.e. continuous vs. categorical)"""
        attr_desc = self.attr_desc
        grouped_attr = {}
        for attr_name in attr_desc:
            encoding_type = attr_desc[attr_name]['encoding']
            if(encoding_type in grouped_attr):
                grouped_attr[encoding_type].append(attr_name)
            else:
                grouped_attr[encoding_type] = [attr_name]
        return(grouped_attr)
            
    def generate_attributes(self, seq, boundaries):
        X = seq.X  
        #observed_attrnames = self.generate_track_attrnames(('mean','std'))
        observed_attrnames = list(X[1].keys())
        # segment attributes dictionary
        self.seg_attr = {}
        new_boundaries = []
        # create segments from observations using the provided boundaries
        for boundary in boundaries:
            if(boundary not in seq.seg_attr):
                self._create_segment(X, boundary, observed_attrnames)
                new_boundaries.append(boundary)
#         print("seg_attr {}".format(self.seg_attr))
#         print("new_boundaries {}".format(new_boundaries))
        if(self.seg_attr):
            # save generated attributes in seq
            seq.seg_attr.update(self.seg_attr)
#             print('saved attribute {}'.format(seq.seg_attr))
            # clear the instance variable seg_attr
            self.seg_attr = {}
        return(new_boundaries)
        
    def _create_segment(self, X, boundary, attr_names, sep = " "):
        self.seg_attr[boundary] = {}
        attr_desc = self.attr_desc
        for attr_name in attr_names:
            segment_value = self._get_segment_value(X, boundary, attr_name)
            self.seg_attr[boundary][attr_name] = attr_desc[attr_name]['repr_func'](segment_value, sep)
            
    def _get_segment_value(self, X, boundary, target_attr):
        u = boundary[0]
        v = boundary[1]
        segment = []
        for i in range(u, v+1):
            segment.append(X[i][target_attr])
        return(segment)
    
    def _represent_categorical_attr(self, attributes, sep):
        """function to represent categorical attributes
        """
        return(sep.join(attributes))

    def _represent_continuous_attr(self, attributes, sep=None):
        """function to represent continuous attributes
        """
        return(sum(float(attr) for attr in attributes))


def trainconfig_1():
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for tracks
    track_attr_names = ["f_{}".format(i) for i in range(NUM_FEATURES)]
    for track_attr_name in track_attr_names:
        template_generator.generate_template_XY(track_attr_name, ('1-gram:2-gram', range(-3,4)), '1-state:2-states', templateXY)
    templateY = {'Y':()}
    filter_obj = None
    ascaler_class = AttributeScaler
    return(templateXY, templateY, ascaler_class, filter_obj)

def trainconfig_2():
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for tracks
    track_attr_names = ["f_{}".format(i) for i in range(NUM_FEATURES)]
    for track_attr_name in track_attr_names:
        template_generator.generate_template_XY(track_attr_name, ('1-gram', range(0,1)), '1-state', templateXY)
    templateY = {'Y':()}
    filter_obj = None
    ascaler_class = AttributeScaler
    return(templateXY, templateY, ascaler_class, filter_obj)

def trainconfig_3():
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for tracks
    track_attr_names = ["f_{}".format(i) for i in range(NUM_FEATURES)]
    for track_attr_name in track_attr_names:
        template_generator.generate_template_XY(track_attr_name, ('1-gram:2-gram', range(-5,6)), '1-state', templateXY)
    templateY = {'Y':()}
    filter_obj = None
    ascaler_class = AttributeScaler
    return(templateXY, templateY, ascaler_class, filter_obj)
def run_lbfgs(model_type, trainconfig):
    optimization_options = {'method': "L-BFGS-B",
                            'regularization_type': 'l2',
                            'regularization_value': 0
                            }
    dsplit_options = {'method':"none"}
    return(train_crfs(model_type, SCALING, optimization_options, dsplit_options, trainconfig))

def run_perceptron(model_type, trainconfig):
    optimization_options = {"method" : "COLLINS-PERCEPTRON",
                            "num_epochs":15,
                            'update_type':'max-fast',
                            'beam_size':-1,
                            'shuffle_seq':False,
                            'avg_scheme':'avg_error',
                            "tolerance":1e-16
                            }
    dsplit_options = {'method':"none"}
    return(train_crfs(model_type, SCALING, optimization_options, dsplit_options, trainconfig))

def run_sga_classic(model_type, trainconfig):
    optimization_options = {"method" : "SGA",
                            "num_epochs":15,
                            "regularization_value":0,
                            "tolerance":1e-16
                            }
    dsplit_options = {'method':"none"}
    return(train_crfs(model_type, SCALING, optimization_options, dsplit_options, trainconfig))
def run_svrg(model_type, trainconfig):
    optimization_options = {"method" : "SVRG",
                            "num_epochs":10,
                            "regularization_value":0,
                            "tolerance":1e-16
                            }
    dsplit_options = {'method':"none"}
    return(train_crfs(model_type, SCALING, optimization_options, dsplit_options, trainconfig))
def run_sga_adadelta(model_type, trainconfig, profile=True):
    import cProfile
    optimization_options = {"method" : "SGA-ADADELTA",
                            "regularization_type": "l2",
                            "regularization_value":0,
                            "num_epochs":15,
                            "tolerance":1e-6
                            }
    dsplit_options = {'method':"none"}
    if(profile):
        local_def = {'model_type':model_type,
                     'scaling_method':SCALING,
                     'optimization_options':optimization_options,
                     'dsplit_options':dsplit_options,
                     'trainconfig':trainconfig
                    }
        global_def = {'train_crfs':train_crfs}
        profiling_dir = create_directory('profiling', root_dir)
        cProfile.runctx('train_crfs(model_type, scaling_method, optimization_options, dsplit_options, trainconfig)',
                        global_def, local_def, filename = os.path.join(profiling_dir, "profile_out"))
    else:
        return(train_crfs(model_type, SCALING, optimization_options, dsplit_options, trainconfig))

def revive_learnedmodel(modelparts_dir, model_type):

    modelrepr_class = HOCRFADModelRepresentation
    model_class = HOCRFAD
    fextractor_class = HOFeatureExtractor    
    aextractor_class = AttributeExtractor
    seqrepresenter_class = SeqsRepresenter
    lmodel = generate_updated_model(modelparts_dir, modelrepr_class,  model_class,
                                    aextractor_class, fextractor_class, seqrepresenter_class,
                                    ascaler_class=AttributeScaler)
    return(lmodel)
    
    
class TestCRFModel(object):
    def __init__(self, templateY, templateXY, model_class, model_repr_class, fextractor_class, scaling_method, optimization_options, filter_obj = None):
        self.template_Y = templateY
        self.template_XY = templateXY
        self.model_class = model_class
        self.model_repr_class = model_repr_class
        self.fextractor_class = fextractor_class
        self.scaling_method = scaling_method
        self.optimization_options = optimization_options
        self.root_dir = root_dir
        self.filter_obj = filter_obj
        
    def test_workflow(self, seqs):
        """ testing scenarios of mixing different templates
        """
        corpus_name = "reference_corpus"
        working_dir = create_directory("working_dir", self.root_dir)
        self._working_dir = working_dir
        unique_id = True
        seqs_dict = {}
        templateY = self.template_Y
        templateXY = self.template_XY
        modelrepr_class = self.model_repr_class
        model_class = self.model_class
        fextractor_class = self.fextractor_class
        scaling_method = self.scaling_method
        
        attr_extractor = AttributeExtractor()
        f_extractor = fextractor_class(templateXY, templateY, attr_extractor.attr_desc)
        seq_representer = SeqsRepresenter(attr_extractor, f_extractor)
        for i in range(len(seqs)):
            seqs_dict[i+1] = deepcopy(seqs[i-1])
        seqs_info = seq_representer.prepare_seqs(seqs_dict, corpus_name, working_dir, unique_id)
        seqs_id = list(seqs_info.keys())
        
        seq_representer.preprocess_attributes(seqs_id, seqs_info, method = scaling_method)
        seq_representer.extract_seqs_globalfeatures(seqs_id, seqs_info)
        model = seq_representer.create_model(seqs_id, seqs_info, modelrepr_class, self.filter_obj)
        seq_representer.extract_seqs_modelactivefeatures(seqs_id, seqs_info, model, "", learning = True)
        crf_model = model_class(model, seq_representer, seqs_info)
        
        self._seq_representer = seq_representer
        self._seqs_id = seqs_id
        self._seqs_info = seqs_info
        self._crf_model = crf_model
        self._model = model
                            
    def test_forward_backward_computation(self):
        crf_model = self._crf_model
        seqs_id = self._seqs_id
        for seq_id in seqs_id:
            lines = ""
            raw_diff, rel_diff = crf_model.validate_forward_backward_pass(np.ones(len(crf_model.weights)), seq_id)
            lines += "raw_diff {}\n".format(raw_diff)
            lines += "rel_diff {}\n".format(rel_diff)
            lines += "#"*40 + "\n"
        print(lines)

    def test_grad_computation(self):
        crf_model = self._crf_model
        seqs_id = self._seqs_id
        for seq_id in seqs_id:
            avg_diff = crf_model.check_gradient(np.ones(len(crf_model.weights)), seq_id)
            lines = ""
            lines += "avg_diff {} \n".format(avg_diff)
            lines += "#"*40 + "\n"
            print(lines)
    def test_grad_computation_2(self):
        crf_model = self._crf_model
        seqs_id = self._seqs_id
        for seq_id in seqs_id:
            avg_diff = crf_model.validate_gradient(np.ones(len(crf_model.weights)), seq_id)
            lines = ""
            lines += "avg_diff {} \n".format(avg_diff)
            lines += "#"*40 + "\n"
            print(lines)   
    def test_model_validity(self):
        """ testing scenarios of mixing different templates
        """
        optimization_options = self.optimization_options
        crf_model = self._crf_model
        seqs_id = self._seqs_id
        working_dir = self._working_dir
        seqs_info = self._seqs_info
        lines = ""

        learner = Learner(crf_model)
        learner.train_model(np.zeros(len(crf_model.weights)), seqs_id, optimization_options, working_dir)
        if(optimization_options["method"] not in {"COLLINS-PERCEPTRON", "SAPO"}):
            crf_model.seqs_info = seqs_info
            avg_fexp_diff = crf_model.validate_expected_featuresum(crf_model.weights, seqs_id)
            lines += "avg_fexp_diff {}\n".format(avg_fexp_diff)
            lines += "#"*40 + "\n"
            print(lines)
                    
    def test_feature_extraction(self):

        seqs_id = self._seqs_id
#         seqs_info = self._seqs_info
        model = self._model
        crf_model = self._crf_model
#         print(crf_model.seqs_info == seqs_info)
        globalfeatures_len = len(model.modelfeatures_codebook)
        activefeatures_len = 0
        f = set()
        for seq_id in seqs_id:
#             print(seqs_info[seq_id])
#             print(seqs_info[seq_id] == crf_model.seqs_info[seq_id])
            crf_model.load_activefeatures(seq_id)
#             print("crf.seqs_info ", crf_model.seqs_info[seq_id])
#             print("seqs_info ", seqs_info[seq_id])
            seq_activefeatures = crf_model.seqs_info[seq_id]["activefeatures"]
            for features_dict in seq_activefeatures.values():
                for z_patt in features_dict:
                    f.update(set(features_dict[z_patt][0]))
            crf_model.clear_cached_info([seq_id])
#             print(seqs_info[seq_id])
        activefeatures_len += len(f)
                    
        statement = ""
        if(activefeatures_len < globalfeatures_len): 
            statement = "len(activefeatures) < len(modelfeatures)"
        elif(activefeatures_len > globalfeatures_len):
            statement = "len(activefeatures) > len(modelfeatures)"
        else:
            statement = "PASS"
        print(statement)

  
def test_crfs(model_type, scaling_method, optimization_options, run_config_option, test_type):
    if(model_type == "HO_AD"):
        crf_model = HOCRFAD
        model_repr = HOCRFADModelRepresentation
        fextractor = HOFeatureExtractor
    elif(model_type == "FO"):
        crf_model = FirstOrderCRF 
        model_repr = FirstOrderCRFModelRepresentation
        fextractor = FOFeatureExtractor
    
    run_config, model_order, perc_categorical = run_config_option
    seqs, f_y, f_xy, filter_obj = run_config(model_order, perc_categorical)
    crf_tester = TestCRFModel(f_y, f_xy, crf_model, model_repr, fextractor, scaling_method, optimization_options, filter_obj)
    crf_tester.test_workflow(seqs)
    
    if(test_type == 'forward backward'):
        # test forward backward computation
        crf_tester.test_forward_backward_computation()
    elif(test_type == "gradient"):
        # test gradient computation
        crf_tester.test_grad_computation()
    elif(test_type == "gradient_2"):
        crf_tester.test_grad_computation_2()
    elif(test_type == "model learning"):
        # test model learning
        crf_tester.test_model_validity()
    elif(test_type == "feature extraction"):
        crf_tester.test_feature_extraction()
            
    crf_tester._crf_model.seqs_info.clear()

    return(crf_tester._crf_model)  
    
def run_config(model_order, perc_categorical):
    seqs = generate_seqs(SEQ_LEN, NUM_LABELS, NUM_FEATURES, NUM_SEQS, percent_categorical=perc_categorical)
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for tracks
    track_attr_names = ["f_{}".format(i) for i in range(NUM_FEATURES)]
    for track_attr_name in track_attr_names:
        template_generator.generate_template_XY(track_attr_name, ('1-gram:2-gram', range(0,1)), model_order, templateXY)
    templateY = {'Y':()}
    filter_obj = None
    return(seqs, templateY, templateXY, filter_obj)

        
def train_crfs(model_type, scaling_method, optimization_options, dsplit_options, trainconfig):
    crf_model = HOCRFAD
    model_repr = HOCRFADModelRepresentation
    fextractor = HOFeatureExtractor
        
    # generate data
    seqs = generate_seqs(SEQ_LEN, NUM_LABELS, NUM_FEATURES, NUM_SEQS)
    template_xy, template_y, ascaler_class, filter_obj = trainconfig()
    workflow_trainer = TrainingWorkflow(template_y, template_xy, model_repr, crf_model,
                                        fextractor, AttributeExtractor,scaling_method,
                                        optimization_options, root_dir, filter_obj)

    data_split = workflow_trainer.seq_parsing_workflow(seqs, dsplit_options)
    models_info = workflow_trainer.traineval_folds(data_split, meval=False)
#     print(models_info)
#     return(models_info)

if __name__ == "__main__":
    pass