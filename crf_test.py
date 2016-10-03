'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
import os
from copy import deepcopy
import numpy
from utilities import filter_templates, generate_templates, create_directory, ReaderWriter, DataFileParser
from attributes_extraction import NERSegmentAttributeExtractor
from features_extraction import FOFeatureExtractor, HOFeatureExtractor, SeqsRepresentation
from crf_learning import Learner
from fo_crf_model import FirstOrderCRF, FirstOrderCRFModelRepresentation
from ho_crf_model import HOCRF, HOCRFModelRepresentation
from hosemi_crf_model import HOSemiCRF, HOSemiCRFModelRepresentation

root_dir = os.path.dirname(os.path.realpath(__file__))

class TestCRFModel(object):
    def __init__(self, ngram_y, ngram_xy, model_class, model_repr_class, fextractor_class, scaling_method, optimization_options, filter_obj = None):
        self.ngram_y = ngram_y
        self.ngram_xy = ngram_xy
        self.model_class = model_class
        self.model_repr_class = model_repr_class
        self.fextractor_class = fextractor_class
        self.scaling_method = scaling_method
        self.optimization_options = optimization_options
        self.root_dir = root_dir
        self.filter_obj = filter_obj
        
    def test_workflow(self, seqs, target_function):
        """ testing scenarios of mixing different templates
        """
    #     numpy.random.seed(1)
        corpus_name = "reference_corpus"
        working_dir = create_directory("working_dir", self.root_dir)
        self._working_dir = working_dir
        unique_id = False
        seqs_dict = {}
        res = {}
        lines = ""
        self._res = res
        self._lines = lines
        
        ngram_y = self.ngram_y
        ngram_xy = self.ngram_xy
        modelrepr_class = self.model_repr_class
        model_class = self.model_class
        fextractor_class = self.fextractor_class
        scaling_method = self.scaling_method
        
        for option_y, templateY in ngram_y.items():
            for option_x, templateX in ngram_xy.items():
                lines += "option for template Y : {} \n".format(option_y)
                lines += "option for templateXY - {}\n".format(option_x)
                attr_extractor = NERSegmentAttributeExtractor()
                f_extractor = fextractor_class(templateX, templateY, attr_extractor.attr_desc)
                seq_representer = SeqsRepresentation(attr_extractor, f_extractor)
                for i in range(len(seqs)):
                    seqs_dict[i+1] = deepcopy(seqs[i-1])
                seqs_info = seq_representer.prepare_seqs(seqs_dict, corpus_name, working_dir, unique_id)
                seqs_id = list(seqs_info.keys())
                
                seq_representer.preprocess_attributes(seqs_id, seqs_info, method = scaling_method)
                seq_representer.extract_seqs_globalfeatures(seqs_id, seqs_info)
                model = seq_representer.create_model(seqs_id, seqs_info, modelrepr_class, self.filter_obj)
                seq_representer.extract_seqs_modelactivefeatures(seqs_id, seqs_info, model, "")
                crf_model = model_class(model, seq_representer, seqs_info)
                
                self._seq_representer = seq_representer
                self._option_x = option_x
                self._option_y = option_y
                self._seqs_id = seqs_id
                self._seqs_info = seqs_info
                self._crf_model = crf_model
                self._model = model
                target_function()
                
        return(res)
            
    def test_forward_backward_computation(self):
        crf_model = self._crf_model
        seqs_id = self._seqs_id
        lines = self._lines
        working_dir = self._working_dir
        option_x = self._option_x
        option_y = self._option_y
        res = self._res
        raw_diff, rel_diff = crf_model.validate_forward_backward_pass(numpy.ones(len(crf_model.weights)), seqs_id[0])
        res[(option_y, option_x)] = rel_diff
        lines += "raw_diff {}\n".format(raw_diff)
        lines += "rel_diff {}\n".format(rel_diff)
        lines += "#"*40 + "\n"
        ReaderWriter.log_progress(lines, os.path.join(working_dir, "test_forward_backward_computation.txt"))
        lines = ""

    def test_grad_computation(self):
        crf_model = self._crf_model
        seqs_id = self._seqs_id
        lines = self._lines
        working_dir = self._working_dir
        option_x = self._option_x
        option_y = self._option_y
        res = self._res
        avg_diff = crf_model.check_gradient(numpy.ones(len(crf_model.weights)), seqs_id[0])
        res[(option_y, option_x)] = avg_diff
        lines += "avg_diff {} \n".format(avg_diff)
        lines += "#"*40 + "\n"
        ReaderWriter.log_progress(lines, os.path.join(working_dir, "test_grad_computation.txt"))
        lines = ""

    def test_model_validity(self):
        """ testing scenarios of mixing different templates
        """
        optimization_options = self.optimization_options
        crf_model = self._crf_model
        seqs_id = self._seqs_id
        lines = self._lines
        working_dir = self._working_dir
        option_x = self._option_x
        option_y = self._option_y
        res = self._res
        seqs_info = self._seqs_info
#         optimization_options = {"method" : "L-BFGS-B",
#                                 "regularization_type": "l2",
#                                 "regularization_value":0
#                                 }
#         optimization_options = {"method" : "COLLINS-PERCEPTRON",
#                                 "regularization_type": "l2",
#                                 "regularization_value":0,
#                                 "num_epochs":30
#                                 }
#         optimization_options = {"method" : "SGA-ADADELTA",
#                                 "regularization_type": "l2",
#                                 "regularization_value":0,
#                                 "num_epochs":100,
#                                 "tolerance":1e-6,
#                                 "p_rho":0.9
#                                 }
#         optimization_options = {"method" : "SVRG",
#                                 "regularization_type": "l2",
#                                 "regularization_value":0,
#                                 "num_epochs":20,
#                                 "tolerance":1e-6,
#                                 "learning_rate_schedule":"t_inverse",
#                                 "a":0.9
#                                 }
        learner = Learner(crf_model)
        learner.train_model(numpy.zeros(len(crf_model.weights)), seqs_id, optimization_options, working_dir)
        crf_model.seqs_info = seqs_info
        avg_fexp_diff = crf_model.validate_expected_featuresum(crf_model.weights, seqs_id)
        res[(option_y, option_x)] = avg_fexp_diff
        lines += "avg_fexp_diff {}\n".format(avg_fexp_diff)
        lines += "#"*40 + "\n"
        ReaderWriter.log_progress(lines, os.path.join(working_dir, "test_expected_featuresum_computation.txt"))
        lines = ""

    def test_feature_extraction(self):

        seqs_id = self._seqs_id
        option_x = self._option_x
        option_y = self._option_y
        res = self._res
        seqs_info = self._seqs_info
        seq_representer = self._seq_representer
        model = self._model
        
        globalfeatures_len = len(model.modelfeatures_codebook)
        
        seqs_activefeatures = seq_representer.get_seqs_modelactivefeatures(seqs_id, seqs_info)
        activefeatures_len = 0
        f = {}
        for seq_id, seq_activefeatures in seqs_activefeatures.items():
            for features_dict in seq_activefeatures.values():
                for z_patt in features_dict:
                    for windx in features_dict[z_patt]:
                        f[windx] = 1
        activefeatures_len += len(f)
                    
        statement = ""
        if(activefeatures_len < globalfeatures_len): 
            statement = "len(activefeatures) < len(modelfeatures)"
        elif(activefeatures_len > globalfeatures_len):
            statement = "len(activefeatures) > len(modelfeatures)"
        else:
            statement = "pass"
        res[(option_y, option_x)] = statement
        
    @staticmethod
    def illformed_templates(res):
        wrong_templates = []
        for template_tup, decision in res.items():
            print(template_tup)
            print(decision)
            if(decision != "pass"):
                wrong_templates.append(template_tup)
        return(wrong_templates)
        
    def find_wrong_templates(self, seqs):
        # identify if wrong templates exist while using y and f_xy as feature templates
        
        fe = self.test_workflow(seqs, self.test_feature_extraction)
        wrong_temp  = self.illformed_templates(fe)
        return(set(wrong_temp))
        
    def test_crf_forwardbackward(self, seqs):
        report = {}
        for i in range(len(seqs)):
            seq = seqs[i]
            fb = self.test_workflow([seq], self.test_forward_backward_computation)
            report[i] = {'forward-backward-check':fb}
        return(report)
    
    def test_crf_grad(self, seqs):
        report = {}
        for i in range(len(seqs)):
            seq = seqs[i]
            gc = self.test_workflow([seq], self.test_grad_computation)
            report[i] = {'gradient-check':gc}
        return(report)
    
    def test_crf_implementation(self, seqs):
        report = {}
        for i in range(len(seqs)):
            seq = seqs[i]
            fb = self.test_workflow([seq], self.test_forward_backward_computation)
            gc = self.test_workflow([seq], self.test_grad_computation)
            report[i] = {'forward-backward-check':fb, 'gradient-check':gc}
        return(report)
    
    def test_crf_learning(self, seqs):
        mv = self.test_workflow(seqs, self.test_model_validity)
        return(mv)



def read_data(file_path, header):
    parser = DataFileParser()
    parser.read_file(file_path, header = header)
    seqs = parser.seqs
    return(seqs)
    
def load_predined_seq():
    from attributes_extraction import SequenceStruct
    X = [{'w':'Peter'}, {'w':'goes'}, {'w':'to'}, {'w':'Britain'}, {'w':'and'}, {'w':'France'}, {'w':'annually'},{'w':'.'}]
    Y = ['P', 'O', 'O', 'L', 'O', 'L', 'O', 'O']
    seq = SequenceStruct(X, Y)
    return([seq])

def test_crfs(model_type, scaling_method, optimization_options):
#     attr_names = ('w', 'seg_numchars')
    attr_names = ('w', )

    window = list(range(-2,2))
#     window = list(range(-1,1))

    n_y = 3
    n_x = 3
# 
    data_file_path = os.path.join(root_dir, "dataset/conll00/train_short_main.txt")
    seqs = read_data(data_file_path, header = "main")
#     seqs = load_predined_seq()
    y, xy = generate_templates(attr_names, window, n_y, n_x)
    # filter templates to keep at least one unigram feature (it is a MUST)
#     f_y = filter_templates(y, '3-gram', "=")
    f_y = filter_templates(y, '1-gram_2-gram', "=")
#     f_xy = filter_templates(xy, '1-gram_2-gram_3-gram:1-gram_2-gram', "=")
    f_xy = filter_templates(xy, '1-gram:1-gram', "=")

    if(model_type == "HOSemi"):
        crf_model = HOSemiCRF 
        model_repr = HOSemiCRFModelRepresentation
        fextractor = HOFeatureExtractor
    elif(model_type == "HO"):
        crf_model = HOCRF 
        model_repr = HOCRFModelRepresentation
        fextractor = HOFeatureExtractor
    elif(model_type == "FO"):
        crf_model = FirstOrderCRF 
        model_repr = FirstOrderCRFModelRepresentation
        fextractor = FOFeatureExtractor
    from features_extraction import FeatureFilter
    
    # O|O|L was causing a problem ...
#     filter_info = {"filter_type":"pattern", "filter_val": ['P','O', 'L', 'L|O|L'], "filter_relation": "!="}
#     filter_obj = FeatureFilter(filter_info)
    filter_obj = None
    crf_tester = TestCRFModel(f_y, f_xy, crf_model, model_repr, fextractor, scaling_method, optimization_options, filter_obj)
    wrong_temp = crf_tester.find_wrong_templates(seqs[0:2])
    if(wrong_temp):
        raise("ill-formed template..")
    else:
        mv = crf_tester.test_crf_learning(seqs[0:2])
#         fb = crf_tester.test_crf_forwardbackward(seqs)
#         gc = crf_tester.test_crf_grad(seqs[0:1])
#         print("fb {}".format(fb))
#         print("gc {}".format(gc))
    return(crf_tester._crf_model)

if __name__ == "__main__":
    pass