'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
import os
from copy import deepcopy
import numpy

from pyseqlab.utilities import TemplateGenerator, DataFileParser, create_directory
from pyseqlab.attributes_extraction import SequenceStruct, NERSegmentAttributeExtractor
from pyseqlab.features_extraction import FOFeatureExtractor, HOFeatureExtractor, SeqsRepresenter, FeatureFilter
from pyseqlab.fo_crf import FirstOrderCRF, FirstOrderCRFModelRepresentation
from pyseqlab.ho_crf import HOCRF, HOCRFModelRepresentation
from pyseqlab.hosemi_crf import HOSemiCRF, HOSemiCRFModelRepresentation
from pyseqlab.ho_crf_ad import HOCRFADModelRepresentation, HOCRFAD
from pyseqlab.hosemi_crf_ad import HOSemiCRFADModelRepresentation, HOSemiCRFAD
from pyseqlab.crf_learning import Learner



root_dir = os.path.dirname(os.path.realpath(__file__))

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
        
        attr_extractor = NERSegmentAttributeExtractor()
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
            raw_diff, rel_diff = crf_model.validate_forward_backward_pass(numpy.ones(len(crf_model.weights)), seq_id)
            lines += "raw_diff {}\n".format(raw_diff)
            lines += "rel_diff {}\n".format(rel_diff)
            lines += "#"*40 + "\n"
        print(lines)

    def test_grad_computation(self):
        crf_model = self._crf_model
        seqs_id = self._seqs_id
        for seq_id in seqs_id:
            avg_diff = crf_model.check_gradient(numpy.ones(len(crf_model.weights)), seq_id)
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
        learner.train_model(numpy.zeros(len(crf_model.weights)), seqs_id, optimization_options, working_dir)
        if(optimization_options["method"] not in {"COLLINS-PERCEPTRON", "SAPO"}):
            crf_model.seqs_info = seqs_info
            avg_fexp_diff = crf_model.validate_expected_featuresum(crf_model.weights, seqs_id)
            lines += "avg_fexp_diff {}\n".format(avg_fexp_diff)
            lines += "#"*40 + "\n"
            print(lines)
                    
    def test_feature_extraction(self):

        seqs_id = self._seqs_id
        seqs_info = self._seqs_info
        model = self._model
        crf_model = self._crf_model
        print(crf_model.seqs_info == seqs_info)
        globalfeatures_len = len(model.modelfeatures_codebook)
        activefeatures_len = 0
        f = {}
        for seq_id in seqs_id:
#             print(seqs_info[seq_id])
#             print(seqs_info[seq_id] == crf_model.seqs_info[seq_id])
            crf_model.load_activefeatures(seq_id)
#             print("crf.seqs_info ", crf_model.seqs_info[seq_id])
#             print("seqs_info ", seqs_info[seq_id])
            seq_activefeatures = crf_model.seqs_info[seq_id]["activefeatures"]
            for features_dict in seq_activefeatures.values():
                for z_patt in features_dict:
                    for windx in features_dict[z_patt]:
                        f[windx] = 1
            crf_model.clear_cached_info([seq_id])
#             print(seqs_info[seq_id])
        activefeatures_len += len(f)
                    
        statement = ""
        if(activefeatures_len < globalfeatures_len): 
            statement = "len(activefeatures) < len(modelfeatures)"
        elif(activefeatures_len > globalfeatures_len):
            statement = "len(activefeatures) > len(modelfeatures)"
        else:
            statement = "pass"
        print(statement)


def read_data(file_path, header, sep=" "):
    parser = DataFileParser()
    parser.read_file(file_path, header = header, column_sep = sep)
    seqs = parser.seqs
    return(seqs)

def load_segments():
    seqs = []
    X = [{'w':'I'}, {'w':'live'}, {'w':'in'}, {'w':'New'}, {'w':'Haven'}]
    Y = ['P', 'O', 'O', 'L', 'L']
    seqs.append(SequenceStruct(X, Y, "O"))
    X = [{'w':'Connecticut'}, {'w':'is'}, {'w':'in'}, {'w':'the'}, {'w':'United'},{'w':'States'}, {'w':'of'}, {'w':'America'}]
    Y = ['L', 'O', 'O', 'O', 'L', 'L', 'L', 'L']
    seqs.append(SequenceStruct(X, Y, "O"))
    return(seqs)

def run_segments():
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for attr_name = w
    template_generator.generate_template_XY('w', ('1-gram', range(0, 1)), '1-gram:2-gram', templateXY)
    templateY = {'Y':()}
    filter_obj = None
    seq = load_segments()
    return(seq, templateY, templateXY, filter_obj)

def load_suppl_example():
    X = [{'w':'Peter'}, {'w':'goes'}, {'w':'to'}, {'w':'Britain'}, {'w':'and'}, {'w':'France'}, {'w':'annually'},{'w':'.'}]
    Y = ['P', 'O', 'O', 'L', 'O', 'L', 'O', 'O']
    seq = SequenceStruct(X, Y)
    return([seq])

def run_suppl_example():
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for attr_name = w
    template_generator.generate_template_XY('w', ('1-gram', range(0,1)), '1-gram', templateXY)
    templateY = template_generator.generate_template_Y('3-gram')
    filter_info = {"filter_type":"pattern", "filter_val": {'P','O', 'L', 'L|O|L'}, "filter_relation": "not in"}
    filter_obj = FeatureFilter(filter_info)
    seq = load_suppl_example()
    return(seq, templateY, templateXY, filter_obj)

def run_conll00_seqs():
    data_file_path = os.path.join(root_dir, "dataset", "conll00", "train.txt")
    seqs = read_data(data_file_path, header = "main")
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for attr_name = w
    template_generator.generate_template_XY('w', ('1-gram', range(0, 1)), '1-gram:2-gram:3-gram', templateXY)
    templateY = {'Y':()}
    filter_obj = None
    return(seqs[:10], templateY, templateXY, filter_obj)

def test_crfs(model_type, scaling_method, optimization_options, run_config, test_type):
    if(model_type == "HOSemi"):
        crf_model = HOSemiCRF 
        model_repr = HOSemiCRFModelRepresentation
        fextractor = HOFeatureExtractor
    elif(model_type == "HOSemi_AD"):
        crf_model = HOSemiCRFAD 
        model_repr = HOSemiCRFADModelRepresentation
        fextractor = HOFeatureExtractor
    elif(model_type == "HO"):
        crf_model = HOCRF 
        model_repr = HOCRFModelRepresentation
        fextractor = HOFeatureExtractor
    elif(model_type == "HO_AD"):
        crf_model = HOCRFAD
        model_repr = HOCRFADModelRepresentation
        fextractor = HOFeatureExtractor
    elif(model_type == "FO"):
        crf_model = FirstOrderCRF 
        model_repr = FirstOrderCRFModelRepresentation
        fextractor = FOFeatureExtractor
    
    seqs, f_y, f_xy, filter_obj = run_config()
    crf_tester = TestCRFModel(f_y, f_xy, crf_model, model_repr, fextractor, scaling_method, optimization_options, filter_obj)
    crf_tester.test_workflow(seqs)
    
    if(test_type == 'forward backward'):
        # test forward backward computation
        crf_tester.test_forward_backward_computation()
    elif(test_type == "gradient"):
        # test gradient computation
        crf_tester.test_grad_computation()
    elif(test_type == "model learning"):
        # test model learning
        crf_tester.test_model_validity()
    elif(test_type == "feature extraction"):
        crf_tester.test_feature_extraction()
            
    crf_tester._crf_model.seqs_info.clear()

    return(crf_tester._crf_model)

def profile_test(model_type, scaling_method, optimization_options, run_config, test_type):
    import cProfile
    local_def = {'model_type':model_type,
                 'scaling_method':scaling_method, 
                 'optimization_options':optimization_options,
                 'run_config':run_config,
                 'test_type':test_type
                }
    global_def = {'test_crfs':test_crfs}
    profiling_dir = create_directory('profiling', root_dir)
    cProfile.runctx('test_crfs(model_type, scaling_method, optimization_options, run_config, test_type)',
                    global_def, local_def, filename = os.path.join(profiling_dir, "profile_out"))
    

if __name__ == "__main__":
    pass