'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
import os
from copy import deepcopy
import numpy

from pyseqlab.utilities import filter_templates, generate_templates, create_directory, TemplateGenerator, ReaderWriter, DataFileParser
from pyseqlab.attributes_extraction import SequenceStruct, NERSegmentAttributeExtractor
from pyseqlab.features_extraction import FOFeatureExtractor, HOFeatureExtractor, SeqsRepresentation
from pyseqlab.features_extraction import FeatureFilter
from pyseqlab.fo_crf_model import FirstOrderCRF, FirstOrderCRFModelRepresentation
from pyseqlab.ho_crf_model import HOCRF, HOCRFModelRepresentation
from pyseqlab.hosemi_crf_model import HOSemiCRF, HOSemiCRFModelRepresentation
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
        
    def test_workflow(self, seqs, target_function):
        """ testing scenarios of mixing different templates
        """
    #     numpy.random.seed(1)
        corpus_name = "reference_corpus"
        working_dir = create_directory("working_dir", self.root_dir)
        self._working_dir = working_dir
        unique_id = False
        seqs_dict = {}
        templateY = self.template_Y
        templateXY = self.template_XY
        modelrepr_class = self.model_repr_class
        model_class = self.model_class
        fextractor_class = self.fextractor_class
        scaling_method = self.scaling_method
        
        attr_extractor = NERSegmentAttributeExtractor()
        f_extractor = fextractor_class(templateXY, templateY, attr_extractor.attr_desc)
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
        self._seqs_id = seqs_id
        self._seqs_info = seqs_info
        self._crf_model = crf_model
        self._model = model
        target_function()
                            
    def test_forward_backward_computation(self):
        crf_model = self._crf_model
        seqs_id = self._seqs_id
        lines = ""
        raw_diff, rel_diff = crf_model.validate_forward_backward_pass(numpy.ones(len(crf_model.weights)), seqs_id[0])
        lines += "raw_diff {}\n".format(raw_diff)
        lines += "rel_diff {}\n".format(rel_diff)
        lines += "#"*40 + "\n"
        print(lines)

    def test_grad_computation(self):
        lines = ""
        crf_model = self._crf_model
        seqs_id = self._seqs_id
        avg_diff = crf_model.check_gradient(numpy.ones(len(crf_model.weights)), seqs_id[0])
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
        if(optimization_options["method"] != "COLLINS-PERCEPTRON"):
            crf_model.seqs_info = seqs_info
            avg_fexp_diff = crf_model.validate_expected_featuresum(crf_model.weights, seqs_id)
            lines += "avg_fexp_diff {}\n".format(avg_fexp_diff)
            lines += "#"*40 + "\n"
            print(lines)

    def test_feature_extraction(self):

        seqs_id = self._seqs_id
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
        print(statement)
        
    def find_wrong_templates(self, seqs):
        # identify if wrong templates exist while using y and f_xy as feature templates
        self.test_workflow(seqs, self.test_feature_extraction)
        
    def test_crf_forwardbackward(self, seqs):
        for i in range(len(seqs)):
            seq = seqs[i]
            self.test_workflow([seq], self.test_forward_backward_computation)
    
    def test_crf_grad(self, seqs):
        for i in range(len(seqs)):
            seq = seqs[i]
            self.test_workflow([seq], self.test_grad_computation)
    
    def test_crf_implementation(self, seqs):
        for i in range(len(seqs)):
            seq = seqs[i]
            self.test_workflow([seq], self.test_forward_backward_computation)
            self.test_workflow([seq], self.test_grad_computation)

    def test_crf_learning(self, seqs):
        self.test_workflow(seqs, self.test_model_validity)



def read_data(file_path, header):
    parser = DataFileParser()
    parser.read_file(file_path, header = header)
    seqs = parser.seqs
    return(seqs)
    
def load_predined_seq():
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
    seq = load_predined_seq()
    return(seq, templateY, templateXY, filter_obj)

def run_loaded_conll00_seqs():
    data_file_path = os.path.join(root_dir, "dataset/conll00/train.txt")
    seqs = read_data(data_file_path, header = "main")
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for attr_name = w
    template_generator.generate_template_XY('w', ('1-gram', range(-2, 3)), '1-gram:2-gram:3-gram', templateXY)
    templateY = template_generator.generate_template_Y('1-gram:2-gram:3-gram')
    filter_obj = None
    return(seqs[0:1], templateY, templateXY, filter_obj)
    
    
def test_crfs(model_type, scaling_method, optimization_options, run_config):
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
    
    seqs, f_y, f_xy, filter_obj = run_config()
    crf_tester = TestCRFModel(f_y, f_xy, crf_model, model_repr, fextractor, scaling_method, optimization_options, filter_obj)
#     crf_tester.find_wrong_templates(seqs)
    
#     crf_tester.test_crf_learning(seqs)
#     crf_tester.test_model_validity()
    crf_tester.test_crf_forwardbackward(seqs)
#     crf_tester.test_crf_grad(seqs[0:1])
    return(crf_tester._crf_model)

if __name__ == "__main__":
    pass