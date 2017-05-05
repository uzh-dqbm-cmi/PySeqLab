'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
import os
from pyseqlab.features_extraction import SeqsRepresenter
from pyseqlab.crf_learning import Learner, Evaluator, SeqDecodingEvaluator
from pyseqlab.utilities import create_directory, generate_datetime_str, ReaderWriter, split_data, \
                               group_seqs_by_length, weighted_sample, aggregate_weightedsample, \
                               generate_updated_model


class TrainingWorkflow(object):
    """general training workflow
    
      .. note::
      
        It is **highly recommended** to start using :class:`TrainingWorkflowIterative` class instead
        of the :class:`TrainingWorkflow` class.
        
    """
    def __init__(self, template_y, template_xy, model_repr_class, model_class,
                 fextractor_class, aextractor_class, scaling_method, 
                 optimization_options, root_dir, filter_obj = None):
        self.template_y = template_y
        self.template_xy = template_xy
        self.model_class = model_class
        self.model_repr_class = model_repr_class
        self.fextractor_class = fextractor_class
        self.aextractor_class = aextractor_class
        self.scaling_method = scaling_method
        self.optimization_options = optimization_options
        self.root_dir = root_dir
        self.filter_obj = filter_obj

    def seq_parsing_workflow(self, seqs, split_options):
        """preparing sequences to be used in the learning framework"""
        
        # create working directory 
        corpus_name = "reference_corpus"
        working_dir = create_directory("working_dir", self.root_dir)
        self.working_dir = working_dir
        unique_id = True
        
        # create sequence dictionary mapping each sequence to a unique id
        seqs_dict = {i+1:seqs[i] for i in range(len(seqs))}
        seqs_id = list(seqs_dict.keys())
        self.seqs_id = seqs_id
        
        # initialize attribute extractor
        attr_extractor = self.aextractor_class()

        # create the feature extractor
        scaling_method = self.scaling_method        
        fextractor_class = self.fextractor_class
        f_extractor = fextractor_class(self.template_xy, self.template_y, attr_extractor.attr_desc)
        
        # create sequence representer
        seq_representer = SeqsRepresenter(attr_extractor, f_extractor)
        
        # get the seqs_info representing the information about the parsed sequences
        seqs_info = seq_representer.prepare_seqs(seqs_dict, corpus_name, working_dir, unique_id) 
        
        # preporcess and generate attributes in case of segments with length >1 or in case of scaling of
        # attributes is needed               
        seq_representer.preprocess_attributes(seqs_id, seqs_info, method = scaling_method)
        
        # extract global features F(X,Y)
        seq_representer.extract_seqs_globalfeatures(seqs_id, seqs_info)
        
        # save the link to seqs_info and seq_representer as instance variables
        self.seqs_info = seqs_info
        self.seq_representer = seq_representer
        
        # split dataset according to the specified split options
        data_split = self.split_dataset(seqs_info, split_options)
        
        # save the datasplit dictionary on disk 
        gfeatures_dir = seqs_info[1]['globalfeatures_dir']
        ref_corpusdir = os.path.dirname(os.path.dirname(gfeatures_dir))
        ReaderWriter.dump_data(data_split, os.path.join(ref_corpusdir, 'data_split'))
        return(data_split)

    def split_dataset(self, seqs_info, split_options):
        if(split_options['method'] == "wsample"):
            # try weighted sample
            # first group seqs based on length
            grouped_seqs = group_seqs_by_length(seqs_info)
            # second get a weighted sample based on seqs length
            w_sample = weighted_sample(grouped_seqs, trainset_size=split_options['trainset_size'])
            print("w_sample ", w_sample)
            # third aggregate the seqs in training category and testing category
            data_split = aggregate_weightedsample(w_sample)
        elif(split_options['method'] == "cross_validation"):
            # try cross validation
            seqs_id = list(seqs_info.keys())
            data_split = split_data(seqs_id, split_options)
        elif(split_options['method'] == 'random'):
            seqs_id = list(seqs_info.keys())
            data_split = split_data(seqs_id, split_options)
        elif(split_options['method'] == 'none'):
            seqs_id = list(seqs_info.keys())
            data_split = {0:{'train':seqs_id}}    
        return(data_split)

    def traineval_folds(self, data_split, meval=True, sep=" "):
        """train and evaluate model on different dataset splits"""
        
        seqs_id = self.seqs_id
        seq_representer = self.seq_representer
        seqs_info = self.seqs_info
        model_repr_class = self.model_repr_class
        model_class = self.model_class
        models_info = []
        ref_corpusdir = os.path.dirname(os.path.dirname(seqs_info[1]['globalfeatures_dir']))
        if(meval):
            traineval_fname = "modeleval_train.txt"
            testeval_fname = "modeleval_test.txt"
        else:
            traineval_fname = None
            testeval_fname = None
               
        for fold in data_split:
            trainseqs_id = data_split[fold]['train']
            # create model using the sequences assigned for training
            model_repr = seq_representer.create_model(trainseqs_id, seqs_info, model_repr_class, self.filter_obj)
            # extract for each sequence model active features
            seq_representer.extract_seqs_modelactivefeatures(seqs_id, seqs_info, model_repr, "f{}".format(fold))
            
            # create a CRF model
            crf_model = model_class(model_repr, seq_representer, seqs_info, load_info_fromdisk = 4)
            # get the directory of the trained model
            savedmodel_info = self.train_model(trainseqs_id, crf_model)      
            # evaluate on the training data 
            trainseqs_info = {seq_id:seqs_info[seq_id] for seq_id in trainseqs_id} 
            self.eval_model(savedmodel_info, {'seqs_info':trainseqs_info},
                            traineval_fname, "dec_trainseqs_fold_{}.txt".format(fold), sep=sep)
           
            # evaluate on the test data 
            testseqs_id = data_split[fold].get('test')
            if(testseqs_id):
                testseqs_info = {seq_id:seqs_info[seq_id] for seq_id in testseqs_id} 
                self.eval_model(savedmodel_info, {'seqs_info':testseqs_info}, 
                                testeval_fname, "dec_testseqs_fold_{}.txt".format(fold), sep=sep)

            models_info.append(savedmodel_info)
        # save workflow trainer instance on disk
        ReaderWriter.dump_data(self, os.path.join(ref_corpusdir, 'workflow_trainer'))
        return(models_info)
    
    def train_model(self, trainseqs_id, crf_model):
        """ training a model and return the directory of the trained model"""
        
        working_dir = self.working_dir
        optimization_options = self.optimization_options
        learner = Learner(crf_model) 
        learner.train_model(crf_model.weights, trainseqs_id, optimization_options, working_dir)
        
        return(learner.training_description['model_dir'])
    
    def eval_model(self, savedmodel_info, eval_seqs, eval_filename, dec_seqs_filename, sep = " "):
        # load learned models
        model_dir = savedmodel_info
        modelparts_dir = os.path.join(model_dir, "model_parts")
        modelrepr_class = self.model_repr_class
        model_class = self.model_class        
        fextractor_class = self.fextractor_class
        aextractor_class = self.aextractor_class
        seqrepresenter_class = SeqsRepresenter
        # revive/generate learned model
        crf_model  = generate_updated_model(modelparts_dir, modelrepr_class,  model_class, 
                               aextractor_class, fextractor_class, 
                               seqrepresenter_class,ascaler_class=None)
        # decode sequences to file
        if(eval_seqs.get('seqs_info')):
            seqs_pred = crf_model.decode_seqs("viterbi", model_dir, seqs_info = eval_seqs['seqs_info'], file_name = dec_seqs_filename, sep = sep)
        elif(eval_seqs.get('seqs')):
            seqs_pred = crf_model.decode_seqs("viterbi", model_dir, seqs = eval_seqs['seqs'], file_name = dec_seqs_filename, sep = sep)
        
        # evaluate model
        if(eval_filename):
            Y_seqs_dict = self.map_pred_to_ref_seqs(seqs_pred)
            evaluator = Evaluator(crf_model.model)
            performance = evaluator.compute_model_performance(Y_seqs_dict, 'f1', os.path.join(model_dir, eval_filename), "")
            print("performance ", performance)

    def map_pred_to_ref_seqs(self, seqs_pred):
        Y_seqs_dict = {}
#         print("seqs_pred {}".format(seqs_pred))
        for seq_id in seqs_pred:
            Y_seqs_dict[seq_id] = {}
            for seq_label in seqs_pred[seq_id]:
                if(seq_label == "seq"):
                    val = seqs_pred[seq_id][seq_label].flat_y
                    key = "Y_ref"
                else:
                    val = seqs_pred[seq_id][seq_label]
                    key = seq_label
                Y_seqs_dict[seq_id][key] = val
        return(Y_seqs_dict)
    
    def verify_template(self):
        """ verifying template -- sanity check"""
        seqs_id = self.seqs_id
        model = self.model
        crf_model = self.crf_model
        num_globalfeatures = model.num_features
        f = set()
        for seq_id in seqs_id:
            crf_model.load_activefeatures(seq_id)
            seq_activefeatures = crf_model.seqs_info[seq_id]["activefeatures"]
            for features_dict in seq_activefeatures.values():
                for z_patt in features_dict:
                    f.update(set(features_dict[z_patt][0]))
            crf_model.clear_cached_info([seq_id])
        num_activefeatures = len(f)
                    
        statement = ""
        if(num_activefeatures < num_globalfeatures): 
            statement = "len(activefeatures) < len(modelfeatures)"
        elif(num_activefeatures > num_globalfeatures):
            statement = "len(activefeatures) > len(modelfeatures)"
        else:
            statement = "PASS"
        print(statement)
        
class TrainingWorkflowIterative(object):
    r"""general training workflow that support reading/preparing **large** training sets
       
    """
    def __init__(self, template_y, template_xy, model_repr_class, model_class,
                 fextractor_class, aextractor_class, scaling_method, ascaler_class, 
                 optimization_options, root_dir, data_parser_options, filter_obj = None):
        self.template_y = template_y
        self.template_xy = template_xy
        self.model_class = model_class
        self.model_repr_class = model_repr_class
        self.fextractor_class = fextractor_class
        self.aextractor_class = aextractor_class
        self.scaling_method = scaling_method
        self.ascaler_class = ascaler_class
        self.optimization_options = optimization_options
        self.root_dir = root_dir
        self.data_parser_options = data_parser_options
        self.filter_obj = filter_obj

    def get_seqs_from_file(self, seq_file):
        parser = self.data_parser_options['parser']
        header = self.data_parser_options['header']
        col_sep = self.data_parser_options['col_sep']
        seg_other_symbol = self.data_parser_options['seg_other_symbol']
        for seq in parser.read_file(seq_file, header, 
                                    column_sep=col_sep,
                                    seg_other_symbol=seg_other_symbol):
            yield seq
    
    def build_seqsinfo(self, seq_file):
        seq_representer = self.seq_representer
        # create working directory 
        corpus_name = "reference_corpus_" + generate_datetime_str()
        working_dir = create_directory("working_dir", self.root_dir)
        self.working_dir = working_dir
        unique_id = False
        # build the seqs_info by parsing the sequences from file iteratively
        seqs_info = {}
        counter=1
        for seq in self.get_seqs_from_file(seq_file):
            if(hasattr(seq, 'id')):
                seq_id = seq.id
            else:
                seq_id = counter
            seqs_info.update(seq_representer.prepare_seqs({seq_id:seq}, corpus_name, working_dir, unique_id, log_progress=False))
            print("{} sequences have been processed".format(counter))
            counter+=1    
        return(seqs_info)
    
    def seq_parsing_workflow(self, seq_file, split_options):
        """preparing sequences to be used in the learning framework"""
        
        # initialize attribute extractor
        attr_extractor = self.aextractor_class()

        # create the feature extractor
        scaling_method = self.scaling_method        
        fextractor_class = self.fextractor_class
        f_extractor = fextractor_class(self.template_xy, self.template_y, attr_extractor.attr_desc)
        
        # create sequence representer
        seq_representer = SeqsRepresenter(attr_extractor, f_extractor)
        self.seq_representer = seq_representer
        # build the seqs_info by parsing the sequences from file iteratively
        seqs_info = self.build_seqsinfo(seq_file)
        seqs_id = list(seqs_info.keys())
        self.seqs_id = seqs_id
        
        # preprocess and generate attributes in case of segments with length >1 or in case of scaling of
        # attributes is needed               
        seq_representer.preprocess_attributes(seqs_id, seqs_info, method = scaling_method)
        
        # extract global features F(X,Y)
        seq_representer.extract_seqs_globalfeatures(seqs_id, seqs_info)
        
        # save the link to seqs_info and seq_representer as instance variables
        self.seqs_info = seqs_info
        self.seq_representer = seq_representer
        
        # split dataset according to the specified split options
        data_split = self.split_dataset(seqs_info, split_options)
        
        # save the datasplit dictionary on disk 
        gfeatures_dir = seqs_info[1]['globalfeatures_dir']
        ref_corpusdir = os.path.dirname(os.path.dirname(gfeatures_dir))
        ReaderWriter.dump_data(data_split, os.path.join(ref_corpusdir, 'data_split'))
        return(data_split)

    def split_dataset(self, seqs_info, split_options):
        if(split_options['method'] == "wsample"):
            # try weighted sample
            # first group seqs based on length
            grouped_seqs = group_seqs_by_length(seqs_info)
            # second get a weighted sample based on seqs length
            w_sample = weighted_sample(grouped_seqs, trainset_size=split_options['trainset_size'])
            print("w_sample ", w_sample)
            # third aggregate the seqs in training category and testing category
            data_split = aggregate_weightedsample(w_sample)
        elif(split_options['method'] == "cross_validation"):
            # try cross validation
            seqs_id = list(seqs_info.keys())
            data_split = split_data(seqs_id, split_options)
        elif(split_options['method'] == 'random'):
            seqs_id = list(seqs_info.keys())
            data_split = split_data(seqs_id, split_options)
        elif(split_options['method'] == 'none'):
            seqs_id = list(seqs_info.keys())
            data_split = {0:{'train':seqs_id}}    
        return(data_split)

    def traineval_folds(self, data_split, **kwargs):
        """train and evaluate model on different dataset splits"""
        
        seqs_id = self.seqs_id
        seq_representer = self.seq_representer
        seqs_info = self.seqs_info
        model_repr_class = self.model_repr_class
        model_class = self.model_class
        models_info = []
        ref_corpusdir = os.path.dirname(os.path.dirname(seqs_info[1]['globalfeatures_dir']))
        
        info_fromdisk = kwargs.get('load_info_fromdisk')
        # specify large number such that we always load the computed data from disk rather keeping them in memory
        if(type(info_fromdisk) != int):
            info_fromdisk = 10
        elif(info_fromdisk < 0):
            info_fromdisk = 10
        # check if file name is specified
        file_name = kwargs.get('file_name')
        for fold in data_split:
            track_perf = {}
            for dtype in ('train', 'test'):
                fold_seqs_id = data_split[fold].get(dtype)
                if(dtype == 'train'):
                    # create model using the sequences assigned for training
                    model_repr = seq_representer.create_model(fold_seqs_id, seqs_info, model_repr_class, self.filter_obj)
                    # extract for each sequence model active features
                    seq_representer.extract_seqs_modelactivefeatures(seqs_id, seqs_info, model_repr, "f{}".format(fold))
                    # create a CRF model
                    crf_model = model_class(model_repr, seq_representer, seqs_info, load_info_fromdisk = info_fromdisk)
                    # get the directory of the trained model
                    savedmodel_dir = self.train_model(fold_seqs_id, crf_model)      
                if(fold_seqs_id):
                    # evaluate on the current data fold 
                    fold_name = '{}_f{}'.format(dtype, fold)
                    fold_seqs_info = {seq_id:seqs_info[seq_id] for seq_id in fold_seqs_id}
                    kwargs['seqs_info'] = fold_seqs_info 
                    
                    if(file_name):
                        # add prefix
                        update_filename = fold_name + "_" + file_name
                        kwargs['file_name'] = update_filename
                    
                    res = self.eval_model(savedmodel_dir, kwargs)
                    res['fold_name'] = fold_name
                    res['model_dir'] = savedmodel_dir
                    models_info.append(res)                
        # save workflow trainer instance on disk
        ReaderWriter.dump_data(self, os.path.join(ref_corpusdir, 'workflow_trainer'))
        return(models_info)
    
    def train_model(self, trainseqs_id, crf_model):
        """ training a model and return the directory of the trained model"""
        
        working_dir = self.working_dir
        optimization_options = self.optimization_options
        learner = Learner(crf_model) 
        learner.train_model(crf_model.weights, trainseqs_id, optimization_options, working_dir)
        
        return(learner.training_description['model_dir'])
    
    def get_learned_crf(self, savedmodel_dir):
        # load learned models
        model_dir = savedmodel_dir
        modelparts_dir = os.path.join(model_dir, "model_parts")
        modelrepr_class = self.model_repr_class
        model_class = self.model_class        
        fextractor_class = self.fextractor_class
        aextractor_class = self.aextractor_class
        seqrepresenter_class = SeqsRepresenter
        ascaler_class = self.ascaler_class
        # revive/generate learned model
        crf_model  = generate_updated_model(modelparts_dir, modelrepr_class,  model_class, 
                                            aextractor_class, fextractor_class, 
                                            seqrepresenter_class,ascaler_class=ascaler_class)
        return(crf_model)
    
    def eval_model(self, savedmodel_dir, options):
        # load learned models
        model_dir = savedmodel_dir
        # revive/generate learned model
        crf_model  = self.get_learned_crf(model_dir)

        # parse the arguments in kwargs
        seqbatch_size = options.get("seqbatch_size")
        if(not seqbatch_size):
            seqbatch_size = 1000
        # check if model evaluation is requested
        model_eval = options.get('model_eval')
        if(model_eval):
            evaluator = SeqDecodingEvaluator(crf_model.model)
            perf_metric = options.get('metric')
            if(not perf_metric):
                perf_metric = 'f1'
            exclude_states = options.get('exclude_states')
            if(not exclude_states):
                exclude_states = []
        
        if(options.get('seqs_info')):
            # decode sequences 
            seqs_info = options.get('seqs_info')
            seqs_id = list(seqs_info.keys())
            start_ind = 0
            stop_ind = seqbatch_size
            while(start_ind<len(seqs_id)):
                batch_seqsinfo = {seq_id:seqs_info[seq_id] for seq_id in seqs_id[start_ind:stop_ind]}              
                seqs_pred = crf_model.decode_seqs("viterbi", model_dir, seqs_info=batch_seqsinfo, 
                                                  file_name=options.get('file_name'), sep=options.get('sep'),
                                                  beam_size=options.get('beam_size'))
                if(model_eval):
                    Y_seqs_dict = self.map_pred_to_ref_seqs(seqs_pred)
                    if(start_ind == 0):
                        taglevel_perf = evaluator.compute_states_confmatrix(Y_seqs_dict)
                    else:
                        taglevel_perf += evaluator.compute_states_confmatrix(Y_seqs_dict)
                start_ind+=seqbatch_size
                stop_ind+=seqbatch_size
        
        # TO adjust the batch size and available sequences..
        elif(options.get('seq_file')):       
            flag = False
            seq_file = options.get('seq_file')
            # the folder name where intermediary sequences and data are stored
            procseqs_foldername = "processed_seqs_" + generate_datetime_str()
            seqs_dict = {}
            bcounter = 1
            seq_counter = 1
            for seq in self.get_seqs_from_file(seq_file):
                seqs_dict[seq_counter] = seq
                if(bcounter >= seqbatch_size):
                    seqs_pred = crf_model.decode_seqs("viterbi", model_dir, seqs_dict=seqs_dict, 
                                                      procseqs_foldername=procseqs_foldername, file_name=options.get('file_name'),
                                                      sep=options.get('sep'), beam_size=options.get('beam_size'))
                    bcounter = 0
                    seqs_dict.clear()
                    if(model_eval):
                        Y_seqs_dict = self.map_pred_to_ref_seqs(seqs_pred)
                        if(seq_counter == seqbatch_size):
                            taglevel_perf = evaluator.compute_states_confmatrix(Y_seqs_dict)
                            flag = True
                        else:
                            taglevel_perf += evaluator.compute_states_confmatrix(Y_seqs_dict)
                bcounter += 1
                seq_counter+=1
            if(len(seqs_dict)):
                # decode the remaining sequences
                seqs_pred = crf_model.decode_seqs("viterbi", model_dir, seqs_dict=seqs_dict, 
                                                  procseqs_foldername=procseqs_foldername, file_name=options.get('file_name'),
                                                  sep=options.get('sep'), beam_size=options.get('beam_size'))
                if(model_eval):
                    Y_seqs_dict = self.map_pred_to_ref_seqs(seqs_pred)
                    if(flag):
                        taglevel_perf += evaluator.compute_states_confmatrix(Y_seqs_dict)
                    else:
                        taglevel_perf = evaluator.compute_states_confmatrix(Y_seqs_dict)
        if(model_eval):
            performance = evaluator.get_performance_metric(taglevel_perf, perf_metric, exclude_states=exclude_states)
            return({perf_metric:performance, 'taglevel_confusion_matrix':taglevel_perf})
        return({})

    def map_pred_to_ref_seqs(self, seqs_pred):
        Y_seqs_dict = {}
#         print("seqs_pred {}".format(seqs_pred))
        for seq_id in seqs_pred:
            Y_seqs_dict[seq_id] = {}
            for seq_label in seqs_pred[seq_id]:
                if(seq_label == "seq"):
                    val = seqs_pred[seq_id][seq_label].flat_y
                    key = "Y_ref"
                else:
                    val = seqs_pred[seq_id][seq_label]
                    key = seq_label
                Y_seqs_dict[seq_id][key] = val
        return(Y_seqs_dict)
    
    def verify_template(self):
        """ verifying template -- sanity check"""

        seqs_id = self.seqs_id
        model = self.model
        crf_model = self.crf_model
        num_globalfeatures = model.num_features
        f = set()
        for seq_id in seqs_id:
            crf_model.load_activefeatures(seq_id)
            seq_activefeatures = crf_model.seqs_info[seq_id]["activefeatures"]
            for features_dict in seq_activefeatures.values():
                for z_patt in features_dict:
                    f.update(set(features_dict[z_patt][0]))
            crf_model.clear_cached_info([seq_id])
        num_activefeatures = len(f)
                    
        statement = ""
        if(num_activefeatures < num_globalfeatures): 
            statement = "len(activefeatures) < len(modelfeatures)"
        elif(num_activefeatures > num_globalfeatures):
            statement = "len(activefeatures) > len(modelfeatures)"
        else:
            statement = "PASS"
        print(statement)

        