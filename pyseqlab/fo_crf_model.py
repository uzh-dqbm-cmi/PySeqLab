'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''

import os
from copy import deepcopy
from collections import OrderedDict
import numpy
from .utilities import FO_AStarSearcher, ReaderWriter, create_directory, vectorized_logsumexp

class FirstOrderCRFModelRepresentation(object):
    def __init__(self):
        """ modelfeatures: set of features defining the model
            state: set of states (i.e. tags) 
            L: 1 (segment length)
        """ 
        self.modelfeatures = None
        self.modelfeatures_codebook = None
        self.Y_codebook = None
        self.Y_codebook_rev = None
        self.startstate_flag = None
        self.L = None
        self.Z_codebook = None
        self.Z_lendict = None
        self.Z_elems = None
        self.Z_numchar= None
        self.patts_len = None
        self.max_patt_len = None
        self.modelfeatures_inverted = None
        self.ypatt_features = None
        self.ypatt_activestates = None
        self.num_features = None
        self.num_states = None
        
    def create_model(self, modelfeatures, states, L):
        """modelfeatures: set of features defining the model
           states: set of states (i.e. tags)
           L: length of longest segment
        """
        self.modelfeatures = modelfeatures
        self.modelfeatures_codebook = self.get_modelfeatures_codebook()
        self.Y_codebook = self.get_modelstates_codebook(states)
        self.Y_codebook_rev = self.get_Y_codebook_reversed()
        self.L = L
        self.generate_instance_properties()
    
    def generate_instance_properties(self):
        self.Z_codebook = self.get_Z_pattern()
        self.Z_lendict, self.Z_elems, self.Z_numchar = self.get_Z_elems_info()
        self.patts_len = set(self.Z_lendict.values())
        self.max_patt_len = max(self.patts_len)

        self.modelfeatures_inverted, self.ypatt_features = self.get_inverted_modelfeatures()
        self.ypatt_activestates = self.find_activated_states(self.ypatt_features, self.patts_len)
        self.num_features = self.get_num_features()
        self.num_states = self.get_num_states()
        
    def get_modelfeatures_codebook(self):
        modelfeatures = self.modelfeatures
        codebook = {}
        code = 0
        for y_patt, featuresum in modelfeatures.items():
            for feature in featuresum:
                fkey = y_patt + "&&" + feature
                codebook[fkey] = code
                code += 1
        return(codebook)
    
    def get_modelstates_codebook(self, states):
        start_state = '__START__'
        if(start_state in states):
            del states[start_state]
            Y_codebook = {s:i+1 for (i, s) in enumerate(states)}
            Y_codebook[start_state] = 0
            states[start_state] = 1
            self.startstate_flag = True
        else:
            Y_codebook = {s:i for (i, s) in enumerate(states)}
            self.startstate_flag = False
        return(Y_codebook)  
        
    def get_Y_codebook_reversed(self):
        Y_codebook = self.Y_codebook
        return({code:state for state, code in Y_codebook.items()})
        
    def get_Z_pattern(self):
        modelfeatures = self.modelfeatures
        Z_codebook = {y_patt:index for index, y_patt in enumerate(modelfeatures)}
        return(Z_codebook)
    
    def get_Z_elems_info(self):
        Z_codebook = self.Z_codebook
        Z_lendict = {}
        Z_split_elems = {}
        Z_nchar = {}
        for z in Z_codebook:
            elems = z.split("|")
            Z_lendict[z] = len(elems)
            Z_split_elems[z] = elems
            Z_nchar[z] = len(z)            
        return(Z_lendict, Z_split_elems, Z_nchar)
    
    def get_inverted_modelfeatures(self):
        modelfeatures = self.modelfeatures
        Z_lendict = self.Z_lendict
        inverted_segfeatures = {}
        ypatt_features = set()
        
        for y_patt, featuresum in modelfeatures.items():
            z_len = Z_lendict[y_patt]
            # get features that are based only on y_patts
            if(y_patt in featuresum):
                ypatt_features.add(y_patt)
            for feature in featuresum:
                if(feature in inverted_segfeatures):
                    if(z_len in inverted_segfeatures[feature]):
                        inverted_segfeatures[feature][z_len].add(y_patt)
                    else:
                        s = set()
                        s.add(y_patt)                      
                        inverted_segfeatures[feature][z_len] = s
                else:
                    s = set()
                    s.add(y_patt)
                    inverted_segfeatures[feature] = {z_len:s}
        return(inverted_segfeatures, ypatt_features)
    
    def represent_globalfeatures(self, seq_featuresum):
        modelfeatures_codebook = self.modelfeatures_codebook
        windx_fval = {}
        for y_patt, seg_features in seq_featuresum.items():
            for seg_featurename in seg_features: 
                fkey = y_patt + "&&" + seg_featurename
                if(fkey in modelfeatures_codebook):
                    windx_fval[modelfeatures_codebook[fkey]] = seg_features[seg_featurename]
        return(windx_fval)
    
    def represent_activefeatures(self, activestates, seg_features):  
        modelfeatures = self.modelfeatures
        modelfeatures_codebook = self.modelfeatures_codebook   
        Z_codebook = self.Z_codebook      
        activefeatures = {}
#         print("segfeatures {}".format(seg_features))
#         print("z_patts {}".format(z_patts))
        for z_len in activestates:
            z_patt_set = activestates[z_len]
            for z_patt in z_patt_set:
#                 print("z_patt ", z_patt)
                windx_fval = {}
                for seg_featurename in seg_features:
                    # this condition might be omitted 
                    if(seg_featurename in modelfeatures[z_patt]):
    #                         print("seg_featurename {}".format(seg_featurename))
    #                         print("z_patt {}".format(z_patt))
                        fkey = z_patt + "&&" + seg_featurename
                        #print(fkey)
                        windx_fval[modelfeatures_codebook[fkey]] = seg_features[seg_featurename]     
                if(z_patt in modelfeatures[z_patt]):
                    fkey = z_patt + "&&" + z_patt
                    windx_fval[modelfeatures_codebook[fkey]] = 1
                    #print(fkey)
                    
                if(windx_fval):
                    #activefeatures[Z_codebook[z_patt]] = windx_fval
                    activefeatures[z_patt] = windx_fval

#         print("activefeatures {}".format(activefeatures))         
        return(activefeatures)     
    
    def find_activated_states(self, seg_features, allowed_z_len):
        modelfeatures_inverted = self.modelfeatures_inverted
        active_states = {}
        for feature in seg_features:
            if(feature in modelfeatures_inverted):
                factivestates = modelfeatures_inverted[feature]
                for z_len in factivestates:
                    if(z_len in allowed_z_len):
                        if(z_len in active_states):
                            active_states[z_len].update(factivestates[z_len])
                        else:
                            active_states[z_len] = set(factivestates[z_len])
                #print("active_states from func ", active_states)
        return(active_states)

    def filter_activated_states(self, activated_states, accum_active_states, pos):
        Z_elems = self.Z_elems
        filtered_activestates = {}
        
        for z_len in activated_states:
            if(z_len == 1):
                continue
            start_pos = pos - z_len + 1
            if(start_pos in accum_active_states):
                filtered_activestates[z_len] = set()
                for z_patt in activated_states[z_len]:
                    check = True
                    zelems = Z_elems[z_patt]
                    for i in range(z_len):
                        if(start_pos+i not in accum_active_states):
                            check = False
                            break
                        if(zelems[i] not in accum_active_states[start_pos+i]):
                            check = False
                            break
                    if(check):                        
                        filtered_activestates[z_len].add(z_patt)
        return(filtered_activestates)
    
    def get_num_features(self):
        return(len(self.modelfeatures_codebook))
    def get_num_states(self):
        return(len(self.Y_codebook))

class FirstOrderCRF(object):
    """ First order CRF
    """
    def __init__(self, model, seqs_representer, seqs_info, load_info_fromdisk=6):
        self.model = model
        self.weights = numpy.zeros(model.num_features, dtype= "longdouble")
        self.seqs_representer = seqs_representer
        self.seqs_info = seqs_info
        self.func_dict = {"alpha": self._load_alpha,
                         "beta": self._load_beta,
                         "activated_states": self.load_activatedstates,
                         "seg_features": self.load_segfeatures,
                         "globalfeatures": self.load_globalfeatures,
                         "globalfeatures_per_boundary": self.load_globalfeatures,
                         "potential_matrix": self._load_potentialmatrix,
                         "activefeatures": self.load_activefeatures,
                         "Y":self._load_Y}
        
        self.def_cached_entities = self.cached_entitites(load_info_fromdisk)
        # default beam size 
        self.beam_size = len(self.model.Y_codebook)
        
    def cached_entitites(self, load_info_fromdisk):
        ondisk_info = ["activefeatures", "l_segfeatures", "seg_features", "activated_states", "globalfeatures_per_boundary", "globalfeatures", "Y"]
        def_cached_entities = ondisk_info[:load_info_fromdisk]
        inmemory_info = ["alpha", "Z", "beta", "potential_matrix", "P_marginal"]
        def_cached_entities += inmemory_info
        return(def_cached_entities)
       
    @property
    def seqs_info(self):
        return self._seqs_info
    @seqs_info.setter
    def seqs_info(self, info_dict):
        # make a copy of the passed seqs_info dictionary
        self._seqs_info = deepcopy(info_dict)

    
    def identify_activefeatures(self, seq_id, boundary, accum_activestates):
        model = self.model
        state_len = 1
        # get activated states per boundary
        activated_states = self.seqs_info[seq_id]['activated_states'][boundary]
        seg_features = self.seqs_info[seq_id]['seg_features'][boundary]
        #^print("boundary ", boundary)
        #^print('seg_features ', seg_features)
        #^print('activated_states ', activated_states)
        #^print("accum_activestates ", accum_activestates)
        u, v = boundary
    
        if(state_len in activated_states):
            accum_activestates[v] = set(activated_states[state_len])
            
            if(boundary != (1,1)):
                filtered_states =  model.filter_activated_states(activated_states, accum_activestates, u)
                filtered_states[state_len] = set(activated_states[state_len])
            # initial point t0
            else:
                filtered_states = activated_states
            
            #print("filtered_states ", filtered_states)
            #print("seg_features ", seg_features)        
            active_features = model.represent_activefeatures(filtered_states, seg_features)

        else:
            accum_activestates[v] = set()
            active_features = {}
        
        return(active_features)   
    
    def generate_activefeatures(self, seq_id):
        # generate active features for every boundary of the sequence 
        # to be used when using gradient-based methods for learning
        T = self.seqs_info[seq_id]["T"]
        accum_activestates = {}
        activefeatures_perboundary = {}
        for j in range(1, T+1):
            boundary = (j, j)
            # identify active features
            active_features = self.identify_activefeatures(seq_id, boundary, accum_activestates)
            activefeatures_perboundary[boundary] = active_features
        return(activefeatures_perboundary)
            
    def compute_psi_potential(self, w, seq_id):
        """ assumes that activefeatures_matrix has been already generated and saved in self.seqs_info dictionary """
        Y_codebook = self.model.Y_codebook
        Z_lendict = self.model.Z_lendict
        Z_elems = self.model.Z_elems
        # T is the length of the sequence 
        T = self.seqs_info[seq_id]["T"]
        # number of possible states including the __START__ and __STOP__ states
        M = self.model.num_states
        # get activefeatures_matrix
        activefeatures = self.seqs_info[seq_id]["activefeatures"]
        potential_matrix = numpy.zeros((T+1,M,M), dtype='longdouble')
        for boundary in activefeatures:
            t = boundary[0]
            for y_patt in activefeatures[boundary]:
                f_val = list(activefeatures[boundary][y_patt].values())
                w_indx = list(activefeatures[boundary][y_patt].keys())
                
                potential = numpy.dot(w[w_indx], f_val)
                if(Z_lendict[y_patt] == 1):
                    y_c = Z_elems[y_patt][0]
                    potential_matrix[t, :, Y_codebook[y_c]] += potential
                else:
                    # case of len(parts) = 2
                    y_p = Z_elems[y_patt][0]
                    y_c = Z_elems[y_patt][1]
                    potential_matrix[t, Y_codebook[y_p], Y_codebook[y_c]] += potential
#         print("potential_matrix {}".format(potential_matrix))
        return(potential_matrix)
    
    def compute_potential(self, w, active_features):
        """ potential matrix for given active features at a specified boundary """
        Y_codebook = self.model.Y_codebook
        Z_lendict = self.model.Z_lendict
        Z_elems = self.model.Z_elems
        # number of possible states including the __START__ and __STOP__ states
        M = self.model.num_states
        # get activefeatures_matrix
        potential_matrix = numpy.zeros((M,M), dtype='longdouble')

        for y_patt in active_features:
            f_val = list(active_features[y_patt].values())
            w_indx = list(active_features[y_patt].keys())
            potential = numpy.dot(w[w_indx], f_val)
            if(Z_lendict[y_patt] == 1):
                y_c = Z_elems[y_patt][0]
                potential_matrix[:, Y_codebook[y_c]] += potential
            else:
                # case of len(parts) = 2
                y_p = Z_elems[y_patt][0]
                y_c = Z_elems[y_patt][1]
                potential_matrix[Y_codebook[y_p], Y_codebook[y_c]] += potential
#         print("potential_matrix {}".format(potential_matrix))
        return(potential_matrix)

    def compute_forward_vec(self, w, seq_id):
        """ assumes the potential_matrix has been already computed and saved in self.seqs_info dictionary """
        # T is the length of the sequence 
        T = self.seqs_info[seq_id]["T"]
        # number of possible states including the __START__ and __STOP__ states
        M = self.model.num_states
        startstate_flag = self.model.startstate_flag
        # get the potential matrix 
        potential_matrix = self.seqs_info[seq_id]["potential_matrix"]
        alpha = numpy.ones((T+1, M), dtype='longdouble') * (-numpy.inf)
        
        if(startstate_flag):
            alpha[0,0] = 0
        # corner case at t = 1
        t = 1; i = 0
        alpha[t, :] = potential_matrix[t, i, :]
        for t in range(1, T):
            for j in range(M):
                alpha[t+1, j] = vectorized_logsumexp(alpha[t, :] + potential_matrix[t+1, :, j])

        return(alpha)
  
    def compute_backward_vec(self, w, seq_id):
        # length of the sequence without the appended states __START__ and __STOP__
        T = self.seqs_info[seq_id]["T"]
        # number of possible states including the __START__ and __STOP__ states
        M = self.model.num_states
        beta = numpy.ones((T+1, M), dtype = 'longdouble') * (-numpy.inf)
        beta[T, :] = 0
        # get the potential matrix 
        potential_matrix = self.seqs_info[seq_id]["potential_matrix"]
        for t in reversed(range(1, T+1)):
            for i in range(M):
                beta[t-1, i] = vectorized_logsumexp(potential_matrix[t, i, :] + beta[t, :])

        return(beta) 

    
    def save_model(self, file_name):
        # to clean things before pickling the model
        self.seqs_info.clear() 
        ReaderWriter.dump_data(self, file_name)
        
    def _load_alpha(self, w, seq_id):
        seq_info = self.seqs_info[seq_id]
        # assumes the potential matrix has been loaded into seq_info
        seq_info["alpha"] = self.compute_forward_vec(w, seq_id)
        seq_info["Z"] = vectorized_logsumexp(seq_info["alpha"][-1,:])
#         print("... Computing alpha probability ...")

    def _load_beta(self, w, seq_id):
        # assumes the potential matrix has been loaded into seq_info
        self.seqs_info[seq_id]["beta"] = self.compute_backward_vec(w, seq_id)
#         print("... Computing beta probability ...")

    def _load_Y(self, seq_id):
        seq = self._load_seq(seq_id, target="seq")
        self.seqs_info[seq_id]['Y'] = {'flat_y':seq.flat_y, 'boundaries':seq.y_sboundaries}
#         print("... loading Y ...")
        
    def _load_potentialmatrix(self, w, seq_id):
        # assumes the activefeatures_by_position has been loaded into seq_info
        # compute potential matrix
        self.seqs_info[seq_id]["potential_matrix"] = self.compute_psi_potential(w, seq_id)
#         print("... Computing potential matrix ...")

    def load_activatedstates(self, seq_id):
        # get the sequence activated states
        seqs_info = self.seqs_info
        seqs_representer = self.seqs_representer
        activated_states = seqs_representer.get_seq_activatedstates(seq_id, seqs_info)
        seqs_info[seq_id]["activated_states"] = activated_states
        #print("loading activated states")
    
    def load_segfeatures(self, seq_id):
        # get the sequence segment features
        seqs_info = self.seqs_info
        seqs_representer = self.seqs_representer
        seg_features = seqs_representer.get_seq_segfeatures(seq_id, seqs_info)
        self.seqs_info[seq_id]["seg_features"] = seg_features
        #print("loading segment features")
        
    def load_activefeatures(self, seq_id):
        # get the sequence model active features
        seqs_representer = self.seqs_representer
        activefeatures = seqs_representer.get_seq_activefeatures(seq_id, self.seqs_info)
        if(not activefeatures):
            # check if activated_states and seg_features are loaded
            l = {}
            l['activated_states'] = (seq_id, )
            l['seg_features'] = (seq_id, )
            self.check_cached_info(seq_id, l)
            activefeatures = self.generate_activefeatures(seq_id)
            seq_dir = self.seqs_info[seq_id]['activefeatures_dir']
            ReaderWriter.dump_data(activefeatures, os.path.join(seq_dir, 'activefeatures'))
        
        self.seqs_info[seq_id]["activefeatures"] = activefeatures

             
    def load_globalfeatures(self, seq_id, per_boundary=True):
        # get sequence global features
        seqs_representer = self.seqs_representer
        gfeatures_perboundary = seqs_representer.get_seq_globalfeatures(seq_id, self.seqs_info, per_boundary=per_boundary)
#         print("per_boundary ", per_boundary)
#         print(gfeatures_perboundary)
        if(per_boundary):
            fname = "globalfeatures_per_boundary"
        else:
            fname = "globalfeatures"
        self.seqs_info[seq_id][fname] = gfeatures_perboundary
        #print("loading globalfeatures")

        
    def load_imposter_globalfeatures(self, seq_id, y_imposter, seg_other_symbol):
        # get sequence global features
        seqs_representer = self.seqs_representer
        imposter_gfeatures_perboundary, y_imposter_boundaries = seqs_representer.get_imposterseq_globalfeatures(seq_id, self.seqs_info, y_imposter, seg_other_symbol)
        return(imposter_gfeatures_perboundary, y_imposter_boundaries)
   
    def represent_globalfeature(self, gfeatures, boundaries):
        # get sequence global features
        seqs_representer = self.seqs_representer
        windx_fval = seqs_representer.represent_gfeatures(gfeatures, self.model, boundaries=boundaries)        
        return(windx_fval)  
     
    def _load_seq(self, seq_id, target = "seq"):
        seqs_representer = self.seqs_representer
        seq = seqs_representer.load_seq(seq_id, self.seqs_info)
        if(target == "seq"):
            return(seq)
        elif(target == "Y"):
            return(seq.Y)
        elif(target == "X"):
            return(seq.X)

    def check_cached_info(self, seq_id, entity_names):
        """order of elements in the entity_names list is important """
        seq_info = self.seqs_info[seq_id]
        func_dict = self.func_dict
        none_type = type(None) 
        for varname, args in entity_names.items():
            if(type(seq_info.get(varname)) == none_type):
                func_dict[varname](*args)

    def clear_cached_info(self, seqs_id, cached_entities = []):
        args = self.def_cached_entities + cached_entities
        for seq_id in seqs_id:
            seq_info = self.seqs_info[seq_id]
            for varname in args:
                if(varname in seq_info):
                    seq_info[varname] = None

    
    def compute_seqs_loglikelihood(self, w, seqs_id):
        seqs_loglikelihood = 0
        for seq_id in seqs_id:
            seqs_loglikelihood += self.compute_seq_loglikelihood(w, seq_id)
        return(seqs_loglikelihood)
    
    def compute_seqs_gradient(self, w, seqs_id):
        seqs_grad = numpy.zeros(len(w))
        for seq_id in seqs_id:
            seq_grad = self.compute_seq_gradient(w, seq_id) 
            w_indx = list(seq_grad.keys())
            f_val = list(seq_grad.values())
            seqs_grad[w_indx] += f_val
        return(seqs_grad)

        
    def compute_seq_loglikelihood(self, w, seq_id):
        """computes the conditional log-likelihood of a sequence (i.e. p(Y|X;w)) 
           it is used as a cost function for the single sequence when trying to estimate parameters w
        """
#         print("-"*40)
#         print("... Evaluating compute_seq_loglikelihood() ...")
        
        # we need global features and alpha matrix to be ready -- order is important
        l = OrderedDict()
        l['globalfeatures'] = (seq_id, False)
        l['activefeatures'] = (seq_id, )
        l['potential_matrix'] = (w, seq_id)
        l['alpha'] = (w, seq_id)
        self.check_cached_info(seq_id, l)
        
        # get the p(X;w) -- probability of the sequence under parameter w
        Z = self.seqs_info[seq_id]["Z"]
        
        gfeatures = self.seqs_info[seq_id]["globalfeatures"]
        globalfeatures = self.represent_globalfeature(gfeatures, None)
        windx = list(globalfeatures.keys())
        fval = list(globalfeatures.values())
        # log(p(Y|X;w))
        loglikelihood = numpy.dot(w[windx], fval) - Z 
        self.seqs_info[seq_id]["loglikelihood"] = loglikelihood
        
        return(loglikelihood)

    def compute_seq_gradient(self, w, seq_id):
        """ 
           compute the gradient of conditional log-likelihood with respect to the parameters vector w
           \frac{\partial}{\partial w} p(Y|X;w)
           
           Params:
           -------
           w: vector representing current weights -- array shape (J,) where J is the total number of features
           seq_id: id of the current sequence being evaluated -- string
        """
#         print("-"*40)
#         print("... Evaluating compute_seq_gradient() ...")


        # we need alpha, beta, global features and active features  to be ready
        l = OrderedDict()
        l['globalfeatures'] = (seq_id, False)
        l['activefeatures'] = (seq_id, )
        l['potential_matrix'] = (w, seq_id)
        l['alpha'] = (w, seq_id)
        l['beta'] = (w, seq_id)
        self.check_cached_info(seq_id, l)
        
        P_marginals = self.compute_marginals(seq_id)
        self.seqs_info[seq_id]["P_marginal"] = P_marginals
        
        f_expectation = self.compute_feature_expectation(seq_id)
        gfeatures = self.seqs_info[seq_id]["globalfeatures"]
        globalfeatures = self.represent_globalfeature(gfeatures, None)
#         print("globalfeatures in grad \n", globalfeatures)
#         print("seq id {}".format(seq_id))
#         print("len(f_expectation) {}".format(len(f_expectation)))
#         print("len(globalfeatures) {}".format(len(globalfeatures)))
        

        if(len(f_expectation) > len(globalfeatures)):
            missing_features = f_expectation.keys() - globalfeatures.keys()
            addendum = {w_indx:0 for w_indx in missing_features}
            globalfeatures.update(addendum)
            #print("normal case --len(f_expectation) > len(globalfeatures)")


#         print("P_marginals {}".format(P_marginals))
#         print("f_expectation {}".format(f_expectation))
#         print("globalfeatures {}".format(globalfeatures))
        
        grad = {}
        for w_indx in f_expectation:
            grad[w_indx]  = globalfeatures[w_indx] - f_expectation[w_indx]
        return(grad)
    

    def compute_marginals(self, seq_id):
        Y_codebook = self.model.Y_codebook
        Z_codebook = self.model.Z_codebook
        Z_lendict = self.model.Z_lendict
        Z_elems = self.model.Z_elems
        T = self.seqs_info[seq_id]["T"]

        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"] 
        Z = self.seqs_info[seq_id]["Z"]   
#         print("alpha {}".format(alpha))
#         print("beta {}".format(beta))
#         print("Z {}".format(Z))
        
        potential_matrix = self.seqs_info[seq_id]["potential_matrix"]
        P_marginals = numpy.zeros((T+1, len(Z_codebook)), dtype='longdouble') 
         
#         print("Z_codebook {}".format(Z_codebook))
        for j in range(1, T+1):
            for y_patt in Z_codebook:
#                 print("y_patt {}".format(y_patt))
                if(Z_lendict[y_patt] == 1):
                    y_c = Y_codebook[Z_elems[y_patt][0]]
                    accumulator = alpha[j, y_c] + beta[j, y_c] - Z
                else:
                    # case of len(parts) = 2
                    y_b = Y_codebook[Z_elems[y_patt][0]]
                    y_c = Y_codebook[Z_elems[y_patt][1]]
                    accumulator = alpha[j-1, y_b] + potential_matrix[j, y_b, y_c] + beta[j, y_c] - Z
                P_marginals[j, Z_codebook[y_patt]] = numpy.exp(accumulator)
        return(P_marginals)
    
    def compute_feature_expectation(self, seq_id):
        """ assumes that activefeatures_matrix has been already generated and saved in self.seqs_info dictionary """
        activefeatures = self.seqs_info[seq_id]["activefeatures"]
        P_marginals = self.seqs_info[seq_id]["P_marginal"]
        Z_codebook = self.model.Z_codebook
        f_expectation = {}
        for boundary, features_dict in activefeatures.items():
            t = boundary[0]
            for z_patt in features_dict:
                for w_indx, f_val in features_dict[z_patt].items():
                    if(w_indx in f_expectation):
                        f_expectation[w_indx] += f_val * P_marginals[t, Z_codebook[z_patt]]
                    else:
                        f_expectation[w_indx] = f_val * P_marginals[t, Z_codebook[z_patt]]
        return(f_expectation)
    

            
    def decode_seqs(self, decoding_method, out_dir, **kwargs):
        """ decoding_method: a string referring to type of decoding {'viterbi', 'per_state_decoding'}
            **kwargs could be one of the following:
                seqs: a list comprising of sequences that are instances of SequenceStrcut() class
                seqs_info: dictionary containing the info about the sequences to parse
        """
        corpus_name = "decoding_seqs"
        w = self.weights

        if(decoding_method == "perstate_decoding"):
            decoder = self.perstate_posterior_decoding
        else:
            decoder = self.viterbi


        file_name = kwargs.get('file_name')
        if(file_name):
            out_file = os.path.join(create_directory(corpus_name, out_dir), file_name)
            if(kwargs.get("sep")):
                sep = kwargs['sep']
            else:
                sep = "\t"
        
        beam_size = kwargs.get('beam_size')
        if(not beam_size):
            beam_size = self.beam_size
            
        if(kwargs.get("seqs_info")):
            self.seqs_info = kwargs["seqs_info"]
            N = len(self.seqs_info)
        elif(kwargs.get("seqs")): 
            seqs = kwargs["seqs"]           
            seqs_dict = {i+1:seqs[i] for i in range(len(seqs))}
            seqs_id = list(seqs_dict.keys())
            N = len(seqs_id)
            seqs_info = self.seqs_representer.prepare_seqs(seqs_dict, "processed_seqs", out_dir, unique_id = True)
            self.seqs_representer.scale_attributes(seqs_id, seqs_info)
            self.seqs_representer.extract_seqs_modelactivefeatures(seqs_id, seqs_info, self.model, "processed_seqs", learning=False)
            self.seqs_info = seqs_info
            


        seqs_pred = {}
        seqs_info = self.seqs_info
        counter = 0
        for seq_id in seqs_info:
            Y_pred, __ = decoder(w, seq_id, beam_size)
            seq = ReaderWriter.read_data(os.path.join(seqs_info[seq_id]["globalfeatures_dir"], "sequence"))
            if(file_name):
                self.write_decoded_seqs([seq], [Y_pred], out_file, sep)
            seqs_pred[seq_id] = {'seq': seq,'Y_pred': Y_pred}
            # clear added info per sequence
            self.clear_cached_info([seq_id])
            counter += 1
            print("sequence decoded -- {} sequences are left".format(N-counter))
        
        # clear seqs_info
        self.seqs_info.clear()
        return(seqs_pred)

            
    def write_decoded_seqs(self, ref_seqs, Y_pred_seqs, out_file, sep = "\t"):
        for i in range(len(ref_seqs)):
            Y_pred_seq = Y_pred_seqs[i]
            ref_seq = ref_seqs[i]
            T = ref_seq.T
            line = ""
            for t in range(1, T+1):
                for field_name in ref_seq.X[t]:
                    line += ref_seq.X[t][field_name] + sep
                if(ref_seq.flat_y):
                    line += ref_seq.flat_y[t-1] + sep
                line += Y_pred_seq[t-1]
                line += "\n" 
            line += "\n"
            ReaderWriter.log_progress(line,out_file) 
            
    def prune_states(self, j, score_mat, beam_size):
        Y_codebook_rev = self.model.Y_codebook_rev
        # using argpartition as better alternative to argsort
        indx_partitioned_y = numpy.argpartition(-score_mat[j, :], beam_size)
        # identify top-k states/pi
        indx_topk_y = indx_partitioned_y[:beam_size]
#         # identify states falling out of the beam
        indx_falling_y = indx_partitioned_y[beam_size:]
        # remove the effect of states/pi falling out of the beam
        score_mat[j, indx_falling_y] = -numpy.inf
        
        # get topk states
        topk_y = {Y_codebook_rev[indx] for indx in indx_topk_y}
        
        return(topk_y)
    
    def viterbi(self, w, seq_id, beam_size, stop_off_beam = False, y_ref=[], K=1):

        # number of possible states
        M = self.model.num_states
        T = self.seqs_info[seq_id]['T']
        Y_codebook_rev = self.model.Y_codebook_rev
        Y_codebook = self.model.Y_codebook
        score_mat = numpy.ones((T+1, M), dtype='longdouble') * -numpy.inf
        score_mat[0,0] = 0
        # back pointer to hold the index of the state that achieved highest score while decoding
        backpointer = numpy.ones((T+1, M), dtype='int') * (-1)
        viol_index = []
        
        if(beam_size == M):
            # case of exact search and decoding
            l = OrderedDict()
            l['activefeatures'] = (seq_id, )
            l['potential_matrix'] = (w, seq_id)
            self.check_cached_info(seq_id, l)
            
            potential_matrix = self.seqs_info[seq_id]["potential_matrix"]
            # corner case at t = 1
            t = 1; i = 0
            score_mat[t, :] = potential_matrix[t, i, :]
            backpointer[t, :] = 0
            
            for t in range(2, T+1):
                for j in range(M):
                    vec = score_mat[t-1, :] + potential_matrix[t, :, j]
                    score_mat[t, j] = numpy.max(vec)
                    backpointer[t, j] = numpy.argmax(vec)
        else:
            # case of inexact search and decoding
            l = {}
            l['activated_states'] = (seq_id, )
            l['seg_features'] = (seq_id, )
            self.check_cached_info(seq_id, l)
            
            accum_activestates = {}

            for t in range(1, T+1):
                boundary = (t, t)
                active_features = self.identify_activefeatures(seq_id, boundary, accum_activestates)
                potential_matrix = self.compute_potential(w, active_features)
                for j in range(M):
                    vec = score_mat[t-1, :] + potential_matrix[:, j]
                    score_mat[t, j] = numpy.max(vec)
                    backpointer[t, j] = numpy.argmax(vec)
                    
                topk_states = self.prune_states(t, score_mat, beam_size)
                # update tracked active states -- to consider renaming it          
                accum_activestates[t] = accum_activestates[t].intersection(topk_states)
                #^print("accum_activestates[{}] = {}".format(j, accum_activestates[j]))
                #^print('score_mat[{},:] = {} '.format(j, score_mat[j,:]))
                #^print("topk_states ", topk_states)
                if(y_ref):
                    if(y_ref[t-1] not in topk_states):
                        viol_index.append(t)
                        if(stop_off_beam):
                            T = t
                            break
        if(K == 1):
            # decoding the sequence
            y_c_T = numpy.argmax(score_mat[T:])
            Y_decoded = [int(y_c_T)]
            counter = 0
            for t in reversed(range(2, T+1)):
                Y_decoded.append(int(backpointer[t, Y_decoded[counter]]))
                counter += 1
            Y_decoded.reverse()
           
            Y_decoded = [Y_codebook_rev[y_code] for y_code in Y_decoded]
            return(Y_decoded, viol_index)
        else:
            asearcher = FO_AStarSearcher(Y_codebook, Y_codebook_rev)
            topK = asearcher.search(score_mat, backpointer, T, K)
            return(topK, viol_index)
    
    def perstate_posterior_decoding(self, w, seq_id):
        Y_codebook_rev = self.model.Y_codebook_rev
        # get alpha, beta and Z
        l = OrderedDict()
        l['activefeatures'] = (seq_id, )
        l['potential_matrix'] = (w, seq_id)
        l['alpha'] = (w, seq_id)
        l['beta'] = (w, seq_id)
        self.check_cached_info(seq_id, l)
        
        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"]
        Z = self.seqs_info[seq_id]["Z"]
#         print("alpha \n {}".format(alpha))
#         print("beta \n {}".format(beta))
        score_mat = alpha + beta - Z
#         print("score mat is \n {}".format(score_mat))
        # remove the corner cases t=0  and t=T+1
        score_mat_ = score_mat[:,1:-1]
        max_indices = list(numpy.argmax(score_mat_, axis = 0))
#         print("max indices \n {}".format(max_indices))
        
        Y_decoded = max_indices
        Y_decoded = [Y_codebook_rev[y_code] for y_code in Y_decoded]
        return(Y_decoded)


    def check_gradient(self, w, seq_id):
        """ implementation of finite difference method similar to scipy.optimize.check_grad()
        """
        self.clear_cached_info([seq_id])
        epsilon = 1e-9
        # basis vector
        ei = numpy.zeros(len(w), dtype="longdouble")
        grad = numpy.zeros(len(w), dtype="longdouble")
        for i in range(len(w)):
            ei[i] = epsilon
            l_wplus = self.compute_seq_loglikelihood(w + ei, seq_id)
            self.clear_cached_info([seq_id])
            l = self.compute_seq_loglikelihood(w, seq_id)
            self.clear_cached_info([seq_id])
            grad[i] = (l_wplus - l) / epsilon
            ei[i] = 0
        estimated_grad = self.compute_seqs_gradient(w, [seq_id])
        print("finite difference gradient: \n {}".format(grad))
        print("Computed gradient: \n {}".format(estimated_grad))
        diff = numpy.absolute(-grad + estimated_grad)
        avg_diff = numpy.mean(diff)
        print("difference between both gradients: \n {}".format(diff))
        print("average difference = {}".format(numpy.mean(avg_diff)))
        # clear seq_id info
        self.clear_cached_info([seq_id])
        return(avg_diff)
        
    def validate_forward_backward_pass(self, w, seq_id):
        self.clear_cached_info([seq_id])
        # this will compute alpha and beta matrices and save them in seqs_info dict
        l = OrderedDict()
        l['activefeatures'] = (seq_id, )
        l['potential_matrix'] = (w, seq_id)
        l['alpha'] = (w, seq_id)
        l['beta'] = (w, seq_id)
        self.check_cached_info(seq_id, l)
        
        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"]
        print("states codebook {}".format(self.model.Y_codebook))
        print("alpha {}".format(alpha))
        print("beta {}".format(beta))
        
        Z_alpha = vectorized_logsumexp(alpha[-1,:])
        Z_beta = numpy.max(beta[0, :])
        raw_diff = numpy.abs(Z_alpha - Z_beta)
        print("alpha[-1,:] = {}".format(alpha[-1,:]))
        print("beta[0,:] = {}".format(beta[0,:]))
        print("Z_alpha : {}".format(Z_alpha))
        print("Z_beta : {}".format(Z_beta))
        print("Z_aplha - Z_beta {}".format(raw_diff))
 
        rel_diff = raw_diff/(Z_alpha + Z_beta)
        print("rel_diff : {}".format(rel_diff))
        self.clear_cached_info([seq_id])
        print("seqs_info {}".format(self.seqs_info))
        return((raw_diff, rel_diff))
    
    def validate_expected_featuresum(self, w, seqs_id):
        self.clear_cached_info(seqs_id)
        grad = self.compute_seqs_gradient(w, seqs_id)
        avg_diff = numpy.mean(grad)
        print("difference between empirical feature sum and model's expected feature sum: \n {}".format(grad))
        print("average difference is {}".format(avg_diff))
        self.clear_cached_info(seqs_id)
        return(avg_diff)
    
if __name__ == "__main__":
    pass
    