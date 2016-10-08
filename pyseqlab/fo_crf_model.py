'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''

import os
from copy import deepcopy
import numpy
from utilities import ReaderWriter, create_directory, vectorized_logsumexp

class FirstOrderCRFModelRepresentation(object):
    def __init__(self, modelfeatures, states, L = 1):
        """ modelfeatures: set of features defining the model
            state: set of states (i.e. tags) 
            L: 1 (segment length)
        """ 
        
        self.modelfeatures = modelfeatures
        self.modelfeatures_codebook = modelfeatures
        self.Y_codebook = states
        self.L = 1
        self.Y_codebook_rev = self.get_Y_codebook_reversed()
        self.Z_codebook = self.get_pattern_set()
        self.patt_order = self.get_pattern_order()
        self.num_features = self.get_num_features()
        self.num_states = self.get_num_states()
        
    @property
    def modelfeatures_codebook(self):
        return(self._modelfeatures_codebook)
    @modelfeatures_codebook.setter
    def modelfeatures_codebook(self, modelfeatures):
        codebook = {}
        code = 0
        for y_patt, featuresum_dict in modelfeatures.items():
            for feature in featuresum_dict:
                fkey = y_patt + "&&" + feature
                codebook[fkey] = code
                code += 1
        self._modelfeatures_codebook = codebook

    @property
    def Y_codebook(self):
        return(self._Y_codebook)
    @Y_codebook.setter     
    def Y_codebook(self, states):
        start_state = '__START__'
        check = start_state in states
        if(check):
            del states[start_state]
            Y_codebook = {s:i+1 for (i, s) in enumerate(states)}
            Y_codebook[start_state] = 0
            states[start_state] = 1
        else:
            Y_codebook = {s:i for (i, s) in enumerate(states)}
            
        self._Y_codebook = Y_codebook  
        
    def get_Y_codebook_reversed(self):
        Y_codebook = self.Y_codebook
        return({code:state for state, code in Y_codebook.items()})    

    def get_pattern_set(self):
        modelfeatures = self.modelfeatures
        Z_codebook = {y_patt:index for index, y_patt in enumerate(modelfeatures)}
        return(Z_codebook)
    
    def get_pattern_order(self):
        modelfeatures = self.modelfeatures
        patt_order = {}
        for y_patt in modelfeatures:
            elems = y_patt.split("|")
            l = len(elems)
            if(l in patt_order):
                patt_order[l].append(y_patt)
            else:
                patt_order[l] = [y_patt]
        return(patt_order)

    def get_num_features(self):
        return(len(self.modelfeatures_codebook))
    def get_num_states(self):
        return(len(self.Y_codebook))
    
    def represent_globalfeatures(self, seq_featuresum):
        modelfeatures_codebook = self.modelfeatures_codebook
        windx_fval = {}
        for y_patt, seg_features in seq_featuresum.items():
            for seg_featurename in seg_features: 
                fkey = y_patt + "&&" + seg_featurename
                if(fkey in modelfeatures_codebook):
                    windx_fval[modelfeatures_codebook[fkey]] = seg_features[seg_featurename]
        return(windx_fval)
    
        
    def represent_activefeatures(self, z_patts, seg_features):  
        modelfeatures = self.modelfeatures
        modelfeatures_codebook = self.modelfeatures_codebook 
        activefeatures = {}
#         print("segfeatures {}".format(seg_features))
#         print("z_patts {}".format(z_patts))
        for z_patt in z_patts:
            if(z_patt in modelfeatures):
                windx_fval = {}
                for seg_featurename in seg_features:
                    if(seg_featurename in modelfeatures[z_patt]):
#                         print("seg_featurename {}".format(seg_featurename))
#                         print("z_patt {}".format(z_patt))
                        fkey = z_patt + "&&" + seg_featurename
                        windx_fval[modelfeatures_codebook[fkey]] = seg_features[seg_featurename]     
                if(z_patt in modelfeatures[z_patt]):
                    fkey = z_patt + "&&" + z_patt
                    windx_fval[modelfeatures_codebook[fkey]] = 1
                    
                if(windx_fval):
                    activefeatures[z_patt] = windx_fval
#         print("activefeatures {}".format(activefeatures))         
        return(activefeatures)

"""
First order CRF
"""
class FirstOrderCRF(object):
    def __init__(self, model, seqs_representer, seqs_info):
        self.model = model
        self.weights = numpy.zeros(model.num_features, dtype= "longdouble")
        self.seqs_representer = seqs_representer
        self.seqs_info = seqs_info
        self.func_dict = {"alpha": self._load_alpha,
                         "beta": self._load_beta,
                         "potential_matrix": self._load_potentialmatrix,
                         "activefeatures_by_position": self.load_activefeatures,
                         "globalfeatures": self.load_globalfeatures,
                         "flat_y":self._load_flaty}
    @property
    def seqs_info(self):
        return self._seqs_info
    @seqs_info.setter
    def seqs_info(self, info_dict):
        # make a copy of the passed seqs_info dictionary
        self._seqs_info = deepcopy(info_dict)
    
    def compute_psi_potential(self, w, seq_id):
        """ assumes that activefeatures_matrix has been already generated and saved in self.seqs_info dictionary """
        Y_codebook = self.model.Y_codebook
        # T is the length of the sequence 
        T = self.seqs_info[seq_id]["T"]
        # number of possible states including the __START__ and __STOP__ states
        M = self.model.num_states
        # get activefeatures_matrix
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
        potential_matrix = numpy.zeros((T+1,M,M), dtype='longdouble')

        for boundary, features_dict in activefeatures.items():
            t = boundary[0]
            for y_patt, windx_fval_dict in features_dict.items():
                f_val = list(windx_fval_dict.values())
                w_indx = list(windx_fval_dict.keys())
                potential = numpy.dot(w[w_indx], f_val)
                parts = y_patt.split("|")
                if(len(parts) == 1):
                    y_c = parts[0]
                    potential_matrix[t, :, Y_codebook[y_c]] += potential
                else:
                    # case of len(parts) = 2
                    y_p = parts[0]
                    y_c = parts[1]
                    potential_matrix[t, Y_codebook[y_p], Y_codebook[y_c]] += potential
#         print("potential_matrix {}".format(potential_matrix))
        return(potential_matrix)

    def compute_forward_vec(self, w, seq_id):
        """ assumes the potential_matrix has been already computed and saved in self.seqs_info dictionary """
        # T is the length of the sequence 
        T = self.seqs_info[seq_id]["T"]
        # number of possible states including the __START__ and __STOP__ states
        M = self.model.num_states
        # get the potential matrix 
        potential_matrix = self.seqs_info[seq_id]["potential_matrix"]
        alpha = numpy.ones((T+1, M), dtype='longdouble') * (-numpy.inf)
        if("__START__" in self.model.Y_codebook):
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

    def _load_flaty(self, w, seq_id):
        seq = self._load_seq(seq_id, target="seq")
        self.seqs_info[seq_id]['flat_y'] = seq.flat_y
        
    def _load_potentialmatrix(self, w, seq_id):
        # assumes the activefeatures_by_position has been loaded into seq_info
        # compute potential matrix
        self.seqs_info[seq_id]["potential_matrix"] = self.compute_psi_potential(w, seq_id)
#         print("... Computing potential matrix ...")

    def load_activefeatures(self, w, seq_id):
        # get the sequence model active features
        seqs_representer = self.seqs_representer
        seqs_activefeatures = seqs_representer.get_seqs_modelactivefeatures([seq_id], self.seqs_info)
        self.seqs_info[seq_id]["activefeatures_by_position"] = seqs_activefeatures[seq_id]
        
    def load_globalfeatures(self, w, seq_id):
        # get sequence global features
        seqs_representer = self.seqs_representer
        seqs_globalfeatures = seqs_representer.get_seqs_globalfeatures([seq_id], self.seqs_info, self.model)
        self.seqs_info[seq_id]["globalfeatures"] = seqs_globalfeatures[seq_id]

    def load_imposter_globalfeatures(self, seq_id, y_imposter, seg_other_symbol):
        # get sequence global features
        seqs_representer = self.seqs_representer
        #print("seq_info[{}] = {}".format(seq_id, self.seqs_info[seq_id]))
        imposter_globalfeatures = seqs_representer.get_imposterseq_globalfeatures(seq_id, self.seqs_info, self.model, y_imposter)
        return(imposter_globalfeatures)
    
    def _load_seq(self, seq_id, target = "seq"):
        seqs_representer = self.seqs_representer
        seq = seqs_representer.load_seq(seq_id, self.seqs_info)
        if(target == "seq"):
            return(seq)
        elif(target == "Y"):
            return(seq.Y)
        elif(target == "X"):
            return(seq.X)

    def check_cached_info(self, w, seq_id, entity_names):
        seq_info = self.seqs_info[seq_id]
        func_dict = self.func_dict
        none_type = type(None) 
        for varname in entity_names:
            if(type(seq_info.get(varname)) == none_type):
                func_dict[varname](w, seq_id)
                    
    def clear_cached_info(self, seqs_id, cached_entities = []):
        default_entitites = ["potential_matrix", "alpha", "Z", "beta", "P_marginal"]
        args = cached_entities + default_entitites
        for seq_id in seqs_id:
            seq_info = self.seqs_info[seq_id]
            for varname in args:
                if(varname in seq_info):
                    seq_info[varname] = None

        
    def reset_seqs_info(self, seqs_id):
        for seq_id in seqs_id:
            self.seqs_info[seq_id] = {"T": self.seqs_info[seq_id]["T"],
                                      "globalfeatures_dir": self.seqs_info[seq_id]["globalfeatures_dir"],
                                      "activefeatures_dir": self.seqs_info[seq_id]["activefeatures_dir"]
                                      }
            

    
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
        
        # we need alpha and global features to be ready
        l = ("globalfeatures", "activefeatures_by_position", "potential_matrix", "alpha")
        self.check_cached_info(w, seq_id, l)
        
        # get the p(X;w) -- probability of the sequence under parameter w
        Z = self.seqs_info[seq_id]["Z"]
        globalfeatures = self.seqs_info[seq_id]["globalfeatures"]
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


        # we need alpha, beta, global features and active features per position to be ready
        l = ("globalfeatures", "activefeatures_by_position", "potential_matrix","alpha", "beta")
        self.check_cached_info(w, seq_id, l)
        
        P_marginals = self.compute_marginals(seq_id)
        self.seqs_info[seq_id]["P_marginal"] = P_marginals
        f_expectation = self.compute_feature_expectation(seq_id)
        globalfeatures = self.seqs_info[seq_id]["globalfeatures"]
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
                parts = y_patt.split("|")
                if(len(parts) == 1):
                    y_c = Y_codebook[parts[0]]
                    accumulator = alpha[j, y_c] + beta[j, y_c] - Z
                else:
                    # case of len(parts) = 2
                    y_b = Y_codebook[parts[0]]
                    y_c = Y_codebook[parts[1]]
                    accumulator = alpha[j-1, y_b] + potential_matrix[j, y_b, y_c] + beta[j, y_c] - Z
                P_marginals[j, Z_codebook[y_patt]] = numpy.exp(accumulator)
        return(P_marginals)
    
    def compute_feature_expectation(self, seq_id):
        """ assumes that activefeatures_matrix has been already generated and saved in self.seqs_info dictionary """
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
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
        """ seqs: a list comprising of sequences that are instances of SequenceStrcut() class
            method: a string referring to type of decoding {'viterbi', 'per_state_decoding'}
            seqs_info: dictionary containing the info about the sequences to parse
        """
        corpus_name = "decoding_seqs"
        out_file = os.path.join(create_directory(corpus_name, out_dir), "decoded.txt")
        w = self.weights
        
        if(decoding_method == "viterbi"):
            decoder = self.viterbi
        elif(decoding_method == "perstate_decoding"):
            decoder = self.perstate_posterior_decoding
            
        if(kwargs.get("seqs_info")):
            self.seqs_info = kwargs["seqs_info"]
        elif(kwargs.get("seqs")): 
            seqs = kwargs["seqs"]           
            seqs_dict = {i+1:seqs[i] for i in range(len(seqs))}
            seqs_info = self.seqs_representer.prepare_seqs(seqs_dict, corpus_name, out_dir, unique_id = True)
            self.seqs_info = seqs_info

        seqs_pred = {}
        seqs_info = self.seqs_info
        for seq_id in seqs_info:
            Y_pred = decoder(w, seq_id)
            seq = ReaderWriter.read_data(os.path.join(seqs_info[seq_id]["globalfeatures_dir"], "sequence"))
            self.write_decoded_seqs([seq], [Y_pred], out_file)
            seqs_pred[seq_id] = {'seq': seq,'Y_pred': Y_pred}
        # clear seqs_info
        self.seqs_info.clear()
        return(seqs_pred)

            
    def write_decoded_seqs(self, ref_seqs, Y_pred_seqs, out_file):
        sep = " "
        for i in range(len(ref_seqs)):
            Y_pred_seq = Y_pred_seqs[i]
            ref_seq = ref_seqs[i]
            T = ref_seq.T
            line = ""
            for t in range(1, T+1):
                for field_name in ref_seq.X[t]:
                    line += ref_seq.X[t][field_name] + sep
                line += Y_pred_seq[t-1]
                if(ref_seq.flat_y):
                    line += sep + ref_seq.flat_y[t-1] + "\n"
                else:
                    line += "\n" 

            ReaderWriter.log_progress(line, out_file) 
            

    def viterbi(self, w, seq_id):

        # number of possible states including the __START__ and __STOP__ states
        M = self.model.num_states
        T = self.seqs_info[seq_id]['T']
        Y_codebook_rev = self.model.Y_codebook_rev
        score_mat = numpy.zeros((T+1, M))
        # compute potential matrix 
        l = ("activefeatures_by_position", "potential_matrix")
        self.check_cached_info(w, seq_id, l)
        potential_matrix = self.seqs_info[seq_id]["potential_matrix"]
        # corner case at t = 1
        t = 1; i = 0
        score_mat[t, :] = potential_matrix[t, i, :]
        # back pointer to hold the index of the state that achieved highest score while decoding
        backpointer = numpy.ones((T+1, M)) * (-1)
        backpointer[t, :] = 0
        for t in range(2, T+1):
            for j in range(M):
                vec = score_mat[t-1, :] + potential_matrix[t, :, j]
                score_mat[t, j] = numpy.max(vec)
                backpointer[t, j] = numpy.argmax(vec)
        
        # decoding the sequence
        y_T = numpy.argmax(backpointer[-1,:])
        Y_decoded = [int(y_T)]
        counter = 0
        for t in reversed(range(2, T+1)):
            Y_decoded.append(int(backpointer[t, Y_decoded[counter]]))
            counter += 1
        Y_decoded.reverse()
       
        print("decoding sequence with id {} \n".format(seq_id))
        Y_decoded = [Y_codebook_rev[y_code] for y_code in Y_decoded]
        return(Y_decoded)
    
    def perstate_posterior_decoding(self, w, seq_id):
        # get alpha, beta and Z
        Y_codebook_rev = self.model.Y_codebook_rev
        l = ("activefeatures_by_position", "potential_matrix", "alpha", "beta")
        self.check_cached_info(w, seq_id, l)
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
            l = self.compute_seqs_loglikelihood(w, [seq_id])
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
        l = ("activefeatures_by_position", "potential_matrix", "alpha", "beta")
        self.check_cached_info(w, seq_id, l)
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
    