'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''

import os
from copy import deepcopy
import numpy
from utilities import ReaderWriter, create_directory, vectorized_logsumexp
 
class HOSemiCRFModelRepresentation(object):
    def __init__(self, modelfeatures, states, L):
        """ modelfeatures: set of features defining the model
            states: set of states (i.e. tags)
            L: length of longest segment
        """ 
        self.modelfeatures = modelfeatures
        self.modelfeatures_codebook = modelfeatures
        self.Y_codebook = states
        self.L = L
        self.Z_codebook = self.get_pattern_set()
        self.patt_order = self.get_pattern_order()
        self.P_codebook = self.get_forward_states()
        self.S_codebook = self.get_backward_states()
        self.f_transition = self.get_forward_transition()
        self.b_transition = self.get_backward_transitions()
        self.pky_z = self.map_pky_z()
        self.siy_z = self.map_siy_z()
        self.z_piy = self.map_z_piy()
        self.num_features = self.get_num_features()
        self.num_states = self.get_num_states()

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
    
    @property
    def modelfeatures_codebook(self):
        return(self._modelfeatures_codebook)
    @modelfeatures_codebook.setter
    def modelfeatures_codebook(self, modelfeatures):
        codebook = {}
        code = 0
        for y_patt, featuresum in modelfeatures.items():
            for feature in featuresum:
                fkey = y_patt + "&&" + feature
                codebook[fkey] = code
                code += 1
        self._modelfeatures_codebook = codebook

    @property
    def Y_codebook(self):
        return(self._Y_codebook)
    @Y_codebook.setter     
    def Y_codebook(self, states):
        self._Y_codebook = {s:i for (i, s) in enumerate(states)}
        
    def get_pattern_set(self):
        modelfeatures = self.modelfeatures
        Z_codebook = {y_patt:index for index, y_patt in enumerate(modelfeatures)}
        return(Z_codebook)
    
    def get_pattern_order(self):
        Z_codebook = self.Z_codebook
        patt_order = {}
        for y_patt in Z_codebook:
            elems = y_patt.split("|")
            l = len(elems)
            if(l in patt_order):
                patt_order[l].append(y_patt)
            else:
                patt_order[l] = [y_patt]
        return(patt_order)  
    
    def get_forward_states(self):
        Y_codebook = self.Y_codebook
        Z_codebook = self.Z_codebook
        P = {}
        for z_patt in Z_codebook:
            elems = z_patt.split("|")
            for i in range(len(elems)-1):
                P["|".join(elems[:i+1])] = 1
        for y in Y_codebook:
            P[y] = 1
        # empty element         
        P[""] = 1
        P_codebook = {s:i for (i, s) in enumerate(P)}
        return(P_codebook) 
    
    def get_backward_states(self):
        Y_codebook = self.Y_codebook
        P_codebook = self.P_codebook
        S = {}
        for p in P_codebook:
            if(p == ""):
                for y in Y_codebook:
                    S[y] = 1
            else:
                for y in Y_codebook:
                    S[p + "|" + y] = 1
        S_codebook = {s:i for (i, s) in enumerate(S)}
        return(S_codebook)
                
    def get_forward_transition(self):
        Y_codebook = self.Y_codebook
        P_codebook = self.P_codebook
        pk_y= {}
        for p in P_codebook:
            for y in Y_codebook:
                pk_y[(p, y)] = 1
        pk_y_suffix = {}
        for p in P_codebook:
            if(p != ""):
                for (pk, y) in pk_y:
                    ref_str = pk + "|" + y
                    check = self.check_suffix(p, ref_str)
                    if(check):
                        if((pk, y) in pk_y_suffix):
                            pk_y_suffix[(pk, y)].append(p)
                        else:
                            pk_y_suffix[(pk, y)] = [p]
                            
        pk_y_suffix = self.keep_largest_suffix(pk_y_suffix)
        f_transition = {}
        for (pk, y), pi in pk_y_suffix.items():
            if(pk == ""):
                elmkey = y
            else:
                elmkey = pk + "|" + y
            if(pi in f_transition):
                f_transition[pi][elmkey] = pk
            else:
                f_transition[pi] = {elmkey:pk}
        return(f_transition)
                    
    def get_backward_transitions(self):
        Y_codebook = self.Y_codebook
        S_codebook = self.S_codebook
        si_y_suffix = {}
        for si in S_codebook:
            si_y = {(si, y):1 for y in Y_codebook}
            for sk in S_codebook:
                for (si, y) in si_y:
                    check = self.check_suffix(sk, si + "|" + y)
                    if(check):
                        if((si, y) in si_y_suffix):
                            si_y_suffix[(si, y)].append(sk)
                        else:
                            si_y_suffix[(si, y)] = [sk]
        #print("si_y_suffix {}".format(si_y_suffix))
        si_y_suffix = self.keep_largest_suffix(si_y_suffix)
        #print("si_y_suffix {}".format(si_y_suffix))
    
        b_transition = {}
        for (si,y), sk in si_y_suffix.items():
            elmkey = si + "|" + y
            if(si in b_transition):
                b_transition[si][elmkey] = sk
            else:
                b_transition[si] = {elmkey:sk}
        return(b_transition)    
        
    def map_pky_z(self):
        f_transition = self.f_transition
        Z_codebook = self.Z_codebook
        pky_z_map = {}
        for pi in f_transition:
            for pky in f_transition[pi]:
                l = []
                for z in Z_codebook:
                    if(self.check_suffix(z, pky)):
                        l.append(z)
                pky_z_map[pky] = l
        return(pky_z_map)
    
    def map_siy_z(self):
        b_transition = self.b_transition
        Z_codebook = self.Z_codebook
        siy_z_map = {}
        for si in b_transition:
            for siy in b_transition[si]:
                l = []
                for z in Z_codebook:
                    if(self.check_suffix(z, siy)):
                        l.append(z)
                siy_z_map[siy] = l
        return(siy_z_map)   
  
    def map_z_piy(self):
        Y_codebook = self.Y_codebook
        Z_codebook = self.Z_codebook
        P_codebook = self.P_codebook
        z_piy = {}
        for z in Z_codebook:
            for p in P_codebook:
                for y in Y_codebook:
                    if(p == ""):
                        ref_str = y
                    else:
                        ref_str = p + "|" + y
                    if(self.check_suffix(z, ref_str)):
                        if(z in z_piy):
                            z_piy[z].update({(p, y):ref_str})
                        else:
                            z_piy[z] = {(p, y):ref_str}

        return(z_piy)
            
    def keep_largest_suffix(self, s):
        largest_suffix = {}
        for tup, l in s.items():
            largest_suffix[tup] = max(l, key = len)
        return(largest_suffix)
                  
    def check_suffix(self, token, ref_str):
        return(ref_str.endswith(token))

    def get_num_features(self):
        return(len(self.modelfeatures_codebook))
    def get_num_states(self):
        return(len(self.Y_codebook))
    
class HOSemiCRF(object):
    def __init__(self, model, seqs_representer, seqs_info):
        self.model = model
        self.weights = numpy.zeros(model.num_features, dtype= "longdouble")
        self.seqs_representer = seqs_representer
        self.seqs_info = seqs_info

    @property
    def seqs_info(self):
        return self._seqs_info
    @seqs_info.setter
    def seqs_info(self, info_dict):
        # make a copy of the passed seqs_info dictionary
        self._seqs_info = deepcopy(info_dict)

    def compute_psi_potential(self, w, seq_id, tup_z_map):
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
        psi_potential= {}
        for j in range(1, T+1):
            for d in range(L):
                if(j+d > T):
                    break
                boundary = (j, j+d)
                psi_potential[boundary] = {}
                if(activefeatures[boundary]):
                    for tup in tup_z_map:
                        potential = 0
                        for z_patt in tup_z_map[tup]:
                            if(z_patt in activefeatures[boundary]):
                                f_val = list(activefeatures[boundary][z_patt].values())
                                w_indx = list(activefeatures[boundary][z_patt].keys())
                                potential += numpy.dot(w[w_indx], f_val)
                        psi_potential[boundary][tup] = potential
                else:
                    potential = 0
                    for tup in tup_z_map:
                        psi_potential[boundary][tup] = potential
                        
        return(psi_potential)
    
    def compute_f_potential(self, w, seq_id):
        pky_z = self.model.pky_z
        f_potential = self.compute_psi_potential(w, seq_id, pky_z)
        return(f_potential)
    
    def compute_forward_vec(self, seq_id):
        f_potential = self.seqs_info[seq_id]["f_potential"]
        f_transition = self.model.f_transition
        P_codebook = self.model.P_codebook
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L
        alpha = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        alpha[0,P_codebook[""]] = 0
         
        for j in range(1, T+1):
            for pi in f_transition:
                accumulator = -numpy.inf
                for d in range(L):
                    u = j - d
                    v = j
                    if(u <= 0):
                        break
                    boundary = (u, v)
                    for pky, pk in f_transition[pi].items():
                        pk_code = P_codebook[pk]
                        potential = f_potential[boundary][pky]
                        accumulator = numpy.logaddexp(accumulator, potential + alpha[u-1, pk_code])
                alpha[j, P_codebook[pi]] = accumulator 
                 
        return(alpha)

    def compute_b_potential(self, w, seq_id):
        siy_z  = self.model.siy_z
        b_potential = self.compute_psi_potential(w, seq_id, siy_z)
        return(b_potential)
    
    def compute_backward_vec(self, seq_id):
        b_potential = self.seqs_info[seq_id]["b_potential"]
        b_transition = self.model.b_transition
        S_codebook = self.model.S_codebook
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L
        beta = numpy.ones((T+2,len(S_codebook)), dtype='longdouble') * (-numpy.inf)
        beta[T+1,] = 0
        for j in reversed(range(1, T+1)):
            for si in b_transition:
                accumulator = -numpy.inf
                for d in range(L):
                    u = j 
                    v = j + d
                    if(v > T):
                        break
                    boundary = (u, v)
                    for siy, sk in b_transition[si].items():
                        sk_code = S_codebook[sk]
                        potential = b_potential[boundary][siy]
                        accumulator = numpy.logaddexp(accumulator, potential + beta[j+d+1, sk_code])
                beta[j, S_codebook[si]] = accumulator 
                    
        return(beta)

    def compute_marginals(self, seq_id):
        f_potential = self.seqs_info[seq_id]["f_potential"]
        P_codebook = self.model.P_codebook
        S_codebook = self.model.S_codebook
        Z_codebook = self.model.Z_codebook
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L

        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"] 
        Z = self.seqs_info[seq_id]["Z"]   
        z_piy = self.model.z_piy

        P_marginals = numpy.zeros((L, T+1, len(self.model.Z_codebook)), dtype='longdouble')
         
        for j in range(1, T+1):
            for d in range(L):
                u = j
                v = j + d
                if(v > T):
                    break
                boundary = (u, v)
                for z_patt in z_piy:
                    accumulator = -numpy.inf
                    for (pi, y) in z_piy[z_patt]:
                        piy = z_piy[z_patt][(pi,y)]
                        numerator = alpha[u-1, P_codebook[pi]] + f_potential[boundary][piy] + beta[v+1, S_codebook[piy]]
                        accumulator = numpy.logaddexp(accumulator, numerator)
                    P_marginals[d, j, Z_codebook[z_patt]] = numpy.exp(accumulator - Z)
        print("P_marginals {}".format(P_marginals))
        return(P_marginals)
    
    def compute_feature_expectation(self, seq_id):
        """ assumes that activefeatures_matrix has been already generated and saved in self.seqs_info dictionary """
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
        P_marginals = self.seqs_info[seq_id]["P_marginal"]
        Z_codebook = self.model.Z_codebook
        f_expectation = {}
        for boundary, features_dict in activefeatures.items():
            u = boundary[0]
            v = boundary[1]
            d = v - u
            for z_patt in features_dict:
                for w_indx, f_val in features_dict[z_patt].items():
                    if(w_indx in f_expectation):
                        f_expectation[w_indx] += f_val * P_marginals[d, u, Z_codebook[z_patt]]
                    else:
                        f_expectation[w_indx] = f_val * P_marginals[d, u, Z_codebook[z_patt]]
        return(f_expectation)
    
    def compute_seq_loglikelihood(self, w, seq_id):
        """computes the conditional log-likelihood of a sequence (i.e. p(Y|X;w)) 
           it is used as a cost function for the single sequence when trying to estimate parameters w
        """
#         print("-"*40)
#         print("... Evaluating compute_seq_loglikelihood() ...")
        
        # we need alpha and global features to be ready
        l = ("globalfeatures", "activefeatures_by_position", "f_potential", "alpha")
        self.check_cached_info(w, seq_id, l)
        
        # get the p(X;w) -- probability of the sequence under parameter w
        Z = self.seqs_info[seq_id]["Z"]
        globalfeatures = self.seqs_info[seq_id]["globalfeatures"]
        w_indx = list(globalfeatures.keys())
        f_val = list(globalfeatures.values())

        # log(p(Y|X;w))
        loglikelihood = numpy.dot(w[w_indx], f_val) - Z 
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
        l = ("globalfeatures", "activefeatures_by_position", "f_potential", "alpha", "b_potential", "beta")
        self.check_cached_info(w, seq_id, l)
        P_marginals = self.compute_marginals(seq_id)
        self.seqs_info[seq_id]["P_marginal"] = P_marginals
        f_expectation = self.compute_feature_expectation(seq_id)
        globalfeatures = self.seqs_info[seq_id]["globalfeatures"]
#         print("seq id {}".format(seq_id))
#         print("len(f_expectation) {}".format(len(f_expectation)))
#         print("len(globalfeatures) {}".format(len(globalfeatures)))
        
        if(len(f_expectation) < len(globalfeatures)):
            missing_features = globalfeatures.keys() - f_expectation.keys()
            addendum = {w_indx:0 for w_indx in missing_features}
            f_expectation.update(addendum)  
            print("missing features len(f_expectation) < len(globalfeatures)")
        elif(len(f_expectation) > len(globalfeatures)):
            missing_features = f_expectation.keys() - globalfeatures.keys()
            addendum = {w_indx:0 for w_indx in missing_features}
            globalfeatures.update(addendum)
            print("missing features len(f_expectation) > len(globalfeatures)")

#         print("P_marginals {}".format(P_marginals))
#         print("f_expectation {}".format(f_expectation))
#         print("globalfeatures {}".format(globalfeatures))
        
        grad = {}
        for w_indx in f_expectation:
            grad[w_indx]  = globalfeatures[w_indx] - f_expectation[w_indx]
        return(grad)
    
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
    
    def check_cached_info(self, w, seq_id, entity_names):
        """ order of args is very important
            to compute 1 - alpha matrix we need to compute f_potential before
                       2 - beta matrix we need to compute the b_potential before
            
        """
        seq_info = self.seqs_info[seq_id]
        none_type = type(None) 
        for varname in entity_names:
            if(type(seq_info.get(varname)) == none_type):
                if(varname == "alpha"):
                    # assumes the f_potential has been loaded into seq_info
                    seq_info[varname] = self.compute_forward_vec(seq_id)
                    seq_info["Z"] = vectorized_logsumexp(seq_info[varname][-1,:])
#                     print("... Computing alpha probability ...")
                             
                elif(varname == "beta"):
                    # assumes the b_potential has been loaded into seq_info
                    seq_info[varname] = self.compute_backward_vec(seq_id)
#                     print("... Computing beta probability ...")
                         
                elif(varname == "f_potential"):
                    # assumes the activefeatures_by_position matrix has been loaded into seq_info
                    # compute forward potential matrix
                    seq_info["f_potential"] = self.compute_f_potential(w, seq_id)
#                     print("... Computing f_potential matrix ...")

                elif(varname == "b_potential"):
                    # assumes the activefeatures_by_position matrix has been loaded into seq_info
                    # compute backward potential matrix
                    seq_info["b_potential"] = self.compute_b_potential(w, seq_id)
#                     print("... Computing b_potential matrix ...")
        
                elif(varname == "activefeatures_by_position"):
                    # load the sequence model active features by position and save in seq_info
                    self.load_activefeatures(seq_id)
#                     print("... Loading active features ...")
                         
                elif(varname == "globalfeatures"):
                    # load the sequence global features and save it in seq_info
                    self.load_globalfeatures(seq_id)
#                     print("... Loading global features ...")
# 
#                 elif(varname == "seq"):
#                     seq = self._load_seq(seq_id, target="seq")
#                     seq_info["seq"] = seq
                        
                elif(varname == "flat_y"):
                    seq = self._load_seq(seq_id, target="seq")
                    seq_info['flat_y'] = seq.flat_y


    def clear_cached_info(self, seqs_id, cached_entities = []):
        default_entitites = ["f_potential", "alpha", "Z", "b_potential", "beta", "P_marginal"]
        args = cached_entities + default_entitites
        for seq_id in seqs_id:
            seq_info = self.seqs_info[seq_id]
            for varname in args:
                if(varname in seq_info):
                    seq_info[varname] = None
            
    def load_activefeatures(self, seq_id):
        # get the sequence model active features
        seqs_representer = self.seqs_representer
        seqs_activefeatures = seqs_representer.get_seqs_modelactivefeatures([seq_id], self.seqs_info)
        self.seqs_info[seq_id]["activefeatures_by_position"] = seqs_activefeatures[seq_id]
        
    def load_globalfeatures(self, seq_id):
        # get sequence global features
        seqs_representer = self.seqs_representer
        seqs_globalfeatures = seqs_representer.get_seqs_globalfeatures([seq_id], self.seqs_info, self.model)
        self.seqs_info[seq_id]["globalfeatures"] = seqs_globalfeatures[seq_id]
        
    def load_imposter_globalfeatures(self, seq_id, y_imposter, seg_other_symbol):
        # get sequence global features
        seqs_representer = self.seqs_representer
        imposter_globalfeatures = seqs_representer.get_imposterseq_globalfeatures(seq_id, self.seqs_info, self.model, y_imposter, seg_other_symbol)
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
        
    def save_model(self, file_name):
        # to clean things before pickling the model
        self.seqs_info.clear() 
        ReaderWriter.dump_data(self, file_name)
    
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

            ReaderWriter.log_progress(line,out_file)  
            

    def viterbi(self, w, seq_id):
        l = ("globalfeatures", "activefeatures_by_position", "f_potential")
        self.check_cached_info(w, seq_id, l)
        f_potential = self.seqs_info[seq_id]["f_potential"]
        print("f_potential \n {}".format(f_potential))
        f_transition = self.model.f_transition
        P_codebook = self.model.P_codebook
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L
        # records max score at every time step
        delta = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        # the score for the empty sequence at time 0 is 1
        delta[0,P_codebook[""]] = 0
        back_track = {}
        for j in range(1, T+1):
            for pi in f_transition:
                max_val = -numpy.inf
                for d in range(L):
                    u = j - d
                    v = j
                    if(u <= 0):
                        break
                    boundary = (u, v)
                    for pky, pk in f_transition[pi].items():
                        pk_code = P_codebook[pk]
                        potential = f_potential[boundary][pky]
                        score = potential + delta[u-1, pk_code]
                        if(score > max_val):
                            max_val = score
                            back_track[(j,P_codebook[pi])] = (d, pk, pky[-1])
                            
                delta[j, P_codebook[pi]] = max_val
        print("delta {}".format(delta))
        print("backtrack {}".format(back_track)) 
        # decoding the sequence
        p_T_code = numpy.argmax(delta[T,:])
        d, p_T, y_T = back_track[(T, p_T_code)]
        Y_decoded = []
        for _ in range(d+1):
            Y_decoded.append((p_T,y_T))
        t = T - d - 1
        while t>0:
            print("t {}".format(t))
            p_tplus1 = Y_decoded[-1][0]
            print("p_tplus1 {}".format(p_tplus1))
            print("p_tplus1 coded {}".format(P_codebook[p_tplus1]))
            d, p_t, y_t = back_track[(t, P_codebook[p_tplus1])]
            for _ in range(d+1):
                Y_decoded.append((p_t, y_t))
            t = t-d-1
        Y_decoded.reverse()

        print("decoding sequence with id {} \n".format(seq_id))
        print("Y_decoded {}".format(Y_decoded))
        self.clear_cached_info([seq_id])
        Y_decoded = [yt for (pt,yt) in Y_decoded]
        print("Y_decoded {}".format(Y_decoded))
        return(Y_decoded)


    def check_gradient(self, w, seq_id):
        """ implementation of finite difference method similar to scipy.optimize.check_grad()
        """
        modelfeatures_codebook_rev = {code:feature for feature, code in self.model.modelfeatures_codebook.items()}

        self.clear_cached_info([seq_id])
        epsilon = 1e-9
        # basis vector
        ei = numpy.zeros(len(w), dtype="longdouble")
        grad = numpy.zeros(len(w), dtype="longdouble")
        for i in range(len(w)):
            print("weight index is {}".format(i))
            print("feature name is {}".format(modelfeatures_codebook_rev[i]))
            ei[i] = epsilon
            l_wplus = self.compute_seq_loglikelihood(w + ei, seq_id)
            print("l_wplus loglikelihood {}".format(l_wplus))
            self.clear_cached_info([seq_id])
            l = self.compute_seqs_loglikelihood(w, [seq_id])
            print("l loglikelihood {}".format(l))
            self.clear_cached_info([seq_id])
            grad[i] = (l_wplus - l) / epsilon
            print("grad is {}".format(grad[i]))
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
        
#     def validate_forward_backward_pass(self, w, seq_id):
#         self.clear_cached_info([seq_id])
#         # this will compute alpha and beta matrices and save them in seqs_info dict
#         l = ("activefeatures_by_position", "f_potential", "alpha", "b_potential", "beta")
#         self.check_cached_info(w, seq_id, l)
#         states_code = [code for code in self.model.Y_codebook.values()]
#         alpha = self.seqs_info[seq_id]["alpha"]
#         beta = self.seqs_info[seq_id]["beta"]
#         Z_alpha = vectorized_logsumexp(alpha[-1,:])
#         raw_diff = numpy.abs(Z_alpha - beta[1, :])
#         print("alpha[-1,:] = {}".format(alpha[-1,:]))
#         print("beta[1,:] = {}".format(beta[1,:]))
#         print("Z_alpha : {}".format(Z_alpha))
#         print("beta of states {}".format(beta[1, states_code]))
#         print("Z_aplha - beta [1,:] {}".format(raw_diff))
# 
#         min_raw_diff = numpy.min(raw_diff)
#         print("min_raw_diff  {}".format(min_raw_diff))
#         argmin_raw_diff = numpy.argmin(raw_diff)
#         print("argmin_raw_diff  {}".format(argmin_raw_diff))
#         rel_diff = min_raw_diff/(Z_alpha + beta[1, argmin_raw_diff])
#         print("rel_diff : {}".format(rel_diff))
#         self.clear_cached_info([seq_id])
#         
#         return((min_raw_diff, rel_diff))
    
    def validate_expected_featuresum(self, w, seqs_id):
        self.clear_cached_info(seqs_id)
        grad = self.compute_seqs_gradient(w, seqs_id)
        avg_diff = numpy.mean(grad)
        print("difference between empirical feature sum and model's expected feature sum: \n {}".format(grad))
        print("average difference is {}".format(avg_diff))
        self.clear_cached_info(seqs_id)
        return(avg_diff)