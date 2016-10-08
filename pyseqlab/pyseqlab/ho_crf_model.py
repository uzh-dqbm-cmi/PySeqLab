'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''

import os
from copy import deepcopy
import numpy
from utilities import ReaderWriter, create_directory, vectorized_logsumexp
 
class HOCRFModelRepresentation(object):
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
        
        self.pi_pky_z = self.map_pky_z()
        self.si_ysk_z = self.map_sky_z()
        
        # useful dictionary regarding length of elements
        self.z_lendict = self.get_len_z()
        self.pi_lendict = self.get_len_pi()
        self.si_lendict = self.get_len_si()
        
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
        Z_codebook = self.Z_codebook
        S = {}
        for z_patt in Z_codebook:
            elems = z_patt.split("|")
            #print("z_patt")
            for i in range(1, len(elems)):
                S["|".join(elems[i:])] = 1
                #print("i = {}".format(i))
                #print("suffix {}".format("|".join(elems[i:])))
        for y in Y_codebook:
            S[y] = 1
        # empty element         
        S[""] = 1
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
                            
        pk_y_suffix = self.keep_longest_elems(pk_y_suffix)
        f_transition = {}
        
        for (pk, y), pi in pk_y_suffix.items():
            if(pk == ""):
                elmkey = y
            else:
                elmkey = pk + "|" + y
            if(pi in f_transition):
                f_transition[pi][elmkey] = (pk, y)
            else:
                f_transition[pi] = {elmkey:(pk, y)}
        return(f_transition)
                    
    def get_backward_transitions(self):
        Y_codebook = self.Y_codebook
        S_codebook = self.S_codebook
        sk_y = {}
        
        for s in S_codebook:
            for y in Y_codebook:
                sk_y[(s, y)] = 1
        
        sk_y_prefix = {}
        for s in S_codebook:
#             if(s != ""):
            for (sk, y) in sk_y:
                ref_str = y + "|" + sk
                check = self.check_prefix(s, ref_str)
                if(check):
                    if((sk, y) in sk_y_prefix):
                        sk_y_prefix[(sk, y)].append(s)
                    else:
                        sk_y_prefix[(sk, y)] = [s]
                            
        sk_y_prefix = self.keep_longest_elems(sk_y_prefix)
        b_transition = {}
        for (sk, y), si in sk_y_prefix.items():
            if(sk == ""):
                elmkey = y
            else:
                elmkey = y + "|" + sk
            if(si in b_transition):
                b_transition[si][elmkey] = sk
            else:
                b_transition[si] = {elmkey:sk}
        return(b_transition)    
        
    def map_pky_z(self):
        f_transition = self.f_transition
        Z_codebook = self.Z_codebook
        pi_pky_z = {}
        for pi in f_transition:
            pky_z_map = {}
            for pky in f_transition[pi]:
                l = []
                for z in Z_codebook:
                    if(self.check_suffix(z, pky)):
                        l.append(z)
                pky_z_map[pky] = l
                
            pi_pky_z[pi] = pky_z_map
        return(pi_pky_z)
    
    
    def map_sky_z(self):
        b_transition = self.b_transition
        Z_codebook = self.Z_codebook
        si_ysk_z = {}
        for si in b_transition:
            ysk_z_map = {}
            for ysk in b_transition[si]:
                l = []
                for z in Z_codebook:
                    if(self.check_prefix(z, ysk)):
                        l.append(z)
                ysk_z_map[ysk] = l
            si_ysk_z[si] = ysk_z_map
        #print("b_transition {}".format(b_transition))
        #print("si_ysk_z {}".format(si_ysk_z))
        return(si_ysk_z)  
    
    def get_len_pi(self): 
        P_codebook = self.P_codebook
        pi_lendict = {}
        for pi in P_codebook:
            if(pi == ""):
                pi_lendict[pi] = 0
            else:
                pi_lendict[pi] = len(pi.split("|"))
        return(pi_lendict)
    
    def get_len_si(self): 
        S_codebook = self.S_codebook
        si_lendict = {}
        for si in S_codebook:
            if(si == ""):
                si_lendict[si] = 0
            else:
                si_lendict[si] = len(si.split("|"))
        return(si_lendict)
    
    def get_len_z(self):
        Z_codebook = self.Z_codebook
        z_lendict = {}
        for z in Z_codebook:
            z_lendict[z] = len(z.split("|"))
        return(z_lendict)
        
    def keep_longest_elems(self, s):
        """ used to figure out longest suffix and prefix on sets """
        longest_elems = {}
        for tup, l in s.items():
            longest_elems[tup] = max(l, key = len)
        return(longest_elems)
                  
    def check_suffix(self, token, ref_str):
        return(ref_str.endswith(token))
    
    def check_prefix(self, token, ref_str):
        return(ref_str.startswith(token))
    
    def get_num_features(self):
        return(len(self.modelfeatures_codebook))
    def get_num_states(self):
        return(len(self.Y_codebook)) 
               
class HOCRF(object):
    def __init__(self, model, seqs_representer, seqs_info, loadinfo_fromdisk = True):
        self.model = model
        self.weights = numpy.zeros(model.num_features, dtype= "longdouble")
        self.seqs_representer = seqs_representer
        self.seqs_info = seqs_info
        self.load_fromdisk = loadinfo_fromdisk
        self.func_dict = {"alpha": self._load_alpha,
                         "beta": self._load_beta,
                         "f_potential": self._load_fpotential,
                         "b_potential": self._load_bpotential,
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
    
    def prepare_f_potentialfeatures(self, seq_id):
        pi_pky_z = self.model.pi_pky_z
        T = self.seqs_info[seq_id]["T"]
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
        cached_pf = self.seqs_info[seq_id]["cached_pf"]
        f_potential_features = {}
         
        for j in range(1, T+1):
            boundary = (j, j)
            for pi in pi_pky_z:
                for pky in pi_pky_z[pi]:
                    f_potential_features[j, pi, pky] = []
                    for z_patt in pi_pky_z[pi][pky]:
                        if(z_patt in activefeatures[boundary]):
                            if((j, z_patt) not in cached_pf):
                                f_val = list(activefeatures[boundary][z_patt].values())
                                w_indx = list(activefeatures[boundary][z_patt].keys())
                                cached_pf[j, z_patt] = {'f_val':f_val,'w_indx':w_indx}
                            if((j, z_patt) not in f_potential_features[j, pi, pky]):
                                f_potential_features[j, pi, pky].append((j, z_patt)) 
                    
#                     if(not f_potential_features[j, pi, pky]):
#                         f_potential_features[j, pi, pky] = None
         
        # write on disk
        target_dir = self.seqs_info[seq_id]['activefeatures_dir']
        ReaderWriter.dump_data(f_potential_features, os.path.join(target_dir, "f_potential_features"))
        self.seqs_info[seq_id]['f_potential_features'] = f_potential_features

    def compute_f_potential(self, w, seq_id):
        f_potential_features = self.seqs_info[seq_id]['f_potential_features']
        cached_pf = self.seqs_info[seq_id]["cached_pf"]
        cached_comp = self.seqs_info[seq_id]["cached_comp"]
        f_potential = {}

        
        for j, pi, pky in f_potential_features:
            potential = 0
            for j, z_patt in f_potential_features[j, pi, pky]:
                if((j, z_patt) not in cached_comp):
                    w_indx = cached_pf[j, z_patt]['w_indx']
                    f_val = cached_pf[j, z_patt]['f_val']
                    cached_comp[j, z_patt] = numpy.inner(w[w_indx], f_val)
                potential += cached_comp[j, z_patt]
            f_potential[j, pky] = potential

        return(f_potential)
    
    def compute_forward_vec(self, seq_id):
        f_potential = self.seqs_info[seq_id]["f_potential"]
        f_transition = self.model.f_transition
        P_codebook = self.model.P_codebook
        T = self.seqs_info[seq_id]["T"]
        alpha = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        alpha[0,P_codebook[""]] = 0
         
        for j in range(1, T+1):
            for pi in f_transition:
                accumulator = -numpy.inf
                for pky, (pk, _) in f_transition[pi].items():
                    pk_code = P_codebook[pk]
                    potential = f_potential[j, pky]
                    accumulator = numpy.logaddexp(accumulator, potential + alpha[j-1, pk_code])
                alpha[j, P_codebook[pi]] = accumulator 
                 
        return(alpha)

    def prepare_b_potentialfeatures(self, seq_id):
        si_ysk_z = self.model.si_ysk_z
        z_lendict = self.model.z_lendict
        T = self.seqs_info[seq_id]["T"]
        activefeatures = self.seqs_info[seq_id]["activefeatures_by_position"]
        b_potential_features = {}
        cached_pf = self.cached_pf
        
        for j in range(1, T+1):
            for si in si_ysk_z:
                for ysk in si_ysk_z[si]:
                    b_potential_features[j, si, ysk] = []
                    for z_patt in si_ysk_z[si][ysk]:
                        b = j + z_lendict[z_patt] - 1
                        upd_boundary = (b, b)
                        if(upd_boundary in activefeatures):
                            if(z_patt in activefeatures[upd_boundary]):
                                if((b, z_patt) not in cached_pf):
                                    f_val = list(activefeatures[upd_boundary][z_patt].values())
                                    w_indx = list(activefeatures[upd_boundary][z_patt].keys())
                                    cached_pf[b, z_patt] = {'f_val':f_val, 'w_indx':w_indx}
                                if((b, z_patt) not in b_potential_features[j, si, ysk]):
                                    b_potential_features[j, si, ysk].append((b, z_patt)) 

        # write on disk  
        target_dir = self.seqs_info[seq_id]['activefeatures_dir']
        ReaderWriter.dump_data(b_potential_features, os.path.join(target_dir, "b_potential_features"))
        self.seqs_info[seq_id]['b_potential_features'] = b_potential_features
    
    def compute_b_potential(self, w, seq_id):
        b_potential_features = self.seqs_info[seq_id]['b_potential_features']
        b_potential= {}
        cached_comp = self.seq_info[seq_id]["cached_comp"]
        cached_pf = self.cached_pf
        
        for j, si, ysk in b_potential_features:
            potential = 0
            for b, z_patt in b_potential_features[j, si, ysk]:
                if((b, z_patt) not in cached_comp):
                    w_indx = cached_pf[b, z_patt]['w_indx']
                    f_val = cached_pf[b, z_patt]['f_val']
                    cached_comp[b, z_patt] = numpy.dot(w[w_indx], f_val)
                potential += cached_comp[b, z_patt]

            b_potential[j, ysk] = potential       
            
        return(b_potential)
    
    def compute_backward_vec(self, seq_id):
        b_potential = self.seqs_info[seq_id]["b_potential"]
        b_transition = self.model.b_transition
        S_codebook = self.model.S_codebook
        T = self.seqs_info[seq_id]["T"]
        beta = numpy.ones((T+2,len(S_codebook)), dtype='longdouble') * (-numpy.inf)

        beta[T+1, S_codebook[""]] = 0
        
        for j in reversed(range(1, T+1)):
            for si in b_transition:
                accumulator = -numpy.inf
                for ysk, sk in b_transition[si].items():
                    sk_code = S_codebook[sk]
                    potential = b_potential[j, ysk]
                    accumulator = numpy.logaddexp(accumulator, potential + beta[j+1, sk_code])
                beta[j, S_codebook[si]] = accumulator 
                    
        return(beta)
    
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
    
    def compute_seqs_loglikelihood(self, w, seqs_id):
        seqs_loglikelihood = 0
        for seq_id in seqs_id:
            seqs_loglikelihood += self.compute_seq_loglikelihood(w, seq_id)
        return(seqs_loglikelihood)
    
    def _load_alpha(self, w, seq_id):
        # assumes the f_potential has been loaded into seq_info
        seq_info = self.seqs_info[seq_id]
        seq_info["alpha"] = self.compute_forward_vec(seq_id)
        seq_info["Z"] = vectorized_logsumexp(seq_info["alpha"][-1,:])
#                     print("... Computing alpha probability ...")
    def _load_beta(self, w, seq_id):
        # assumes the b_potential has been loaded into seq_info
        seq_info = self.seqs_info[seq_id]
        seq_info["beta"] = self.compute_backward_vec(seq_id)
#                     print("... Computing beta probability ...")

    def _load_fpotential(self, w, seq_id):
        # assumes the activefeatures_by_position matrix has been loaded into seq_info
        # load f_potential_features
        seq_info = self.seqs_info[seq_id]
        seq_info["cached_comp"] = {}
        if(seq_info.get("f_potential_features") == "on_disk"):
            target_dir = seq_info["activefeatures_dir"]
            f_potential_features = ReaderWriter.read_data(os.path.join(target_dir, "f_potential_features"))
#             f_potential = ReaderWriter.read_data(os.path.join(target_dir, "f_potential"))
            seq_info["f_potential_features"] = f_potential_features
#             seq_info["f_potential"] = f_potential
            print("loading f_potential_features from disk")
        elif(seq_info.get("f_potential_features") == None):
            seq_info["cached_pf"] = {}
            self.prepare_f_potentialfeatures(seq_id)
            print("preparing f_potential_features")

        seq_info["f_potential"] = self.compute_f_potential(w, seq_id)
#                     print("... Computing f_potential ...")

    def _load_bpotential(self, w, seq_id):
        # assumes the activefeatures_by_position matrix has been loaded into seq_info
        seq_info = self.seqs_info[seq_id]
        if(seq_info.get("b_potential_features") == "on_disk"):
            target_dir = seq_info["activefeatures_dir"]
            b_potential_features = ReaderWriter.read_data(os.path.join(target_dir, "b_potential_features"))
            seq_info["b_potential_features"] = b_potential_features
            print("loading b_potential_features from disk")

        elif(seq_info.get("b_potential_features") == None):
            self.prepare_b_potentialfeatures(seq_id)
            print("preparing b_potential_features")

        seq_info["b_potential"] = self.compute_b_potential(w, seq_id)
#                     print("... Computing b_potential ...")

    def _load_flaty(self, w, seq_id):
        seq = self._load_seq(seq_id, target="seq")
        self.seqs_info[seq_id]['flat_y'] = seq.flat_y

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
        
    def check_cached_info(self, w, seq_id, entity_names):
        """ order of args is very important
            to compute 1 - alpha matrix we need to compute f_potential before
                       2 - beta matrix we need to compute the b_potential before
            
        """
        seq_info = self.seqs_info[seq_id]
        func_dict = self.func_dict
        none_type = type(None) 
        for varname in entity_names:
            if(type(seq_info.get(varname)) == none_type):
                func_dict[varname](w, seq_id)

    def clear_cached_info(self, seqs_id, cached_entities = []):
        default_entitites = ["f_potential", "alpha", "Z", "b_potential", "beta", "cached_comp"]
        args = default_entitites  + cached_entities
        for seq_id in seqs_id:
            seq_info = self.seqs_info[seq_id]
            for varname in args:
                if(varname in seq_info):
                    seq_info[varname] = None
        
        if(self.load_fromdisk):
            args = ("f_potential_features", "b_potential_features")
            for seq_id in seqs_id:
                seq_info = self.seqs_info[seq_id]
                for varname in args:
                    if(varname in seq_info):
                        seq_info[varname] = "on_disk"
        
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
        f_transition = self.model.f_transition
        P_codebook = self.model.P_codebook
        T = self.seqs_info[seq_id]["T"]
        # records max score at every time step
        delta = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        # the score for the empty sequence at time 0 is 1
        delta[0,P_codebook[""]] = 0
        back_track = {}
        for j in range(1, T+1):
            for pi in f_transition:
                max_val = -numpy.inf
                for pky, (pk,y) in f_transition[pi].items():
                    pk_code = P_codebook[pk]
                    potential = f_potential[j, pky]
                    score = potential + delta[j-1, pk_code]
                    if(score > max_val):
                        max_val = score
                        back_track[(j,P_codebook[pi])] = (pk, y)
                            
                delta[j, P_codebook[pi]] = max_val
        # decoding the sequence
        p_T_code = numpy.argmax(delta[T,:])
        p_T, y_T = back_track[(T, p_T_code)]
        Y_decoded = []
      
        Y_decoded.append((p_T,y_T))
        t = T - 1
        while t>0:
            p_tplus1 = Y_decoded[-1][0]
            p_t, y_t = back_track[(t, P_codebook[p_tplus1])]
            Y_decoded.append((p_t, y_t))
            t -= 1
        Y_decoded.reverse()

        self.clear_cached_info([seq_id])
        Y_decoded = [yt for (pt,yt) in Y_decoded]
        #print("Y_decoded {}".format(Y_decoded))
        return(Y_decoded)
        
 
    def validate_forward_backward_pass(self, w, seq_id):
        self.clear_cached_info([seq_id])
        # this will compute alpha and beta matrices and save them in seqs_info dict
        l = ("activefeatures_by_position", "f_potential", "alpha", "b_potential", "beta")
        self.check_cached_info(w, seq_id, l)
        alpha = self.seqs_info[seq_id]["alpha"]
        beta = self.seqs_info[seq_id]["beta"]
        print("states codebook {}".format(self.model.Y_codebook))
        print("alpha {}".format(alpha))
        print("beta {}".format(beta))
        
        Z_alpha = vectorized_logsumexp(alpha[-1,:])
        Z_beta = vectorized_logsumexp(beta[1, :])
        raw_diff = numpy.abs(Z_alpha - Z_beta)
        print("alpha[-1,:] = {}".format(alpha[-1,:]))
        print("beta[0,:] = {}".format(beta[0,:]))
        print("beta[1,:] = {}".format(beta[1,:]))
        print("Z_alpha : {}".format(Z_alpha))
        print("Z_beta : {}".format(Z_beta))
        print("Z_aplha - Z_beta {}".format(raw_diff))
 
        rel_diff = raw_diff/(Z_alpha + Z_beta)
        print("rel_diff : {}".format(rel_diff))
        self.clear_cached_info([seq_id])
        print("seqs_info {}".format(self.seqs_info))
        return((raw_diff, rel_diff))   
    
