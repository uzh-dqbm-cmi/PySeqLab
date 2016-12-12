'''
@author: ahmed allam <ahmed.allam@yale.edu>

'''

import os
from copy import deepcopy
import numpy
from .utilities import ReaderWriter, create_directory, vectorized_logsumexp

class HOSemiCRFModelRepresentation(object):
    def __init__(self):
        """ HOSemi CRF model representation
        """         
        self.modelfeatures = None
        self.modelfeatures_codebook = None
        self.Y_codebook = None
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
        self.P_codebook = None
        self.P_codebook_rev = None
        self.pi_lendict = None
        self.S_codebook = None
        self.si_lendict = None
        self.si_numchar = None
        self.f_transition = None
        self.b_transition = None
        self.pky_codebook = None
        self.pky_codebook_rev = None
        self.pi_pky_codebook = None
        self.siy_codebook = None        
        self.pky_z = None
        self.siy_z = None
        self.num_features = None
        self.num_states = None
        
    def create_model(self, modelfeatures, states, L):
        self.modelfeatures = modelfeatures
        self.modelfeatures_codebook = self.get_modelfeatures_codebook()
        self.Y_codebook = self.get_modelstates_codebook(states)
        self.L = L
        self.generate_instance_properties()
        
    def generate_instance_properties(self):
        self.Z_codebook = self.get_Z_pattern()
        self.Z_lendict, self.Z_elems, self.Z_numchar = self.get_Z_info()
        self.patts_len = set(self.Z_lendict.values())
        self.max_patt_len = max(self.patts_len)

        self.modelfeatures_inverted, self.ypatt_features = self.get_inverted_modelfeatures()
        self.ypatt_activestates = self.find_activated_states(self.ypatt_features, self.patts_len)
        
        self.P_codebook = self.get_forward_states()
        self.P_codebook_rev = self.get_P_codebook_rev()
        self.P_len = len(self.P_codebook)
        self.pi_lendict, self.pi_elems, self.pi_numchar = self.get_pi_info()
        
        
        self.S_codebook = self.get_backward_states()
        self.si_lendict, self.si_elems, self.si_numchar = self.get_si_info()
        
        self.f_transition = self.get_forward_transition()
        self.pky_z = self.map_pky_z()

        self.siy_codebook, self.siy_numchar, self.siy_components = self.get_siy_info()
        self.b_transition = self.get_backward_transitions()
        self.siy_z = self.map_siy_z()
        
        self.pi_pky_codebook = self.get_pi_pky_codebook()
        self.si_siy_codebook = self.get_si_siy_codebook()        
        self.num_features = self.get_num_features()
        self.num_states = self.get_num_states()   
    
    
    def get_modelfeatures_codebook(self):
        """flatten model features into a codebook (i.e. each feature is assigned a unique code/number)"""
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
        """generate Y codebook (i.e. assign each state a unique code)"""
        return({s:i for (i, s) in enumerate(states)})
        
    def get_Z_pattern(self):
        """ get y patterns (called z patterns) detected in the training dataset"""
        modelfeatures = self.modelfeatures
        Z_codebook = {y_patt:index for index, y_patt in enumerate(modelfeatures)}
        return(Z_codebook)
    
    def get_Z_info(self):
        """ generates information about z patterns """
        Z_codebook = self.Z_codebook
        Z_lendict = {}
        Z_elems = {}
        Z_numchar = {}
        for z in Z_codebook:
            elems = z.split("|")
            Z_elems[z] = elems
            Z_lendict[z] = len(elems)
            Z_numchar[z] = len(z)            
        return(Z_lendict, Z_elems, Z_numchar)
    

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
    
    def get_forward_states(self):
        Y_codebook = self.Y_codebook
        Z_elems = self.Z_elems
        Z_lendict = self.Z_lendict
        P = {}
        for z_patt in Z_elems:
            elems = Z_elems[z_patt]
            z_len = Z_lendict[z_patt]
            for i in range(z_len-1):
                P["|".join(elems[:i+1])] = 1
        for y in Y_codebook:
            P[y] = 1
        # empty element         
        P[""] = 1
        P_codebook = {s:i for (i, s) in enumerate(P)}
        #print("P_codebook ", P_codebook)
        return(P_codebook) 
    
    def get_P_codebook_rev(self):
        P_codebook = self.P_codebook
        P_codebook_rev = {code:pi for pi, code in P_codebook.items()}
        return(P_codebook_rev)
    
    def get_pi_info(self): 
        P_codebook = self.P_codebook
        pi_lendict = {}
        pi_numchar = {}
        pi_elems = {}
        
        for pi in P_codebook:
            elems = pi.split("|")
            pi_elems[pi] = elems 
            if(pi == ""):
                pi_lendict[pi] = 0
                pi_numchar[pi] = 0
                
            else:
                pi_lendict[pi] = len(elems)
                pi_numchar[pi] = len(pi)
        return(pi_lendict, pi_elems, pi_numchar)
    
    def get_backward_states(self):
        """ combining P_codebook with Y_codebook """
        Y_codebook = self.Y_codebook
        P_codebook = self.P_codebook
        S = {}
        
        for p in P_codebook:
            if(p == ""):
                for y in Y_codebook:
                    S[y] = 1
            else:
                for y in Y_codebook:
                    py = p + "|" + y
                    S[py] = 1
        S_codebook = {s:i for (i, s) in enumerate(S)}
        return(S_codebook)
                    
    def get_si_info(self): 
        S_codebook = self.S_codebook
        si_lendict = {}
        si_numchar = {}
        si_elems = {}
        
        for si in S_codebook:
            elems = si.split("|")
            si_elems[si] = elems 
            si_lendict[si] = len(elems)
            si_numchar[si] = len(si)
        return(si_lendict, si_elems, si_numchar)
    
    def get_forward_transition(self):
        Y_codebook = self.Y_codebook
        P_codebook = self.P_codebook
        pi_numchar = self.pi_numchar
        Z_numchar = self.Z_numchar
        
        pk_y= {}
        for p in P_codebook:
            for y in Y_codebook:
                pk_y[(p, y)] = 1

        pk_y_suffix = {}
        for p in P_codebook:
            if(p != ""):
                len_p = pi_numchar[p]
                for (pk, y) in pk_y:
                    ref_str = pk + "|" + y
                    # in case pk is the empty sequence the number of character will be zero
                    len_ref = pi_numchar[pk] + Z_numchar[y] + 1
                    start_pos = len_ref - len_p
                    if(start_pos>=0):
                        # check suffix relation
                        check = ref_str[start_pos:] == p
                        #check = self.check_suffix(p, ref_str)
                        if(check):
                            if((pk, y) in pk_y_suffix):
                                pk_y_suffix[pk, y].append(p)
                            else:
                                pk_y_suffix[pk, y] = [p]
                            
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
        #print("f_transition ", f_transition)
        return(f_transition)                
                    
    def map_pky_z(self):
        f_transition = self.f_transition
        Z_codebook = self.Z_codebook
        # given that we demand to have a unigram label features then Z set will always contain Y elems
        Z_numchar = self.Z_numchar
        pky_codebook = self.S_codebook
        si_numchar = self.si_numchar
        
        z_pky = {}
        for pi in f_transition:
            for pky in f_transition[pi]:
                # get number of characters in the pky 
                len_pky = si_numchar[pky]
                for z in Z_codebook:
                    len_z = Z_numchar[z]
                    # check suffix relation
                    start_pos = len_pky - len_z
                    if(start_pos >= 0):
                        check = pky[start_pos:] == z
                        if(check):
                            pky_c = pky_codebook[pky]
                            if(z in z_pky):
                                z_pky[z].append(pky_c)
                            else:
                                z_pky[z] = [pky_c]
        return(z_pky) 
    
    
    def get_siy_info(self):
        S_codebook = self.S_codebook
        Y_codebook = self.Y_codebook
        Z_numchar = self.Z_numchar
        si_numchar = self.si_numchar
        
        siy_components = {}
        siy_codebook = {}
        siy_numchar = {}
        counter = 0
        for si in S_codebook:
            for y in Y_codebook:
                siy = si + "|" + y
                siy_codebook[siy] = counter
                siy_numchar[siy] = si_numchar[si] + Z_numchar[y] + 1
                siy_components[siy] = (si, y)
                counter += 1        
        return(siy_codebook, siy_numchar, siy_components)
    
    def get_backward_transitions(self):
        S_codebook = self.S_codebook
        si_numchar = self.si_numchar
        si_y_suffix = {}
        siy_components = self.siy_components
        siy_numchar = self.siy_numchar
        
        for sk in S_codebook:
            len_sk = si_numchar[sk] 
            for siy in siy_components:
                len_ref = siy_numchar[siy]
                start_pos = len_ref - len_sk
                if(start_pos >= 0): 
                    # check suffix relation
                    check = siy[start_pos:] == sk
                    #check = self.check_suffix(sk, si + "|" + y)
                    if(check):
                        si_y_tup = siy_components[siy]
                        if(si_y_tup in si_y_suffix):
                            si_y_suffix[si_y_tup].append(sk)
                        else:
                            si_y_suffix[si_y_tup] = [sk]
        
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

                
    def map_siy_z(self):
        b_transition = self.b_transition
        Z_codebook = self.Z_codebook
        # given that we demand to have a unigram label features then Z set will always contain Y elems
        Z_numchar = self.Z_numchar
        siy_codebook = self.siy_codebook
        siy_numchar = self.siy_numchar
        
        z_siy = {}
        for si in b_transition:
            for siy in b_transition[si]:
                # get number of characters in the siy 
                # +1 is for the separator '|'
                len_siy = siy_numchar[siy] 
                for z in Z_codebook:
                    len_z = Z_numchar[z]
                    # check suffix relation
                    start_pos = len_siy - len_z
                    if(start_pos >= 0):
                        check = siy[start_pos:] == z
                        if(check):
                            siy_c = siy_codebook[siy]
                            if(z in z_siy):
                                z_siy[z].append(siy_c)
                            else:
                                z_siy[z] = [siy_c]
        return(z_siy)    

    def get_pi_pky_codebook(self):
        f_transition = self.f_transition
        pky_codebook = self.S_codebook
        P_codebook = self.P_codebook
        
        pi_pky_codebook = {}
        for pi in f_transition:
            pi_pky_codebook[pi]=([],[])
            for pky, (pk, _) in f_transition[pi].items():
                pi_pky_codebook[pi][0].append(pky_codebook[pky])
                pi_pky_codebook[pi][1].append(P_codebook[pk])

        return(pi_pky_codebook)
    
    def get_si_siy_codebook(self):
        b_transition = self.b_transition
        siy_codebook = self.siy_codebook
        S_codebook = self.S_codebook
        
        si_siy_codebook = {}
        for si in b_transition:
            si_siy_codebook[si] = ([],[])
            for siy, sk in b_transition[si].items():
                si_siy_codebook[si][0].append(siy_codebook[siy])
                si_siy_codebook[si][1].append(S_codebook[sk])

        return(si_siy_codebook)
    
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
        # TOUPDATE where accum_active_states support boundary keys 
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

class HOSemiCRF(object):
    def __init__(self, model, seqs_representer, seqs_info, load_info_fromdisk = 4):
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
                         "Y":self._load_Y}
        
        self.def_cached_entities = self.cached_entitites(load_info_fromdisk)
        # default beam size covers all the prefix values (i.e. pi)
        self.beam_size = len(self.model.P_codebook)

    def cached_entitites(self, load_info_fromdisk):
        ondisk_info = ["activefeatures", "l_segfeatures", "seg_features", "activated_states", "globalfeatures_per_boundary", "globalfeatures", "Y"]
        def_cached_entities = ondisk_info[:load_info_fromdisk]
        inmemory_info = ["alpha", "Z", "f_potential", "beta", "b_potential", "P_marginal"]
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
        L = self.model.L
        
        accum_activestates = {}
        activefeatures_perboundary = {}
        for j in range(1, T+1):
            for d in range(L):
                u = j - d
                v = j
                boundary = (u, v)
                # identify active features
                active_features = self.identify_activefeatures(seq_id, boundary, accum_activestates)
                activefeatures_perboundary[boundary] = active_features
        return(activefeatures_perboundary)
    
    def compute_fpotential(self, w, active_features):
        """ compute the potential of active features at a defined boundary """
        model = self.model
        pky_codebook = model.S_codebook
        z_pky = model.pky_z
        f_potential = numpy.zeros(len(pky_codebook))

        # to consider caching the w_indx and fval as in cached_pf
        for z in active_features:
            w_indx = list(active_features[z].keys())
            f_val = list(active_features[z].values())
            potential = numpy.inner(w[w_indx], f_val)
            # get all pky's in coded format where z maintains a suffix relation with them
            pky_c_list = z_pky[z]
            f_potential[pky_c_list] += potential

        return(f_potential)
               
    def compute_forward_vec(self, seq_id):
        f_potential = self.seqs_info[seq_id]["f_potential"]
        pi_pky_codebook = self.mode.pi_pky_codebook
        pi_lendict = self.model.pi_lendict
        P_codebook = self.model.P_codebook
        T = self.seqs_info[seq_id]["T"]
        L = self.model.L
        alpha = numpy.ones((T+1,len(P_codebook)), dtype='longdouble') * (-numpy.inf)
        alpha[0,P_codebook[""]] = 0
         
        for j in range(1, T+1):
            for pi in pi_pky_codebook:
                if(j >= pi_lendict[pi]):
                    accumulator = numpy.ones(L, dtype='longdouble') * -numpy.inf
                    for d in range(L):
                        u = j - d
                        v = j
                        if(u <= 0):
                            break
                        time_diff = v - u
                        vec = f_potential[time_diff, pi_pky_codebook[pi][0]] + alpha[u-1, pi_pky_codebook[pi][1]]
                        accumulator[d] = vectorized_logsumexp(vec)
                    
                    alpha[j, P_codebook[pi]] = vectorized_logsumexp(accumulator) 
                 
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