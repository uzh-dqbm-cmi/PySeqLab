'''

@author: Ahmed Allam <ahmed.allam@yale.edu>
'''

import os
from datetime import datetime
import numpy
from .utilities import ReaderWriter, create_directory, generate_datetime_str

class Learner(object):
    def __init__(self, crf_model):
        self.crf_model = crf_model
        self.training_description = None
    
    def train_model(self, w0, seqs_id, optimization_options, working_dir, save_model = True):
        """ Available options based on the selected method
             optimization_options:
                            1- {'method': SGA-ADADELTA
                              'regularization_type': {'l1', 'l2'}
                               regularization_value: float
                               num_epochs: int
                              'tolerance': float
                              'rho': float
                              'epsilon': float
                              }
                            1- {'method': SGA
                              'regularization_type': {'l1', 'l2'}
                               'regularization_value': float
                               num_epochs: int
                              'tolerance': float
                              'learning_rate_schedule': one of ("bottu", "exponential_decay", "t_inverse", "constant")
                              't0': float
                              'alpha': float
                              'eta0': float
                              }
                               
                            2- {'method': Newton-CG
                              'regularization_type': 'l2'
                               regularization_value: float
                              'disp': False,
                              'xtol': 1e-05,
                              'eps': 1.4901161193847656e-08, 
                              'return_all': False,
                              'maxiter': None,
                              'norm': inf
                              }
                              
                            3- {'method': CG, BFGS
                              'regularization_type': 'l2'
                               regularization_value: float
                              'disp': False,
                              'gtol': 1e-05,
                              'eps': 1.4901161193847656e-08, 
                              'return_all': False,
                              'maxiter': None,
                              'norm': inf
                              }
                              
                            4- {'method': L-BFGS-B
                              'regularization_type': 'l2'
                               regularization_value: float
                              'disp': False
                              'maxls': 20,
                              'iprint': -1,
                              'gtol': 1e-05,
                              'eps': 1e-08, 
                              'maxiter': 15000, 
                              'ftol': 2.220446049250313e-09, 
                              'maxcor': 10, 
                              'maxfun': 15000
                              }
        """

        pop_keys = set()
        
        lambda_type = optimization_options.get("regularization_type")
        pop_keys.add("regularization_type")
        if(lambda_type not in ('l1', 'l2')):
            # default regularization type is l2
            lambda_type = 'l2'
            print("regularization by default is l2")

        # get the regularization parameter value
        lambda_val = optimization_options.get("regularization_value")
        pop_keys.add("regularization_value")
        
        if(lambda_val == None):
            # assign default lambda value
            lambda_val = 0.0
        elif(lambda_val < 0):
            # regularization should be positive
            lambda_val = 0.0

        # initialization of weight vector w node
#         w0 = numpy.zeros(len(self.weights))
        method = optimization_options.get("method")
        pop_keys.add("method")
        if(method not in ("L-BFGS-B", "BFGS", "SGA","SGA-ADADELTA","SVRG","COLLINS-PERCEPTRON")):
            # default weight learning/optimization method
            method = "SGA-ADADELTA"
            
        if(method in ("L-BFGS-B", "BFGS")):
            # initialize the new optimization options
            option_keys = set(optimization_options.keys()) - pop_keys
            options = {elmkey:optimization_options[elmkey] for elmkey in option_keys}
            optimization_config = {'method':method,
                                   'regularization_value':lambda_val,
                                   'regularization_type':'l2',
                                   'options':options
                                   }
            estimate_weights = self._optimize_scipy

            
        elif(method in ("SGA", "SGA-ADADELTA", "SVRG", "COLLINS-PERCEPTRON")):
            num_epochs = optimization_options.get("num_epochs")
            if(type(num_epochs) != int):
                # default number of epochs if not specified
                num_epochs = 3
            elif(num_epochs < 0):
                # num_epochs should be positive
                num_epochs = 3
                
            tolerance = optimization_options.get("tolerance")
            if(tolerance == None):
                # default value of tolerance if not specified
                tolerance = 1e-8
            elif(tolerance < 0):
                tolerance = 1e-8
                
            optimization_config = {'method': method,
                                   'regularization_type':lambda_type,
                                   'regularization_value': lambda_val,
                                   'num_epochs': num_epochs,
                                   'tolerance': tolerance
                                   }
            
            if(method == "COLLINS-PERCEPTRON"):
                estimate_weights = self._structured_perceptron
                # if segmentation problem the non-entity symbol is specified using this option else it is None
                seg_other_symbol = optimization_options.get("seg_other_symbol")
                optimization_config['seg_other_symbol'] = seg_other_symbol
                avg_scheme = optimization_options.get("avg_scheme")
                if(avg_scheme not in ("avg_uniform", "avg_error", "survival")):
                    avg_scheme = "avg_error"
                optimization_config["avg_scheme"] = avg_scheme

            elif(method in ("SGA", "SVRG")):
            
                # get the other parameters to be tuned such as t0 and alpha
                learning_rate_schedule = optimization_options.get("learning_rate_schedule")
                if(learning_rate_schedule not in ("bottu", "exponential_decay", "t_inverse", "constant")):
                    # default learning rate schedule
                    learning_rate_schedule = "t_inverse"
                optimization_config["learning_rate_schedule"] = learning_rate_schedule
                
                t0 = optimization_options.get("t0")
                if(t0 == None):
                    # use default value
                    t0 = 0.1
                elif(t0 < 0):
                    t0 = 0.1
                optimization_config['t0'] = t0
    
                if(learning_rate_schedule in ("t_inverse", "exponential_decay")):
                    # get the alpha parameter
                    a = optimization_options.get("a")
                    if(a == None):
                        # use a default value
                        a = 0.9
                    elif(a <= 0 or a >= 1):
                        a = 0.9
                    optimization_config['a'] = a

                if(method == "SGA"):
                    estimate_weights = self._sga_classic   
                else:
                    estimate_weights = self._sga_svrg     
                                
            elif(method == "SGA-ADADELTA"):
                estimate_weights = self._sga_adadelta
                
                p_rho = optimization_options.get("p_rho")
                if(p_rho == None):
                    # default value
                    p_rho = 0.95
                elif(p_rho < 0):
                    # num_epochs should be positive
                    p_rho = 0.95
                    
                epsilon = optimization_options.get("epsilon")
                if(epsilon == None):
                    # default value of tolerance if not specified
                    epsilon = 1e-6
                elif(epsilon < 0):
                    epsilon = 1e-6
                optimization_config['p_rho'] = p_rho
                optimization_config['epsilon'] = epsilon

        # save the training options
        self.training_description = optimization_config
        model_foldername = generate_datetime_str()
        model_dir = create_directory(model_foldername, create_directory("models", working_dir))
        model_name = model_foldername + ".model"
        self.training_description["model_dir"] = model_dir
        self.training_description["model_name"] = model_name
        self.training_description["train_seqs_id"] = seqs_id
        
        # if everything is defined correctly then estimate the parameters
        w_hat = estimate_weights(w0, seqs_id)
        # update model weights to w_hat
        self.crf_model.weights = w_hat
        
        if(save_model):
            # pickle the model
            self.crf_model.save_model(file_name = os.path.join(model_dir, model_name), seqs_id)
            
        # cleanup the instance variables
        self.cleanup()
   
            
    def _report_training(self):
        method = self.training_description["method"]
        regularization_type = self.training_description["regularization_type"]
        # regularization parameter lambda
        C = self.training_description['regularization_value']
        model_dir = self.training_description["model_dir"]
        model_name = self.training_description["model_name"]
        # log file 
        log_file = os.path.join(model_dir, "crf_training_log.txt")
        line = "---Model training-- starting time {} \n".format(datetime.now())
        line += "model name: {} \n".format(model_name)
        line += "model directory: {} \n".format(model_dir)
        line += "model type: {} \n".format(self.crf_model.__class__)
        line += "training method: {} \n".format(method)
        line += "type of regularization: {} \n".format(regularization_type)
        line += "value of regularization: {} \n".format(C)
        
        if(method  == "SGA"):
            learning_rate_schedule = self.training_description["learning_rate_schedule"]
            t0 = self.training_description["t0"]
            line += "learning rate schedule: {} \n".format(learning_rate_schedule)
            line += "eta0: {} \n".format(t0)
            if(learning_rate_schedule in ("t_inverse", "exponential_decay")):
                # get the alpha parameter
                a = self.training_description["a"]
                line += "a: {} \n".format(a)
        elif(method == "SGA-ADADELTA"):
            rho = self.training_description["p_rho"]
            epsilon = self.training_description["epsilon"]
            line += "p_rho: {} \n".format(rho)
            line += "epsilon: {} \n".format(epsilon)
        elif(method == "COLLINS-PERCEPTRON"):
            avg_scheme = self.training_description["avg_scheme"]
            line += "averaging scheme: {} \n".format(avg_scheme)
        if(method not in ("L-BFGS-B", "BFGS")):
            line += "number of epochs: {} \n".format(self.training_description['num_epochs'])
        # write to file    
        ReaderWriter.log_progress(line, log_file)
        
    def _check_reldiff(self, x, y):
        tolerance = self.training_description["tolerance"]
        if(x != y):            
            reldiff = numpy.abs(x - y) / (numpy.abs(x) + numpy.abs(y))
            print("reldiff = {}".format(reldiff))
            if(reldiff <= tolerance):
                self._exitloop = True
            else:
                self._exitloop = False 
        else:
            self._exitloop = True
            
        #############################
        # optimize using scipy optimize function
        #############################
    def _optscipy_seqs_loglikelihood(self, w, seqs_id):
        """compute seqs loglikelihood when using the BFGS and L-BFGS-B optimization options"""
        crf_model = self.crf_model
        seqs_loglikelihood = crf_model.compute_seqs_loglikelihood(w, seqs_id)
        # clear cached info 
        crf_model.clear_cached_info(seqs_id)
        # check for regularization parameter
        l2 = self.training_description["regularization_value"]
        if(l2>0):
            # log(p(Y|X;w)) - lambda/2 * ||w||**2
            seqs_loglikelihood = seqs_loglikelihood - ((l2/2) * numpy.dot(w, w))
        # since the optimization will be based on minimization, hence we multiply by -1
        seqs_loglikelihood = seqs_loglikelihood * -1
        return(seqs_loglikelihood)
    
    def _optscipy_seqs_gradient(self, w, seqs_id):
        """compute seqs gradient when using the BFGS and L-BFGS-B optimization options"""
        crf_model = self.crf_model
        seqs_grad = crf_model.compute_seqs_gradient(w, seqs_id)
        # clear cached info 
        crf_model.clear_cached_info(seqs_id)
        l2 = self.training_description["regularization_value"]
        if(l2>0):
            seqs_grad = seqs_grad - (l2 * w)
        # since the optimization will be based on minimization, hence we multiply by -1
        seqs_grad = seqs_grad * -1
        return(seqs_grad)
    
  
    def _optimize_scipy(self, w, train_seqs_id):
        """ estimate the parameters w of the model
            it uses optimize.minimize() function from the scipy package
            Params:
            -------
            w: initial weights vector
            options: dictionary that contains the configuration/options to pass for minimize function
        """
        from scipy import optimize
        self._report_training() 
        objfunc = self._optscipy_seqs_loglikelihood
        gradfunc = self._optscipy_seqs_gradient
        method = self.training_description["method"]
        options = self.training_description['options']
  
        # to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._iter_count = 0
        result = optimize.minimize(fun = objfunc,
                                   x0 = w,
                                   args = (train_seqs_id),
                                   method = method,
                                   jac = gradfunc,
                                   options = options,
                                   callback = self._track_scipy_optimizer)
          
        model_dir = self.training_description["model_dir"]
        # log file 
        log_file = os.path.join(model_dir, "crf_training_log.txt")
        line = "---Model training--- end time {} \n".format(datetime.now())
        line += "\n \n"
        ReaderWriter.log_progress(line, log_file)
          
        print("results \n {}".format(result))
          
        # estimated optimal weights
        w_hat = result.x
          
        return(w_hat)
    
    def _track_scipy_optimizer(self, w):
        # increment iteration count
        self._iter_count += 1
        delta_time = datetime.now() - self._elapsed_time 
        crf_model = self.crf_model
        # approximate estimation of sum of loglikelihood -- using previous weights
        train_seqs_id = self.training_description["train_seqs_id"]
        seqs_loglikelihood = 0
        for seq_id in train_seqs_id:
            seq_loglikelihood = crf_model.seqs_info[seq_id]["loglikelihood"]
            seqs_loglikelihood += seq_loglikelihood
        seqs_loglikelihood *= -1 
          
        """ use the below command >> to compute the sum of sequences' loglikelihood using the updated/current weights
            the sum should be decreasing after each iteration for successful training (used as diagnostics)
            however it is expensive/costly to recompute;
            >> seqs_loglikelihood = self._optscipy_seqs_loglikelihood(w, train_seqs_id)
        """
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")
        line = "--- Iteration {} --- \n".format(self._iter_count)
        line += "Estimated average negative loglikelihood is {} \n".format(seqs_loglikelihood)
        line += "Number of seconds spent: {} \n".format(delta_time.total_seconds())
        ReaderWriter.log_progress(line, log_file)
        self._elapsed_time = datetime.now()


    # needs still some work and fixing....
    def _structured_perceptron(self, w, train_seqs_id):
        """ implements structured perceptron algorithm in particular the average perceptron that was
            introduced by Michael Collins in 2002 (see his paper xx)
        """
        self._report_training()
        num_epochs = self.training_description["num_epochs"]
        # TODO add support for regularization while using structured perceptron as training algorithm
#         regularization_type = self.training_description["regularization_type"]
#         # regularization parameter lambda
#         C = self.training_description['regularization_value']
        seg_other_symbol = self.training_description['seg_other_symbol']
        avg_scheme = self.training_description["avg_scheme"]
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")

        N = len(train_seqs_id)
        crf_model = self.crf_model
        seqs_info = crf_model.seqs_info
        # instance variable to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._exitloop = False
        
        if(avg_scheme == "survival"):
            # accumulated sum of estimated weights
            w_avg = numpy.zeros(len(w), dtype = "longdouble")
            avg_error_list = [0]
            track_seqs = []
            survival_len = 0
            total_survival = 0

            for k in range(num_epochs):
                seq_left = N
                error_count = 0
                numpy.random.shuffle(train_seqs_id)
                for seq_id in train_seqs_id:
#                     print("sequences left {}".format(seq_left))
                    y_imposter = crf_model.viterbi(w, seq_id)

                    if(k == 0):
                        crf_model.check_cached_info(w, seq_id, ("flat_y",))
                    
                    y_original = seqs_info[seq_id]['flat_y']
                    T = seqs_info[seq_id]['T']
#                     print("y original {}".format(y_original))
#                     print("y imposter {}".format(y_imposter))
                    missmatch = [i for i in range(T) if y_original[i] != y_imposter[i]]
                    len_diff = len(missmatch)
                    if(len_diff):
#                         print("miss match with seq_id {}".format(seq_id))
                        if(seq_id not in track_seqs):
                            track_seqs.append(seq_id)
                        if(survival_len):
                            w_avg += survival_len * w
                            total_survival += survival_len
                        survival_len = 0
                        error_count += len_diff/T
                        crf_model.check_cached_info(w, seq_id, ("globalfeatures",))
                        y_original_gfeatures = seqs_info[seq_id]['globalfeatures']
                        # generate global features for the current imposter 
                        y_imposter_gfeatures = crf_model.load_imposter_globalfeatures(seq_id, y_imposter, seg_other_symbol)                     
#                         print("y_imposter gfeatures {}".format(y_imposter_gfeatures))
#                         print("y_original_gfeatures {}".format(y_original_gfeatures))

                        # the contribution of global features when using the correct label sequence
                        w[list(y_original_gfeatures.keys())] += list(y_original_gfeatures.values())
                        
                        # the contribution of global features when using the imposter label sequence
                        w[list(y_imposter_gfeatures.keys())] -= list(y_imposter_gfeatures.values())
                        crf_model.clear_cached_info(track_seqs)
                        track_seqs = []
                    else:
#                         print("nomiss match with seq_id {}".format(seq_id))
                        survival_len += 1
                        if(seq_id not in track_seqs):
                            track_seqs.append(seq_id)
                    seq_left -= 1
#                 print("error count {}".format(error_count))
                avg_error_list.append(float(error_count/N))
                self._track_perceptron_optimizer(w, k, avg_error_list)
                print("average error : {}".format(avg_error_list))
                print("self._exitloop {}".format(self._exitloop))
                print("track_seqs {}".format(track_seqs))
                if(self._exitloop):
                    break
                self._elapsed_time = datetime.now()
            if(survival_len):
                w_avg += survival_len * w
                total_survival += survival_len
                w_avg /= total_survival
                survival_len = 0
            elif(survival_len == 0 and total_survival == 0):
                # in case the previous weights were not surviving (i.e. not capable of decoding at leaset one sequence correctly)
                # return the last weight version
                w_avg = w   
                    
        elif(avg_scheme in ("avg_error", "avg_uniform")):
            # accumulated sum of estimated weights
            w_avg = numpy.zeros(len(w), dtype = "longdouble")
            avg_error_list = [0]
            track_seqs = []
            num_upd = 0
            for k in range(num_epochs):
                seq_left = N
                error_count = 0
                numpy.random.shuffle(train_seqs_id)
                for seq_id in train_seqs_id:
#                     print("sequences left {}".format(seq_left))
                    y_imposter = crf_model.viterbi(w, seq_id)

                    if(k == 0):
                        crf_model.check_cached_info(w, seq_id, ("flat_y",))
                    
                    y_original = seqs_info[seq_id]['flat_y']
                    T = seqs_info[seq_id]['T']
#                     print("y original {}".format(y_original))
#                     print("y imposter {}".format(y_imposter))
                    missmatch = [i for i in range(T) if y_original[i] != y_imposter[i]]
                    len_diff = len(missmatch)
                    if(len_diff):
#                         print("miss match with seq_id {}".format(seq_id))
                        if(seq_id not in track_seqs):
                            track_seqs.append(seq_id)
                            
                        # range of error is [0-1]
                        seq_err_count = len_diff/T
                        error_count += seq_err_count

                        crf_model.check_cached_info(w, seq_id, ("globalfeatures",))
                        y_original_gfeatures = seqs_info[seq_id]['globalfeatures']
                        # generate global features for the current imposter 
                        y_imposter_gfeatures = crf_model.load_imposter_globalfeatures(seq_id, y_imposter, seg_other_symbol)                     
#                         print("y_imposter gfeatures {}".format(y_imposter_gfeatures))
#                         print("y_original_gfeatures {}".format(y_original_gfeatures))

                        # the contribution of global features when using the correct label sequence
                        w[list(y_original_gfeatures.keys())] += list(y_original_gfeatures.values())
                        
                        # the contribution of global features when using the imposter label sequence
                        w[list(y_imposter_gfeatures.keys())] -= list(y_imposter_gfeatures.values())
                        
                        if(avg_scheme == "avg_error"):
                            # consider/weigh more previous weights that have small average error per sequence
                            w_avg += (1-seq_err_count) * w
                        else:
                            w_avg += w
                        crf_model.clear_cached_info(track_seqs)
                        track_seqs = []
                        num_upd += 1
                    else:
#                         print("nomiss match with seq_id {}".format(seq_id))
                        if(seq_id not in track_seqs):
                            track_seqs.append(seq_id)
                    seq_left -= 1
#                 print("error count {}".format(error_count))
                avg_error_list.append(float(error_count/N))
                self._track_perceptron_optimizer(w, k, avg_error_list)
                print("average error : {}".format(avg_error_list))
                print("self._exitloop {}".format(self._exitloop))
                print("track_seqs {}".format(track_seqs)) 
                if(self._exitloop):
                    break
                self._elapsed_time = datetime.now()
            w_avg /= num_upd
            
        line = "---Model training--- end time {} \n".format(datetime.now())
        ReaderWriter.log_progress(line, log_file)
                 
        return(w_avg)      

    def _track_perceptron_optimizer(self, w, k, avg_error_list):
        delta_time = datetime.now() - self._elapsed_time 
        self._check_reldiff(avg_error_list[-2], avg_error_list[-1])
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")
        line = "--- Iteration {} --- \n".format(k+1)
        line += "Average percentage of decoding error: {} \n".format(avg_error_list[-1]*100)
        line += "Number of seconds spent: {} \n".format(delta_time.total_seconds())
        ReaderWriter.log_progress(line, log_file)
        # dump the learned weights for every pass
#         self.dump_file(w, os.path.join(model_dir, "model_weights_epoch_{}".format(k+1)))

        
    def _sga_adadelta(self, w, train_seqs_id):
        self._report_training()
        crf_model = self.crf_model
        num_epochs = self.training_description["num_epochs"]
        regularization_type = self.training_description["regularization_type"]
        # regularization parameter lambda
        C = self.training_description['regularization_value']
        # number of training sequences
        N = len(train_seqs_id)
         
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")

        # keeps track of the log-likelihood of a sequence before weight updating
        seqs_loglikelihood_vec = numpy.zeros(N)
        seqs_id_mapper = {seq_id:unique_id for unique_id, seq_id in enumerate(train_seqs_id)}
        # step size decides the number of data points to average in the seqs_loglikelihood_vec
        # using 10% of data points
        step_size = round(N * 0.1)
        if step_size == 0:
            step_size = 1
        mean_cost_vec = [0]
        
        p_rho = self.training_description["p_rho"]
        epsilon = self.training_description["epsilon"]
        E_g2 = numpy.zeros(len(w), dtype="longdouble")
        E_deltaw2 = numpy.zeros(len(w), dtype="longdouble")
        if(regularization_type == "l1"):
            u = 0
            q = numpy.zeros(len(w), dtype = "longdouble")
        
        # instance variable to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._exitloop = False
                
        for k in range(num_epochs):
            # shuffle sequences at the beginning of each epoch 
            numpy.random.shuffle(train_seqs_id)
            numseqs_left = N
    
            for seq_id in train_seqs_id:
#                     print(seq_id)
                    
                seq_loglikelihood = crf_model.compute_seq_loglikelihood(w, seq_id)
                seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood
                seq_grad = crf_model.compute_seq_gradient(w, seq_id)
                windx = list(seq_grad.keys())
                fval = list(seq_grad.values())
                
                if(C):
                    grad = numpy.zeros(len(w), dtype = "longdouble")
                    grad[windx] = fval
                    if(regularization_type == 'l2'):
                        seq_loglikelihood += - ((C/N) * (1/2) * numpy.dot(w, w))
                        grad -= ((C/N)* w)
                        
                    elif(regularization_type == 'l1'):
                        seq_loglikelihood += - (C/N) * numpy.sum(numpy.abs(w))
                        
                    # update the computed sequence loglikelihood by adding the regularization term contribution   
                    seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood

                    # accumulate gradient
                    E_g2 = p_rho * E_g2 + (1-p_rho) * numpy.square(grad) 
                    RMS_g = numpy.sqrt(E_g2 + epsilon)
                    RMS_deltaw = numpy.sqrt(E_deltaw2 + epsilon)
                    ratio = (RMS_deltaw/RMS_g)
                    deltaw =  ratio * grad
                    E_deltaw2 = p_rho * E_deltaw2 + (1-p_rho) * numpy.square(deltaw)                    
                    w += deltaw
                    if(regularization_type == "l1"):
                        u += ratio * (C/N)
                        w_upd, q_upd = self._apply_l1_penalty(w, q, u, windx)
                        w = w_upd
                        q = q_upd
                else:

                    # accumulate gradient
                    E_g2 = p_rho * E_g2
                    E_g2[windx] += (1-p_rho) * numpy.square(fval)
                    RMS_g = numpy.sqrt(E_g2 + epsilon)
                    RMS_deltaw = numpy.sqrt(E_deltaw2 + epsilon)
                    ratio = (RMS_deltaw/RMS_g)
                    deltaw = ratio[windx] * fval
                    E_deltaw2 = p_rho * E_deltaw2 
                    E_deltaw2[windx] += (1-p_rho) * numpy.square(deltaw)                    
                    w[windx] += deltaw
                

                # clean cached info
                crf_model.clear_cached_info([seq_id])
                numseqs_left -= 1
                print("num seqs left: {}".format(numseqs_left))
            
            seqs_cost_vec = [numpy.mean(seqs_loglikelihood_vec[i:i+step_size]) for i in range(0, N, step_size)]
            # to consider plotting this vector
            mean_cost_vec.append(numpy.mean(seqs_loglikelihood_vec))
            self._track_sga_optimizer(w, seqs_cost_vec, mean_cost_vec, k)
            if(self._exitloop):
                break
            self._elapsed_time = datetime.now()

            
        line = "---Model training--- end time {} \n".format(datetime.now())
        ReaderWriter.log_progress(line, log_file)
                
        return(w)  
    
    def _sga_classic(self, w, train_seqs_id):
        self._report_training()
        crf_model = self.crf_model
        num_epochs = self.training_description["num_epochs"]
        regularization_type = self.training_description["regularization_type"]
        # regularization parameter lambda
        C = self.training_description['regularization_value']
        # number of training sequences
        N = len(train_seqs_id)
         
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")

        # keeps track of the log-likelihood of a sequence before weight updating
        seqs_loglikelihood_vec = numpy.zeros(N)
        seqs_id_mapper = {seq_id:unique_id for unique_id, seq_id in enumerate(train_seqs_id)}
        # step size decides the number of data points to average in the seqs_loglikelihood_vec
        # using 10% of data points
        step_size = round(N * 0.1)
        if step_size == 0:
            step_size = 1
        mean_cost_vec = [0]
        
        # instance variable to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._exitloop = False
        
        if(regularization_type == "l1"):
            u = 0
            q = numpy.zeros(len(w), dtype = "longdouble")
               
        learning_rate_schedule = self.training_description["learning_rate_schedule"]
        t0 = self.training_description["t0"]
        # 0<a<1 -- a parameter should be between 0 and 1 exclusively
        a = self.training_description["a"]
        t = 0
        
        for k in range(num_epochs):
            # shuffle sequences at the beginning of each epoch 
            numpy.random.shuffle(train_seqs_id)
            numseqs_left = N
            
            for seq_id in train_seqs_id:
                # compute/update learning rate
                if(learning_rate_schedule == "bottu"):
                    eta = C/(t0 + t)
                elif(learning_rate_schedule == "exponential_decay"):
                    eta = t0*a**(t/N)
                elif(learning_rate_schedule == "t_inverse"):
                    eta = t0/(1 + a*(t/N))
                elif(learning_rate_schedule == "constant"):
                    eta = t0
                    
                print("eta {}".format(eta))
                print(seq_id)
                
                seq_loglikelihood = crf_model.compute_seq_loglikelihood(w, seq_id)
                seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood
                seq_grad = crf_model.compute_seq_gradient(w, seq_id)
                print("seq_grad {}".format(seq_grad))
                windx = list(seq_grad.keys())
                fval = list(seq_grad.values())
                if(C):
                    grad = numpy.zeros(len(w), dtype = "longdouble")
                    grad[windx] = fval
                    if(regularization_type == 'l2'):
                        seq_loglikelihood += - ((C/N) * (1/2) * numpy.dot(w, w))
                        grad -= ((C/N)* w)
                        w += eta * grad
                        
                    elif(regularization_type == 'l1'):
                        seq_loglikelihood += - (C/N) * numpy.sum(numpy.abs(w))
                        u += eta * (C/N)
                        w_upd, q_upd = self._apply_l1_penalty(w, q, u, windx)
                        w = w_upd
                        q = q_upd
                        
                    # update the computed sequence loglikelihood by adding the regularization term contribution   
                    seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood

                else:                   
                    print("fval {}".format(fval)) 
                    w[windx] += numpy.multiply(eta, fval)
                    
                t += 1
                # clean cached info
                crf_model.clear_cached_info([seq_id])
                numseqs_left -= 1
                print("num seqs left: {}".format(numseqs_left))
                
            seqs_cost_vec = [numpy.mean(seqs_loglikelihood_vec[i:i+step_size]) for i in range(0, N, step_size)]
            # to consider plotting this vector
            mean_cost_vec.append(numpy.mean(seqs_loglikelihood_vec))
            self._track_sga_optimizer(w, seqs_cost_vec, mean_cost_vec, k)
            if(self._exitloop):
                break
            self._elapsed_time = datetime.now()

            
        line = "---Model training--- end time {} \n".format(datetime.now())
        ReaderWriter.log_progress(line, log_file)
                
        return(w)

    def _sga_svrg(self, w, train_seqs_id):
        """ implements the stochastic variance reduced gradient
            see Johnson R, Zhang T. Accelerating Stochastic Gradient Descent using  Predictive Variance Reduction. 
        """
        
        num_epochs = self.training_description["num_epochs"]
        # rung stochastic gradient ascent to initialize the weights
        self.training_description["num_epochs"] = 1
        w_tilda_p = self._sga_classic(w, train_seqs_id)
        self.cleanup()

        self.training_description["num_epochs"] = num_epochs
        crf_model = self.crf_model
        regularization_type = self.training_description["regularization_type"]
        # regularization parameter lambda
        C = self.training_description['regularization_value']
        # number of training sequences
        N = len(train_seqs_id)
         
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")

        # keeps track of the log-likelihood of a sequence before weight updating
        seqs_loglikelihood_vec = numpy.zeros(N)
        seqs_id_mapper = {seq_id:unique_id for unique_id, seq_id in enumerate(train_seqs_id)}
        # step size decides the number of data points to average in the seqs_loglikelihood_vec
        # using 10% of data points
        step_size = round(N * 0.1)
        if step_size == 0:
            step_size = 1
        mean_cost_vec = [0]
        
        if(regularization_type == "l1"):
            u = 0
            q = numpy.zeros(len(w), dtype = "longdouble")
               
        eta = self.training_description["t0"]

        m = 2*N
        saved_grad = {}
        # instance variable to keep track of elapsed time between optimization iterations
        self._elapsed_time = datetime.now()
        self._exitloop = False
        
        for s in range(num_epochs):
            print("we are in stage {}".format(s))
            w_tilda_c = w_tilda_p
            
            # ###################################
            # compute the average gradient 
            mu_grad = numpy.zeros(len(w_tilda_c), dtype = "longdouble")
            # compute average gradient
            for seq_id in train_seqs_id:
                seq_grad = crf_model.compute_seq_gradient(w_tilda_c, seq_id)
                windx = list(seq_grad.keys())
                fval = list(seq_grad.values())
                mu_grad[windx] += fval
                crf_model.clear_cached_info([seq_id])
                saved_grad[seq_id] = {'windx':windx, 'fval':numpy.asarray(fval)}
            if(C and regularization_type == "l2"):
                mu_grad -= (C* w_tilda_c)
            mu_grad = mu_grad/N
            #######################################
                
            w = numpy.copy(w_tilda_c) 
                
            for t in range(m):
                seq_id = numpy.random.choice(train_seqs_id, 1)[0]
                print("eta {}".format(eta))
                print(seq_id)
                print("we are in round {} out of {}".format(t+1, m))
                
                seq_loglikelihood = crf_model.compute_seq_loglikelihood(w, seq_id)
                seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood
                seq_grad = crf_model.compute_seq_gradient(w, seq_id)
                windx = list(seq_grad.keys())
                fval = numpy.asarray(list(seq_grad.values()))
                if(C):
                    grad = numpy.zeros(len(w), dtype = "longdouble")
                    grad[windx] = fval
                    if(regularization_type == 'l2'):
                        seq_loglikelihood += - ((C/N) * (1/2) * numpy.dot(w, w))
                        grad -= ((C/N)* w)
                        grad[saved_grad[seq_id]['windx']] -= saved_grad[seq_id]['fval'] 
                        grad += ((C/N) * w_tilda_c) + mu_grad
                        w += eta * grad
                        
                    elif(regularization_type == 'l1'):
                        seq_loglikelihood += - (C/N) * numpy.sum(numpy.abs(w))
                        u += eta * (C/N)
                        grad[saved_grad[seq_id]['windx']] -= saved_grad[seq_id]['fval'] 
                        grad +=  mu_grad
                        w_upd, q_upd = self._apply_l1_penalty(w, q, u, windx)
                        w = w_upd
                        q = q_upd
                        
                    # update the computed sequence loglikelihood by adding the regularization term contribution   
                    seqs_loglikelihood_vec[seqs_id_mapper[seq_id]] = seq_loglikelihood

                else:                    
                    w[windx] += eta * (fval - saved_grad[seq_id]['fval'])
                    w += eta * mu_grad
                    
                t += 1
                # clean cached info
                crf_model.clear_cached_info([seq_id])
            w_tilda_p = w
                
            seqs_cost_vec = [numpy.mean(seqs_loglikelihood_vec[i:i+step_size]) for i in range(0, N, step_size)]
            # to consider plotting this vector
            mean_cost_vec.append(numpy.mean(seqs_loglikelihood_vec))
            self._track_sga_optimizer(w, seqs_cost_vec, mean_cost_vec, s)
            if(self._exitloop):
                break
            self._elapsed_time = datetime.now()

            
        line = "---Model training--- end time {} \n".format(datetime.now())
        ReaderWriter.log_progress(line, log_file)
                
        return(w)     
        
    def _apply_l1_penalty(self, w, q, u, w_indx):
        for indx in w_indx:
            z = w[indx]
#             print("z is {}".format(z))
#             print("q[indx] is {}".format(q[indx]))
            if(w[indx] > 0):
#                 print("we want the max between 0 and {}".format(w[indx] - (u + q[indx])))
                w[indx] = numpy.max([0, w[indx] - (u + q[indx])])
            elif(w[indx] < 0):
#                 print("we want the min between 0 and {}".format(w[indx] + (u - q[indx])))
                w[indx] = numpy.min([0, w[indx] + (u - q[indx])])
#             print("z is {}".format(z))
#             print("w[indx] is {}".format(w[indx]))
            q[indx] = q[indx] + (w[indx] - z)
        return((w, q))
#             print("q[indx] becomes {}".format(q[indx]))

    def _track_sga_optimizer(self, w, seqs_loglikelihood, mean_loglikelihood, k):
        
        delta_time = datetime.now() - self._elapsed_time 
        self._check_reldiff(mean_loglikelihood[-2], mean_loglikelihood[-1])
#         
        epoch_num = k
        # log file 
        model_dir = self.training_description["model_dir"]
        log_file = os.path.join(model_dir, "crf_training_log.txt")
        line = "--- Epoch/pass {} --- \n".format(epoch_num+1)
        line += "Estimated training cost (average loglikelihood) is {} \n".format(mean_loglikelihood[-1])
        line += "Number of seconds spent: {} \n".format(delta_time.total_seconds())
        ReaderWriter.log_progress(line, log_file)
        ####  dump data for every epoch/pass ####
        # dump the learned weights for every pass
#         self.dump_file(w, os.path.join(model_dir, "model_weights_epoch_{}".format(k+1)))
#         self.dump_file(seqs_loglikelihood, os.path.join(model_dir, "seqs_loglikelihood_epoch_{}".format(k+1)))
#         
        # plot the estimated seqs_loglikelihood
#         colormap = plt.cm.get_cmap("Spectral")
#         colors = [colormap(i) for i in numpy.linspace(0, 0.9, total_epochs)]
#         color = colors[epoch_num]
#         plt_dir = create_directory("plot", model_dir)
#         self._plot_cost(seqs_loglikelihood, fig_num = epoch_num+1, plt_dir = plt_dir, color=color)
        
#     def _plot_cost(self, seqs_cost, fig_num, plt_dir, color):
#         # fig_num = 1 means we are in the first epoch and hence new training -- 
#         if(fig_num == 1):
#             plt.clf()
#             plt.figure(1)
#         group_num = numpy.arange(1, len(seqs_cost)+1)
#         plt_axis = plt.subplot(111)
#         plt_axis.plot(group_num, seqs_cost, marker = "o", color=color)
#         plt.ylabel('Average log-likelihood per group of sequences')
#         plt.xlabel('Group number')
# #         plt_box = plt_axis.get_position()
# #         plt_axis.set_position([plt_box.x0, plt_box.y0 + plt_box.height * 0.05, plt_box.width, plt_box.height * 0.95])
#         plt_axis.legend(["Epoch {}".format(i+1) for i in range(fig_num)],
#                     ncol=4, 
#                     loc='best', 
#                     numpoints = 1,
#                     #bbox_to_anchor=[0.5, -0.15], 
#                     prop = {'size':10},
#                     columnspacing=1.0, labelspacing=0.0,
#                     handletextpad=0.0, handlelength=1,
#                     fancybox=False, shadow=False)
#         plt.savefig(os.path.join(plt_dir,'pass_{}.pdf'.format(fig_num)))

    
    def cleanup(self):
        #---------------------
        # End of training -- cleanup
        #---------------------
        # reset iteration counter
        self._iter_count = None
        # reset elapsed time between iterations
        self._elapsed_time = None
        self._exitloop = None


class Evaluator(object):
    def __init__(self, model_repr):
        """model_repr : the crf model representation 
        (i.e. any instance of model class ending with ModelRepresentation such as HOSemiCRFModelRepresentation)"""
        self.model_repr = model_repr
        
    def compute_model_performance(self, Y_seqs_dict, metric, output_file):
        """ tags that did not show up in the training data can not be counted in this process
            In other words, the model cannot predict a tag that did not exist in the training data
            hence we give one unique index for tags that did not occur in the training data such as len(Y_codebook)
        """
        Y_codebook = self.model_repr.Y_codebook
        Y_codebook_rev = {code:state for state, code in Y_codebook.items()}
        M = self.model_repr.num_states
        model_taglevel_performance = numpy.zeros((M, 2, 2))

        for seq_id in Y_seqs_dict:
            Y_pred = Y_seqs_dict[seq_id]['Y_pred']
            Y_ref = Y_seqs_dict[seq_id]['Y_ref']
            taglevel_performance = self.compute_tags_confusionmatrix(self.map_states_to_num(Y_ref, Y_codebook, M),
                                                                     self.map_states_to_num(Y_pred, Y_codebook, M),
                                                                     Y_codebook_rev,
                                                                     M)
#             print("taglevel_performance {}".format(taglevel_performance))
#             print("tagging performance \n {}".format(taglevel_performance))
            model_taglevel_performance += taglevel_performance

        # perform sum across all layers to get micro-average
        collapsed_performance = model_taglevel_performance.sum(axis = 0)
#         print("collapsed performance \n {}".format(collapsed_performance))
        tp = collapsed_performance[0,0]
        fp = collapsed_performance[0,1]
        fn = collapsed_performance[1,0]
        tn = collapsed_performance[1,1]

        if(metric == "f1"):
            precision = tp/(tp + fp)
            recall = tp/(tp + fn)
            f1 = 2 * ((precision * recall)/(precision +  recall))
            print("f1 {}".format(f1))
            return(f1)
        elif(metric == "precision"):
            print("precision {}".format(precision))
            return(precision)
        elif(metric == "recall"):
            recall = tp/(tp + fn)
            print("recall {}".format(recall))
            return(recall)
        elif(metric == "accuracy"):
            accuracy = (tp + tn)/(tp + fp + fn + tn)
            print("accuracy {}".format(accuracy))
            return(accuracy)
        
    def map_states_to_num(self, Y, Y_codebook, M):
        Y_coded = [Y_codebook[state] if state in Y_codebook else M for state in Y]
#         print("Y_coded {}".format(Y_coded))
        return(Y_coded)
        
    def compute_tags_confusionmatrix(self, Y_ref, Y_pred, Y_codebook_rev, M):
        # compute confusion matrix on the level of the tag/state
#         print("Y_codebook {}".format(Y_codebook_rev))
        detected_statescode = set(Y_ref).union(set(Y_pred))
#         print("detected_statescode {}".format(detected_statescode))
        valid_statescode = [statecode for statecode in detected_statescode if statecode in Y_codebook_rev]
#         print("valid_statescode {}".format(valid_statescode))
        Y_ref = numpy.asarray(Y_ref)
        Y_pred = numpy.asarray(Y_pred)
#         print("Y_ref as numpy array {}".format(Y_ref))
#         print("detected states code \n {}".format(detected_statescode))
        tagslevel_performance = numpy.zeros((M, 2,2))
        
        for statecode in valid_statescode:
            # get all indices of the target tag (gold-standard)
            tag_indx_origin = numpy.where(Y_ref == statecode)[0]
            # get all indices of the target tag (predicted)
            tag_indx_pred = numpy.where(Y_pred == statecode)[0]
            tag_tp = len(numpy.where(numpy.in1d(tag_indx_origin, tag_indx_pred))[0])
            tag_fn = len(tag_indx_origin) - tag_tp
            other_indx_origin = numpy.where(Y_ref != statecode)[0]
            tag_fp = len(numpy.where(numpy.in1d(other_indx_origin, tag_indx_pred))[0])
            tag_tn = len(other_indx_origin) - tag_fp
            tagslevel_performance[statecode] = numpy.array([[tag_tp, tag_fp], [tag_fn, tag_tn]])
            
        return(tagslevel_performance)
if __name__ == "__main__":
    pass
    