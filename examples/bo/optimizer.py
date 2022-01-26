import numpy as np
import time
import GPyOpt
from GPyOpt.core.task.objective import SingleObjective
from GPyOpt.methods import ModularBayesianOptimization
from GPyOpt.util.general import get_d_moments


def spawn(f):
    '''
    Function for parallel evaluation of the acquisition function
    '''
    def fun(pipe,*args):
        pipe.send(f(*args))
        pipe.close()
    return fun


class Objective(SingleObjective):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        self.config = config

    def evaluate(self, x, generation):
        """
        Performs the evaluation of the objective at x.
        """
        if self.n_procs == 1:
            f_evals, cost_evals = self._eval_func(x, generation)
        else:
            try:
                f_evals, cost_evals = self._syncronous_batch_evaluation(x, generation)
            except:
                if not hasattr(self, 'parallel_error'):
                    print('Error in parallel computation. Fall back to single process!')
                else:
                    self.parallel_error = True
                f_evals, cost_evals = self._eval_func(x, generation)

        return f_evals, cost_evals

    def _eval_func(self, x, generation, idx=None, queue=None):
        """
        Performs sequential evaluations of the function at x (single location or batch). The computing time of each
        evaluation is also provided.
        """
        cost_evals = []
        f_evals     = np.empty(shape=[0, 1])
        if idx is None: idx = list(range(x.shape[0]))

        for i in range(x.shape[0]):
            st_time    = time.time()
            rlt = self.func(np.atleast_2d(x[i]), self.config, idx[i], generation)
            f_evals     = np.vstack([f_evals,rlt])
            cost_evals += [time.time()-st_time]
        if queue is None:
            return f_evals, cost_evals
        else:
            queue.put([idx, f_evals, cost_evals])

    def _syncronous_batch_evaluation(self, x, generation):
        """
        Evaluates the function a x, where x can be a single location or a batch. The evaluation is performed in parallel
        according to the number of accessible cores.
        """
        from multiprocessing import Process, Queue

        # --- parallel evaluation of the function
        divided_samples = [x[i::self.n_procs] for i in range(self.n_procs)]
        divided_idx = [list(range(x.shape[0]))[i::self.n_procs] for i in range(self.n_procs)]
        queue = Queue()
        proc = [Process(target=self._eval_func,args=(k, generation, idx, queue)) for k, idx in zip(divided_samples, divided_idx)]
        [p.start() for p in proc]

        # --- time of evaluation is set to constant (=1). This is one of the hypothesis of synchronous batch methods.
        f_evals = np.zeros((x.shape[0],1))
        cost_evals = np.ones((x.shape[0],1))
        for _ in proc:
            idx, f_eval, _ = queue.get() # throw away costs
            f_evals[idx] = f_eval
        return f_evals, cost_evals


class Optimization(ModularBayesianOptimization):

    def run_optimization(self, max_iter = 0, max_time = np.inf,  eps = 1e-8, context = None, verbosity=False, save_models_parameters= True, report_file = None, evaluations_file = None, models_file=None,
        before_evaluate = None, after_evaluate = None):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param report_file: file to which the results of the optimization are saved (default, None).
        :param evaluations_file: file to which the evalations are saved (default, None).
        :param models_file: file to which the model parameters are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.save_models_parameters = save_models_parameters
        self.report_file = report_file
        self.evaluations_file = evaluations_file
        self.models_file = models_file
        self.model_parameters_iterations = None
        self.context = context

        # --- Check if we can save the model parameters in each iteration
        if self.save_models_parameters == True:
            if not (isinstance(self.model, GPyOpt.models.GPModel) or isinstance(self.model, GPyOpt.models.GPModel_MCMC)):
                print('Models printout after each iteration is only available for GP and GP_MCMC models')
                self.save_models_parameters = False

        # --- Setting up stop conditions
        self.eps = eps
        if  (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:

            if before_evaluate is not None:
                before_evaluate(0)

            self.Y, cost_values = self.objective.evaluate(self.X, 0)

            if after_evaluate is not None:
                after_evaluate(0, self.Y.flatten())

            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y

        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time):
            # --- Update model
            try:
                self._update_model(self.normalization_type)
            except np.linalg.linalg.LinAlgError as e:
                # break
                print('Error: np.linalg.linalg.LinAlgError')

            if (self.num_acquisitions >= self.max_iter):
                break

            if (len(self.X) > 1 and self._distance_last_evaluations() <= self.eps):
                # break
                print('Error: distance_last_evaluations <= eps')

            self.num_acquisitions += 1

            self.suggested_sample = self._compute_next_evaluations()

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))

            if before_evaluate is not None:
                before_evaluate(self.num_acquisitions)

            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            if after_evaluate is not None:
                after_evaluate(self.num_acquisitions, self.Y_new.flatten())

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero

            if verbosity:
                print("num acquisition: {}, time elapsed: {:.2f}s".format(
                    self.num_acquisitions, self.cum_time))

        # --- Stop messages and execution time
        self._compute_results()

        # --- Print the desired result in files
        if self.report_file is not None:
            self.save_report(self.report_file)
        if self.evaluations_file is not None:
            self.save_evaluations(self.evaluations_file)
        if self.models_file is not None:
            self.save_models(self.models_file)

    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        self.Y_new, cost_new = self.objective.evaluate(self.suggested_sample, self.num_acquisitions)
        self.cost.update_cost_model(self.suggested_sample, cost_new)
        self.Y = np.vstack((self.Y,self.Y_new))