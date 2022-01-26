"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
import multiprocessing as mp

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.

# class Pool(mp.pool.Pool):
#     Process = NoDaemonProcess

class NonDaemonPool(mp.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc


class ParallelEvaluator(object):
    def __init__(self, num_workers, fitness_function, constraint_function=None, timeout=None):
        """
        fitness_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        constraint_function should take one argument, a tuple of
        (genome object, config object), and return
        a single bool (the genome's validity).
        """
        self.num_workers = num_workers
        self.fitness_function = fitness_function
        self.constraint_function = constraint_function
        self.timeout = timeout
        self.pool = NonDaemonPool(num_workers)

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate_fitness(self, genomes, config, generation):
        jobs = []
        for i, (_, genome) in enumerate(genomes):
            jobs.append(self.pool.apply_async(self.fitness_function, (genome, config, i, generation)))

        # assign the fitness back to each genome
        for job, (_, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)

    def evaluate_constraint(self, genomes, config, generation):
        validity_all = []
        for i, (_, genome) in enumerate(genomes):
            validity = self.constraint_function(genome, config, i, generation)
            validity_all.append(validity)
        return validity_all
