import multiprocessing
import time
import traceback

def job_wrapper(func, args, data_container):
        try:
            out_value = func(*args)
        except:
            print("ERROR\n")
            traceback.print_exc()
            print()
            return 0   
        data_container.value = out_value

class Group():

    def __init__(self):
        
        self.jobs = []
        self.return_data = []
        self.callback = []


    def add_job(self, func, args, callback):

        self.return_data.append(multiprocessing.Value("d", 0.0))
        self.jobs.append(multiprocessing.Process(target=job_wrapper, args=(func, args, self.return_data[-1])))
        self.callback.append(callback)

    def run_jobs(self, num_proc):
        
        next_job = 0
        num_jobs_open = 0
        jobs_finished = 0

        jobs_open = set()

        while(jobs_finished != len(self.jobs)):

            jobs_closed = []
            for job_index in jobs_open:
                if not self.jobs[job_index].is_alive():
                    self.jobs[job_index].join()
                    self.jobs[job_index].terminate()
                    num_jobs_open -= 1
                    jobs_finished += 1
                    jobs_closed.append(job_index)

            for job_index in jobs_closed:
                jobs_open.remove(job_index)

            while(num_jobs_open < num_proc and next_job != len(self.jobs)):
                self.jobs[next_job].start()
                jobs_open.add(next_job)
                next_job += 1
                num_jobs_open += 1

            time.sleep(0.1)

        for i in range(len(self.jobs)):
            self.callback[i](self.return_data[i].value)



