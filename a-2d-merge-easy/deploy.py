from multiprocessing import Process, Queue
import os



free_cores = [4, 5, 6, 7]
n_threads_per_core = 10
  


def run_job(q, gpu_i):
    print("process started")
    print(f"using gpu {gpu_i}")
    while q:
        config = q.get()
        print("config "+str(config))
        os.system(f"screen -Dm -L -Logfile out/out{config}.txt python test_env.py -g {gpu_i}")
    print("process done")

  
if __name__ == '__main__':

    q = Queue()
    configs = list(range(100))

    for config in configs:
        q.put(config)


    processes = []
    for gpu_i in free_cores:
        for _ in range(n_threads_per_core):
            p = Process(target=run_job, args=(q, gpu_i))
            processes.append(p)
            p.start()
    for p in processes:
        p.join()




