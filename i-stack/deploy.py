from multiprocessing import Process, Queue
import os



free_cores = [4, 5, 6, 7]
n_threads_per_core = 10
  


def run_job(p_name, q, gpu_i):
    print(f"process {p_name} gpu {gpu_i}")
    while True:
        try:
            config = q.get_nowait()
            print(f"process {p_name} config "+str(config))
            os.system(f"screen -Dm -L -Logfile out/out{config}.txt python test_env.py -g {gpu_i}")
        except:
            break
    print(f"process {p_name} done")

  
if __name__ == '__main__':

    q = Queue()
    configs = list(range(100))

    for config in configs:
        q.put(config)


    processes = []
    for gpu_i in free_cores:
        for _ in range(n_threads_per_core):
            p_name = len(processes)
            p = Process(target=run_job, args=(p_name, q, gpu_i))
            processes.append(p)
            p.start()
    for p in processes:
        p.join()




