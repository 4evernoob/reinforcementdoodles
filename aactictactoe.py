import torch.multiprocessing as mp
#actor critic version of tictactoe
from actorcriticutils import ActorCritic,worker
if __name__ == '__main__':

    Masta = ActorCritic(9,30,29,26,9,1)
    Masta.share_memory()
    processes=[]
    params={'epochs':2000,'workers':2}
    counter=mp.Value('i',0)

    for i in range(params['workers']):
        p=mp.Process(target=worker,args=(i,Masta,counter,params))
        p.start()
        processes.append(p)
    for p in processes:
            p.join()
    for p in processes:
            p.terminate()
    print(counter.value,processes[1].exitcode)
