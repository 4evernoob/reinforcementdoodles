from os import system, name

# import sleep to show output for some time period
from time import sleep

# define our clear function
def clear():

    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

from matplotlib import animation



def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('stuff.mp4', writer=writer)
    #display(display_animation(anim, default_mode='loop'))
from gym import envs
from collections import deque
print(envs.registry)
import numpy as np
import torch
import gym
from torch import nn,optim
import random
from matplotlib import pyplot as plt
#state size vars
rz=8
ins=rz*rz
torch.manual_seed(666)
en= gym.make('FrozenLake8x8-v0')
def desc2mat(desc):
    res=np.zeros((5,desc.shape[0],desc.shape[1]))
    x,y=desc.shape
    mapp=[b'S',b'F',b'H',b'G']
    for row in range(y):
        for col in range(x):
            idx=mapp.index(desc[row][col])
            res[idx][row][col]=1
            if idx==0:
                res[4][row][col]=1

    return res+np.random.rand(res.shape[0],res.shape[1],res.shape[2])/10.0#.reshape(1,4*8*8)
def updatepos(rew,newpos,v=False):
   # print(rew)
    rew=np.random.rand(rew.shape[0],rew.shape[1])/10.00
    #print(int(newpos/8))
    rew[int(newpos/rz),newpos%rz]+=1

    if v:
        print(rew)
    return rew
    #return res



def mk_model(en,basemodel,cut=.3):
    cur_state=en.reset()
    npenv=desc2mat(en.desc)
    print(en.action_space)

    lay1=200
    lay2=190
    outp=4
    model =nn.Sequential(
        nn.Linear(ins,lay1),
        nn.ReLU(),
        nn.Linear(lay1,lay2),
        nn.ReLU(),
        nn.Linear(lay2,outp),
        nn.Sigmoid()

    )
    model.load_state_dict(basemodel)
    loss_fn=nn.MSELoss()
    op=optim.Adam(model.parameters(),lr=.0001)
    #print(en.reward_range)
    #print(en.render())
    #print(en.observation_space)
    cur_s_ten=torch.Tensor(torch.from_numpy(npenv[4]).float().reshape(1,ins))
    epochs=800
    eps=1.0
    gamma=.9
    losses=[]
    epss=[]
    mem_size=1000
    mem=deque(maxlen=mem_size)
    batch=500

    ii=0
    #implement evolution in this shit jajajaj
    for i in range(epochs):
        #
        #print(eps,'hhh')

        don=True
        cur_state=en.reset()
        npenv=desc2mat(en.desc)
        cur_s_ten=torch.Tensor(torch.from_numpy(npenv[4,:,:]).float().reshape(1,ins))
        istep=0
        #print(ii)
        while don:
            ii+=1
            pred=model(cur_s_ten)
            pred_np=pred.data.numpy()


            if random.random()<eps:
                act=random.randint(0,3)
            else:
                act=np.argmax(pred_np)
            #try:
            kk=en.step(act)
            if len(kk)!=4:
                print(kk)
            else:
                pos,rew,done,_=kk[0],kk[1],kk[2],kk[3]
            #except:
            #    break
            npenv2=desc2mat(en.desc)
            npenv2[4,:,:]=updatepos(npenv2[4,:,:],pos)
            new_s_ten=torch.Tensor(torch.from_numpy(npenv2[4,:,:]).float().reshape(1,ins))
            with torch.no_grad():
                 newq=model(new_s_ten)
            maxq=torch.max(newq)
            if rew==0:
                Y = rew + (gamma * maxq)
            else:
                Y= rew
            Y = torch.Tensor([Y]).detach()
            X = pred.squeeze()[act] #O
            loss = loss_fn(X, Y) #P
            #print(i,eps, loss.item())
            op.zero_grad()
            loss.backward()
            losses.append(loss.item())
            op.step()

            don=not done
            # done = True if rew > 0 else False
            # exp =  (cur_s_ten, act, rew, new_s_ten, done) #G
            # mem.append(exp) #H
            # cur_s_ten = new_s_ten
            # # istep+=1
            # if len(mem) > batch: #I
            #     minibatch = random.sample(mem, batch) #J
            #     state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]) #K
            #     action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            #     reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            #     state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            #     done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
            #
            #     Q1 = model(state1_batch) #L
            #     with torch.no_grad():
            #         Q2 = model(state2_batch) #M
            #
            #     Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0]) #N
            #     X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            #     loss = loss_fn(X, Y.detach())
            #     if(ii%100==0):
            #         print(i, loss.item())
            #     #clear_output(wait=True)
            #     op.zero_grad()
            #     loss.backward()
            #     losses.append(loss.item())
            #     epss.append(eps)
            #     op.step()
            # if eps > 0.1: #R
            #     eps = np.exp(-1*i/epochs)
            # don=not done
            # if rew>0 or istep>(500*1+(i/100)):
            #
            #     break

            #print(don)

        eps = .99 if i<cut*epochs else .05#.99**(.36*(i+1)) #if eps>.3 else.05
    pass
    #print(losses)
    #plt.yscale('log')
    #plt.plot(losses)
    #plt.plot(epss)
    #plt.show()
    return model
def create(en,basemodel,gen_size=10,ngen=20):
    lim=.5
    for j in range(ngen):
        gen=[]
        print('gen ', j)
        for i in range(gen_size):

            seed=lim+random.uniform(-.5,.5)/((2**(0.7*j)))
            print('child ', i,' seed ',seed)
            mdl=mk_model(en,basemodel, cut=seed)
            fit=test(mdl,en)
            gen.append([seed,fit])
        gen.sort(key=lambda x: x[1], reverse=True)
        #clone
        lim=gen[0][0]+gen[1][0]
        lim =lim/2
    return mk_model(en,cut=lim)
def test(model,en,return_gif=False):
    fram=[]
    wc=0
    ng=100
    for i in range(ng):
        #print('new game ',i)
        cur_state=en.reset()
        npenv=desc2mat(en.desc)
        while True:

            cur_s_ten=torch.Tensor(torch.from_numpy(npenv[4,:,:]).float().reshape(1,ins))
            pred=model(cur_s_ten)
            pred_np=pred.data.numpy()


            #if random.random()<eps:
            #    act=random.randint(0,3)
            #else:
            act=np.argmax(pred_np)
            #print(act)
            try:
                pos,rew,done,_=en.step(act)
            except:
                break
            npenv2=desc2mat(en.desc)
            #print(pos)
            npenv2[4,:,:]=updatepos(npenv2[4,:,:],pos,v=False)
            #print(np.round(npenv2[4,:,:]))
            #print(en.render())
            fram.append(npenv2[4,:,:].copy()*100+npenv2[1,:,:].copy()*50+npenv2[2,:,:].copy()*-100)
            new_s_ten=torch.Tensor(torch.from_numpy(npenv2[4,:,:]).float().reshape(1,ins))
            cur_s_ten = new_s_ten

            #print(en.render())

            #clear()
            if rew==1:
                wc+=1
            if done:
                #print(rew)
                break
        fram.append(np.zeros((8,8)))
        fram.append(np.zeros((8,8)))
        fram.append(np.zeros((8,8)))
        fram.append(np.zeros((8,8)))
    print(100*wc/ng,'% algorithm won')
    if return_gif:
        display_frames_as_gif(fram)
    return 100*wc/ng
#evo
lay1=200
lay2=190
outp=4
basmodel =nn.Sequential(
    nn.Linear(ins,lay1),
    nn.ReLU(),
    nn.Linear(lay1,lay2),
    nn.ReLU(),
    nn.Linear(lay2,outp),
    nn.Sigmoid()

)


model=create(en,basmodel.state_dict())

test(model,en,return_gif=True)

torch.save(model, 'froz.modl')

#Load:

# Model class must be defined somewhere
#model = torch.load(PATH)
#model.eval()

