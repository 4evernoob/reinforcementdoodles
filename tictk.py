import numpy as np
import torch
import gym
from torch import nn,optim
import random
from gifmak import creategif
from matplotlib import pyplot as plt
# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
torch.autograd.set_detect_anomaly(True)

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    #fig, ax=plt.subplots()
    patch = plt.imshow(frames[0])
    print(type(patch))
    # tt=''
    # for k in range(3):
    #
    #     for j in range(3):
    #         if frames[i][k, j]==-.5:
    #             tt='X'
    #         elif frames[i][k, j]==.5:
    #             tt='O'
    #         text = ax.text(j, k,tt ,
    #                ha="center", va="center", color="w")
    plt.colorbar(patch)
    plt.axis('off')

    def animate(i):
        patch.set_data(np.zeros((3,3)))
        patch.set_data(frames[i])

        # print(i)
        # tt=''
        # for k in range(3):
        #      for j in range(3):
        #          if frames[i][k, j]==-.5:
        #              tt='X'
        #          elif frames[i][k, j]==.5:
        #              tt='O'
        #          text = ax.text(j, k,tt ,
        #                 ha="center", va="center", color="w")



    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=30)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('stuff.mp4', writer=writer)
    #display(display_animation(anim, default_mode='loop'))
from tictactoe import TicTacToe
def statemaker(a,b):
    #return np.concatenate((a.reshape(1,9),b.reshape(1,9))).copy()
    return a.reshape(1,9)
ins=9
lay1=25
lay2=18
outp=9
model =nn.Sequential(
    nn.Linear(ins,lay1),
    nn.ReLU(),
    nn.Linear(lay1,lay2),
    nn.ReLU(),
    nn.Linear(lay2,outp),
    nn.Sigmoid()

)
model2 =nn.Sequential(
    nn.Linear(ins,lay1),
    nn.ReLU(),
    nn.Linear(lay1,lay2+10),
    nn.ReLU(),
    nn.Linear(lay2+10,outp),
    nn.Sigmoid()

)
#model.load_state_dict(basemodel)
loss_fn=nn.MSELoss()
op=optim.Adam(model.parameters(),lr=.001)

loss_fn2=nn.MSELoss()
op2=optim.Adam(model2.parameters(),lr=.003)
epochs=1000
eps=1
verbose=False
game=TicTacToe()
print(game.av_moves())
flag2cpu=True
gamma=.9
losses=[]
losses2=[]
for i in range(epochs):
   eps=.5+.5*np.tanh(-5*(i/epochs)+2)
   #print(eps)
   game=TicTacToe()
   state=torch.Tensor(torch.from_numpy(statemaker(game.render(),game.render())).float().reshape(1,ins))
   #second input
   if flag2cpu:
        state2=torch.Tensor(torch.from_numpy(statemaker(game.render().copy(),game.render().copy())).float().reshape(1,ins))
   #print(state)
   ii=0
   while True:
       pred=model(state)
       moves_used, moves_pos=game.av_moves()
       #player1
       if len(moves_used)==9:
            newstatus=game.render()
       else:
            pred_n=pred.data.numpy().copy()
            pred_n[0,moves_used]=0
            act=np.argmax(pred_n)
            if random.random()> eps:
                newstatus,reward,moved=game.move(act,1)
            else:
                newstatus,reward,moved=game.move(random.choice(moves_pos),1)
            if reward==1:
                reward2=-3
            newstatus=game.render()
       moves_used, moves_pos=game.av_moves()
       #player2
       if len(moves_used)==9:
           newstatus2=game.render()
       else:
           if(not flag2cpu):
            newstatus2,reward2,moved2 =game.move(random.choice(moves_pos),2)
           else:
            #print("train 2s player")
            pred2=model2(state2)
            pred2_n=pred2.data.numpy().copy()
            pred2_n[0,moves_used]=0
            act2=np.argmax(pred2_n)
            if random.random()> eps:
                newstatus2,reward2,moved2=game.move(act2,2)
            else:
                newstatus2,reward2,moved2=game.move(random.choice(moves_pos),2)
            newstatus2=game.render()
            if reward2==1 and reward!=1:
              reward=-3
       #print(moved2,'p2')
      #print(game.render())
            newstatus2=game.render()

       newstate=torch.Tensor(torch.from_numpy(statemaker(newstatus,newstatus2)).float().reshape(1,ins))
       if flag2cpu:
        moves_used, moves_pos=game.av_moves()
        if len(moves_pos)==0:
            break

        newstate2=torch.Tensor(torch.from_numpy(statemaker(newstatus2.copy(),newstatus.copy())).float().reshape(1,ins))
        with torch.no_grad():
         newq2=model2(newstate2)
        maxq2=torch.max(newq2)
        if reward2<0:
          Y2 = reward2*10+ (gamma * maxq2)
        else:
          Y2= reward2*2
        Y2 = torch.Tensor([Y2]).detach()
        X2 = pred2.squeeze()[act2] #O
        loss2 = loss_fn2(X2, Y2) #P
        if verbose:
            print(game.av_moves())
            print(game.render(clean=True))
        print(i," net 2 ",ii,eps, loss2.item())
        op2.zero_grad()
        loss2.backward()
        losses2.append(loss2.item())
        op2.step()
        laststate=state2.clone()
        state2=newstate2
        ###original bot
       with torch.no_grad():
         newq=model(newstate)
       maxq=torch.max(newq)
       if reward<0:
          Y = reward + (gamma * maxq)
       else:
          Y= reward
       Y = torch.Tensor([Y]).detach()
       X = pred.squeeze()[act] #O
       loss = loss_fn(X, Y) #P
       if verbose:
            print(game.av_moves())
            print(game.render(clean=True))
       print(i,' net 1 ',ii,eps, loss.item())
       op.zero_grad()
       loss.backward()
       losses.append(loss.item())
       op.step()
       state=newstate
       ii+=1
       if(reward==1 or reward==-2 or reward==-3):
           print('p1 won'if reward==1 else'p1 didn won')
           print(reward)
           break

       #print(i,loss.item())
print(losses2)
plt.plot(losses)
plt.plot(losses2)
plt.show()
tests=500
cnt=0
tiesc=0
frames=[]
flag2cpu=True
for i in range(tests):
    game=TicTacToe()
    state=torch.Tensor(torch.from_numpy(statemaker(game.render(),game.render())).float().reshape(1,ins))
    if flag2cpu:
            state2=torch.Tensor(torch.from_numpy(statemaker(game.render().copy(),game.render().copy())).float().reshape(1,ins))

    frames.append(game.render(clean=True))
    #frames.append(np.zeros((3,3)))
    c=0
    while True:


        pred= model(state)
        moves_used, moves_pos=game.av_moves()

        #print(game.av_moves(), 'initial')
        if len(moves_pos)==0:
           break

        if c==0:
            act=random.choice(moves_pos)
            c+=1
        else:
            pred_n=pred.data.numpy()
            pred_n[0,moves_used]=0
            #print(pred_n)
            act=np.argmax(pred_n)


        #act=np.argmax(pred.data.numpy())
        #print(act)
        newstatus, reward,mov=game.move(act,1)

        if reward==1:
            print('CPU won\n')
            cnt+=1
            break
        moves_used, moves_pos=game.av_moves()
        frames.append(game.render(clean=True))
        #frames.append(np.zeros((3,3)))
        #print(game.av_moves(),'p1')
        newstatus=game.render()
        moved2=False
        if flag2cpu:
            print('CPU2')
            pred2= model2(state2)
            moves_used, moves_pos=game.av_moves()
            print(game.av_moves(), 'CPU2 mov')
            if len(moves_pos)==0:
               break
            pred2_n=pred2.data.numpy()
            pred2_n[0,moves_used]=0
            act2=np.argmax(pred2_n)
            newstatus2, reward2,mov2=game.move(act2,2)
            newstatus2=game.render()
        else:
            while not moved2:
                if len(moves_pos)==0:
                    break
                if i >tests-3:
                    print(game.render(clean=True))
                    movedata=input('your move')
                    newstatus2,reward2,moved2 =game.move(int(movedata)-1,2)
                else:
                    newstatus2,reward2,moved2 =game.move(random.choice(moves_pos),2)
            newstatus2=game.render()
        frames.append(game.render(clean=True))
        #frames.append(np.zeros((3,3)))
        #print(game.av_moves(),'p2')
        #print(game.av_moves(),'p2')
        if reward2==1 and reward!=1:
            reward=-3
        print(game.render(clean=True))
        print('*-'*20)
        newstate=torch.Tensor(torch.from_numpy(statemaker(newstatus,newstatus2)).float().reshape(1,ins))
        if flag2cpu:
            newstate2=torch.Tensor(torch.from_numpy(statemaker(newstatus2.copy(),newstatus.copy())).float().reshape(1,ins))
        state=newstate
        if flag2cpu:
            state2=newstate2
        if(reward==1 or reward==-2 or reward==-3):
           print('CPU won\n'if reward==1 else'CPU didn won\n')
           print(reward)
           if reward==1:
               cnt+=1
           if reward==-2:
               tiesc+=1
           break
    frames.append(np.zeros((3,3)))
    frames.append(np.zeros((3,3)))
    frames.append(np.zeros((3,3)))
    frames.append(np.zeros((3,3)))
    frames.append(np.zeros((3,3)))
#display_frames_as_gif(frames)
creategif('games',frames)
print('won',cnt)
print('ties',tiesc)
print('lost',tests-cnt-tiesc)
