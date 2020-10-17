import numpy as np
import torch
import gym
from torch import nn,optim
import random
from gifmak import creategif
from matplotlib import pyplot as plt
torch.autograd.set_detect_anomaly(True)
from tictactoe import TicTacToe

def statemaker(a,b):
    #return np.concatenate((a.reshape(1,9),b.reshape(1,9))).copy()
    return a.reshape(1,9)

ins=9
lay1=50
lay2=40
outp=9
#player1
model =nn.Sequential(
    nn.Linear(ins,lay1),
    nn.ReLU(),
    nn.Linear(lay1,lay2),
    nn.ReLU(),
    nn.Linear(lay2,outp),
    nn.Sigmoid()

)
#player2
model2 =nn.Sequential(
    nn.Linear(ins,lay1),
    nn.ReLU(),
    nn.Linear(lay1,lay2),
    nn.ReLU(),
    nn.Linear(lay2,outp),
    nn.Sigmoid()

)
#model.load_state_dict(basemodel)
loss_fn=nn.MSELoss()
op=optim.Adam(model.parameters(),lr=.001)

loss_fn2=nn.MSELoss()
op2=optim.Adam(model2.parameters(),lr=.001)

epochs=5000
eps=1
verbose=False
game=TicTacToe()
print(game.av_moves())
flag2cpu=True
gamma=.8
losses=[]
losses2=[]
framest=[]
tw=0
tt=0
tl=0
tnm=0
for i in range(2*epochs):
   if epochs==i:
       flag2cpu=False
   #epsilon decaying factor
   eps=.5+.5*np.tanh(-10*(i/epochs)+2)
   #print(eps)
   game=TicTacToe()
   state=torch.Tensor(torch.from_numpy(statemaker(game.render(),game.render())).float().reshape(1,ins))
   #second state
   if flag2cpu:
        state2=torch.Tensor(torch.from_numpy(statemaker(game.render().copy(),game.render().copy())).float().reshape(1,ins))
   #print(state)
   framest.append(game.render(clean=True))
   ii=0
   while True:
       #print(game.render(clean=True))
       pred=model(state)
       moves_used, moves_pos=game.av_moves()
       print(game.player1,' ', game.player2) if verbose else None
       #player1
       if game.player2 not in [game.win]:
           if len(moves_used)==9:
                newstatus=game.render()
           else:
                pred_n=pred.data.numpy().copy()
                pred_n[0,moves_used]=0
                act=np.argmax(pred_n)
                if random.random()> eps:
                    newstatus,reward,reward2,moved=game.move(act,1)
                else:
                    #moves_pos=[i for i in range(9)]
                    newstatus,reward,reward2,moved=game.move(random.choice(moves_pos),1)
                newstatus=game.render()
                framest.append(game.render(clean=True))
           newstate=torch.Tensor(torch.from_numpy(newstatus).float().reshape(1,ins))
           print(game.render(clean=True))if verbose else None
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

       #get moves available
       moves_used, moves_pos=game.av_moves()
       print(game.player1,' ', game.player2)if verbose else None
       #player2
       if game.player1 not in [game.win]:
           if len(moves_used)==9:
               newstatus2=game.render()
           else:
               if(not flag2cpu):
                newstatus2,reward,reward2,moved2 = game.move(random.choice(moves_pos),2)
               else:
                #print("train 2s player")
                pred2=model2(state2)
                pred2_n=pred2.data.numpy().copy()
                pred2_n[0,moves_used]=0
                act2=np.argmax(pred2_n)
                if random.random()> eps:
                    newstatus2,reward,reward2,moved2=game.move(act2,2)
                else:
                    #moves_pos=[i for i in range(9)]
                    newstatus2,reward,reward2,moved2=game.move(random.choice(moves_pos),2)
                newstatus2=game.render()
                framest.append(game.render(clean=True))
                print(game.render(clean=True))if verbose else None
           #print(moved2,'p2')
          #print(game.render())
           if flag2cpu:
            moves_used, moves_pos=game.av_moves()
            if len(moves_pos)==0:
                pass
            else:
                newstate2=torch.Tensor(torch.from_numpy(newstatus2).float().reshape(1,ins))
                with torch.no_grad():
                 newq2=model2(newstate2)
                maxq2=torch.max(newq2)
                if reward2<0:
                  Y2 = reward2+ (gamma * maxq2)
                else:
                  Y2= reward2
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
       state=newstate
       print(game.player1,' ', game.player2)
       ii+=1

       if game.isgameover():
           framest.append(np.zeros((3,3)))
           framest.append(np.zeros((3,3)))
           framest.append(np.zeros((3,3)))
           if game.player1 ==game.win:
              tw+=1
              print('P1 won')
           elif game.player1 ==game.tie:
              print('P1 tied')
              tt+=1
           elif game.player1 == game.notmoving:
              print('P1 lost by not moving')
              tnm+=1
           else:
               print('P1 Lost')
               tl+=1
           break

       #print(i,loss.item())
print('train wins ',100*tw/epochs)
print('train ties ',100*tt/epochs)
print('train losses ',100*tl/epochs)
plt.plot(losses)
plt.plot(losses2)
plt.show()
#creategif('traingames',framest)
#exit()
tests=80
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
        newstatus, reward,reward2,mov=game.move(act,1)
        print(newstatus, reward,reward2,mov) if verbose else None
        print(game.render(clean=True))
        frames.append(game.render(clean=True))
        if game.isgameover():
           game.printp1()
           if game.player1==game.win:
               cnt+=1
           if game.player1==game.tie:
               tiesc+=1
           break
        moves_used, moves_pos=game.av_moves()

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
            newstatus2,reward, reward2,mov2=game.move(act2,2)
            newstatus2=game.render()
        else:
            while not moved2:
                if len(moves_pos)==0:
                    break
                if i >tests-3:
                    print(game.render(clean=True))
                    movedata=input('your move')
                    newstatus2,reward,reward2,moved2 =game.move(int(movedata)-1,2)
                else:
                    newstatus2,reward,reward2,moved2 =game.move(random.choice(moves_pos),2)
            newstatus2=game.render()
        frames.append(game.render(clean=True))
        #frames.append(np.zeros((3,3)))
        #print(game.av_moves(),'p2')
        #print(game.av_moves(),'p2')
        print(game.render(clean=True))
        print('*-'*20)
        print('\n'*10)
        newstate=torch.Tensor(torch.from_numpy(newstatus).float().reshape(1,ins))
        if flag2cpu:
            newstate2=torch.Tensor(torch.from_numpy(newstatus2).float().reshape(1,ins))
        state=newstate
        if flag2cpu:
            state2=newstate2
        if(reward in game.endingcriteria()):
           game.printp1()
           if game.player1==game.win:
               cnt+=1
           if game.player1==game.tie:
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
