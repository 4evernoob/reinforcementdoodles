import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from collections import deque
mem_size = 5000 #A
batch_size = 2000 #B
replay = deque(maxlen=mem_size) #C
def softmax(av, tau=1.12):
    softm = ( np.exp(av / tau) / np.sum( np.exp(av / tau) ) )
    return softm
mlayer=124
model = torch.nn.Sequential(
    torch.nn.Linear(3, mlayer),
    torch.nn.ReLU(),
    torch.nn.Linear(mlayer, 2),
    #torch.nn.Softmax(dim=0)
    torch.nn.Tanh()
)
loss_fn = torch.nn.MSELoss()
J = 0.01  #moment of inertia of the rotor
b = 0.1   #damping ratio of the mechanical system
K = 0.01  #electromotive force constant; K=Ke=Kt
R = 1     #electric resistance
L = 0.5   #electric inductance
def motor(x,u,r):
 k1 = x[1]
 k2  = -.1 * x[1] - .80 * x[0] + u - r
 return np.array([k1,k2])
losses=[]
x=np.array([.9,-1])
des=0.8
u=0.0
ei=0
r=0
resp=[]
uh=[]
rewards=[]
tmps=np.concatenate((x,[des-x[1]])).reshape(1,3)
#tmps=x.reshape(1,2)
cur_state = torch.Tensor(tmps) #A
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

pert=[]
gamma=.9
for i in range(20000):
 if i %1000==0:
  pass
  #u=1
  r=9*random.random()
 if i==2000:
  des=6.3*random.random()


 #.3*(des-x[1])+.04*ei
 y_pred = model(cur_state) #B
 y_pred_=y_pred.data.numpy()
 if (random.random() < .1): #I
    tm = np.random.randint(0,2)
 else:
    tm = np.argmax(y_pred_)
    #tm=np.argmax(y_pred_)
    #actions
 if tm==0:
        u=u+.03

# elif tm==1:
 #      u=u
 else:
       u=u-.03
  #execute shit
 xn=motor(x,u,r)
 ei=(des-xn[1])
 tmprw=np.concatenate((xn,[des-x[1]])).reshape(1,3)
 #tmprw=xn.reshape(1,2)
 ns=torch.Tensor(tmprw)
 #new state
 #xn

 #rewards
 if ei>.00:
  reward=1
 else:
  reward=-10
 #exp =  (cur_state, tm, reward,ns , True if reward > 0 else False)
 #print(exp)
 #replay.append(exp)

 #
 # if len(replay) > batch_size: #I
 #     minibatch = random.sample(replay, batch_size) #J
 #     state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]) #K
 #     #print(state1_batch)
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
 #     print(i, loss.item())
 #     #clear_output(wait=True)
 #     optimizer.zero_grad()
 #     loss.backward()
 #     losses.append(loss.item())
 #     optimizer.step()

 # q2 actualization
 with torch.no_grad():
  newy_pred=model(ns)
 maxQ=torch.max(newy_pred)
 if reward<0:
  Y=reward+gamma*maxQ
 else:
  Y=reward
 #print(y_pred)
 Y=torch.Tensor([Y]).detach()
 X=y_pred.squeeze()[tm]
 #print(Y,X)
 loss=loss_fn(X,Y)
 #print(y_pred)
 optimizer.zero_grad()
 loss.backward()
 losses.append(loss.item())
 optimizer.step()
 rewards.append(reward)
 #print(xn)
 x=xn
 uh.append(u)
 resp.append(xn[1])
 #print(xn)
 x=xn
 cur_state = torch.Tensor(ns)
 #print(x)
 #loss = loss_fn(y_pred, torch.Tensor([cur_reward]))
 #print(loss)

 #cur_state = torch.Tensor(np.concatenate((x,[des-x[1]])))
plt.plot(losses)
plt.show()
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(rewards)
axs[0,1].plot(resp)

#exit()
x=np.array([.9,-1])
des=0.3
u=0.0
ei=0
r=0
resp=[]
uh=[]
rewards=[]
spp=[]

xx=np.concatenate((x,[des-x[1]])).reshape(1,3)
cur_state = torch.Tensor(xx) #A
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
for i in range(50000):
 if i%3000==0:
  pass
  #u=1
  r=9*random.random()
 if i%5000==0:
  des=5*random.random()
  print(des)
 y_pred = model(cur_state) #B
 y_pred_=y_pred.data.numpy()
 tm=np.argmax(y_pred_)
    #actions
 if tm==0:
        u=u+.03

# elif tm==1:
 #      u=u
 else:
       u=u-.03
  #execute shit
 xn=motor(x,u,r)
 ei=(des-xn[1])

 #print(xn)
 x=xn
 uh.append(u)
 resp.append(xn[1])
 pert.append(r)
 spp.append(des)
 #print(xn)

 #print(x)
 #loss = loss_fn(y_pred, torch.Tensor(cur_reward))
 #print(loss)
 #optimizer.zero_grad()
 #loss.backward()
 #optimizer.step()
 xx2=np.concatenate((x,[des-x[1]])).reshape(1,3)
 #xx2=x.reshape(1,2)
 cur_state = torch.Tensor(xx2)

axs[1,0].plot(uh)
#plt.show()
axs[1,1].plot(resp)
axs[1,1].plot(pert)
axs[1,1].plot(spp)

#plt.plot(uh)
plt.show()

