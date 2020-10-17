import torch as t
from tictactoe import TicTacToe
from torch import nn,optim
from torch.nn import functional as F
def run_ep(worker_env,worker_model):
    state=t.from_numpy(worker_env.render().reshape(1,9)).float()
    values,logprobs,rewards=[],[],[]
    done=False
    j=0
    while(not worker_env.isgameover()):
        j+=1
        policy,value=worker_model(state)
        values.append(value)
        logits=policy.view(-1)
        action_dist =t.distributions.Categorical(logits=logits)
        action=action_dist.sample()
        logprob_=policy.view(-1)[action]
        logprobs.append(logprob_)
        state_,rw,_,_ = worker_env.move(action.detach().numpy(),1)
        print(rw,worker_env.render(clean=True))
        state=t.from_numpy(state_).float()
        if worker_env.player1 in [worker_env.win,worker_env.tie]:
            reward=1
            worker_env.reset()
        elif worker_env.player1 in [worker_env.lose,worker_env.notmoving]:
            reward=-10
            worker_env.reset()
        elif worker_env.player1 in [worker_env.movesav]:
            reward=0
        elif worker_env.player1 in [worker_env.danger]:
            reward=-1
        rewards.append(reward)
        return values,logprobs,rewards
def update_params(w_opt,values,logprobs,rewards,clc=0.1,gamma=.8):
    rewards=t.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs=t.stack(logprobs).flip(dims=(0,)).view(-1)
    values=t.stack(values).flip(dims=(0,)).view(-1)
    Returns=[]
    ret_=t.Tensor([0])
    for r in range(rewards.shape[0]):
        ret_=rewards[r]+gamma*ret_
        Returns.append(ret_)
    Returns=t.stack(Returns).view(-1)
    Returns=F.normalize(Returns,dim=0)
    actor_loss=-1*logprobs*(Returns-values.detach())
    critic_loss=t.pow(values-Returns,2)
    loss=actor_loss.sum()+clc*critic_loss.sum()
    loss.backward()
    w_opt.step()
    print(loss.item())
    return actor_loss,critic_loss,len(rewards)



def worker(t,worker_model,counter,params):
    worker_env=TicTacToe(autoplay2=True)
    worker_env.reset()
    worker_optimizer=optim.Adam(lr=0.002,params=worker_model.parameters())
    worker_optimizer.zero_grad()
    for i in range(params['epochs']):
        worker_optimizer.zero_grad()
        values,logprobs,rewards=run_ep(worker_env,worker_model)
        actorLoss,criticLoss,eplen = update_params(worker_optimizer,values,logprobs,rewards)
        counter.value=counter.value +1



class ActorCritic(nn.Module):
    def __init__(self,input,layer1,layer2,layer3,outputActor,outputCritic):
        super(ActorCritic,self).__init__()
        self.l1=nn.Linear(input,layer1)
        self.l2=nn.Linear(layer1,layer2)
        self.actor=nn.Linear(layer2,outputActor)
        self.l3=nn.Linear(layer2,layer3)
        self.critic=nn.Linear(layer3,outputCritic)
    def forward(self,x):
        x=F.normalize(x,dim=0)
        y=F.relu(self.l1(x))
        y=F.relu(self.l2(y))
        actorO=F.log_softmax(self.actor(y),dim=0)
        criticI=F.relu(self.l3(y.detach()))
        criticO=10*t.tanh(self.critic(criticI))
        return actorO,criticO
