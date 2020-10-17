import numpy as np
import random
class TicTacToe():
    win=1
    movesav=0#-.10
    danger=-1
    tie=.5
    lose=-.0000001
    notmoving =-10.0
    def isdangervector(self,vect,player):
        player1=-.5
        player2=.5
        if player==player1:
            nposx = np.sum([1 for item in vect == player2 if item])
            nposy = np.sum([1 for item in vect == player1 if item])
            if nposx==2 and nposy==0:
                self.player1=self.danger
        elif player==player2:
            nposx = np.sum([1 for item in vect == player1 if item])
            nposy = np.sum([1 for item in vect == player2 if item])
            if nposx==2 and nposy==0:
                self.player2=self.danger

    def isindanger(self,player):
        #print('+-'*30)
        #print(player)
        #print(self.board)
        for i in range(self.board.shape[0]):
            self.isdangervector(self.board[i,:],player)
            self.isdangervector(self.board[:,i],player)

        #diagonal check for play1 and play2
        tmp=self.board.copy().reshape(1,9)
        self.isdangervector(tmp[0,[0,4,8]],player)
        self.isdangervector(tmp[0,[2,4,6]],player)

    def printp1(self):
        if self.player1==self.win:
            print('P1 won')
        elif self.player1==self.tie:
            print('P1 and P2 tied')
        else:
            print('P1 lost')
    def isgameover(self):
        return (True if self.player1 in self.endingcriteria() else False) or (True if self.player2 in self.endingcriteria() else False)

    def endingcriteria(self):
        return [self.win,self.tie,self.lose,self.notmoving]
    def reset(self):
        self.__init__()
    def __init__(self,autoplay2=False):#player is who are you1= X or2= O
        self.board=np.zeros((3,3))
        self.player1=0
        self.player2=0
        self.autoplay = autoplay2

    def av_moves(self):
        tmp=self.board.copy().reshape(1,9)
        #print(tmp.shape)
        return [i for i in range(tmp.shape[1]) if tmp[0,i]!=0],[i for i in range(tmp.shape[1]) if tmp[0,i]==0]

    def move(self,move,playerv):
        player = -.5 if playerv==1 else .5
        i,j = int(move/3),move%3
        #print(i, j)
        moved=False
        if not self.isgameover():
            if (self.board[i,j]==0):
                self.board[i,j]=player
                moved=True
            self.game_status()
            # invalid moves
            if not moved:
                if playerv==1:
                    self.player1=self.notmoving
                elif playerv==2:
                    self.player2=self.notmoving
            moves,pos_moves=self.av_moves()
            if moved and len(pos_moves)>0 and self.autoplay:
                machine=random.choice(pos_moves)
                self.board[int(machine/3),machine%3]=.5
        return self.render().reshape(1,9),self.player1,self.player2, moved

    def render(self,clean=False):
        if not clean:
            return self.board+np.random.rand(self.board.shape[0],self.board.shape[1])/100.00
        else:
            return self.board.copy()

    def game_status(self):
        player1 = -.5
        player2 = .5
        #print(self.board)
        #check play1
        for i in range(self.board.shape[0]):
            #print(self.board[i,:])
            #print(np.all(self.board[i,:] == player))
            if np.all(self.board[i,:] == player1):
                self.player1 = self.win
                self.player2 = self.lose
            if np.all(self.board[:,i] == player1):
                self.player1=self.win
                self.player2=self.lose
        #check play2
        for j in range(self.board.shape[1]):
            #print(self.board[:,j])
            #print(np.all(self.board[:,j] == player))
            if np.all(self.board[2,:] == player2):
                self.player1= self.lose
                self.player2=self.win
            if np.all(self.board[:,j] == player2):
                self.player1=self.lose
                self.player2=self.win
        #diagonal check for play1 and play2
        tmp=self.board.copy().reshape(1,9)
        if np.all(tmp[0,[0,4,8]]==player1):
            self.player1=self.win
            self.player2=self.lose
        if np.all(tmp[0,[2,4,6]]==player1):
            self.player1=self.win
            self.player2=self.lose
        if np.all(tmp[0,[0,4,8]]==player2):
            self.player1=self.lose
            self.player2=self.win
        if np.all(tmp[0,[2,4,6]]==player2):
            self.player1=self.lose
            self.player2=self.win
        #tie checker
        if self.isfull():
            self.player1=self.player2=self.tie
        if not self.isgameover():
            self.player1=self.player2= self.movesav
            self.isindanger(player1)
            self.isindanger(player2)
    def isfull(self):
        return np.all(self.board!=0)

#if __name__ == '__main__':
# game= TicTacToe()
# game.board[0,:]=np.ones(3)
# print(game.game_status(1))
# print(game.game_status(2))
# print(game.render())
