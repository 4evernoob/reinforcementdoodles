import numpy as np
class TicTacToe():
    def __init__(self):#player is who are you X or O

        self.board=np.zeros((3,3))

        #self.board=np.arange(9).reshape(3,3)#(3,3))
        #self.you=player
    def av_moves(self):
        tmp=self.board.copy().reshape(1,9)
        #print(tmp.shape)
        return [i for i in range(tmp.shape[1]) if tmp[0,i]!=0],[i for i in range(tmp.shape[1]) if tmp[0,i]==0]
    def move(self,move,playerv):
        player = -.5 if playerv==1 else .5
        i,j = int(move/3),move%3
        #print(i, j)
        moved=False
        if (self.board[i,j]==0):
            self.board[i,j]=player
            moved=True
        return self.board, self.game_status(playerv), moved
    def render(self,clean=False):
        if not clean:
            return self.board+np.random.rand(self.board.shape[0],self.board.shape[1])/100.00
        else:
            return self.board.copy()
    def game_status(self,playerv):
        player = -.5 if playerv==1 else .5
        flag=False
        #print(self.board)
        for i in range(self.board.shape[0]):
            #print(self.board[i,:])
            #print(np.all(self.board[i,:] == player))
            if np.all(self.board[i,:] == player):
                return 1
        for j in range(self.board.shape[1]):
            #print(self.board[:,j])
            #print(np.all(self.board[:,j] == player))
            if np.all(self.board[:,j] == player):
                return 1

        tmp=self.board.copy().reshape(1,9)
        if np.all(tmp[0,[0,4,8]]==player):
            return 1
        if np.all(tmp[0,[2,4,6]]==player):
            return 1
        if np.all(self.board!=0):
            return -2

        return 0
    def isfull(self):
        return np.all(self.board!=0)

#if __name__ == '__main__':
# game= TicTacToe()
# game.board[0,:]=np.ones(3)
# print(game.game_status(1))
# print(game.game_status(2))
# print(game.render())
