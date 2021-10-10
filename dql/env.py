import numpy as np

class Env:
    def __init__(self,list_radar_TRs,list_radar_RRs,target,delta_t=0.1):
        self.list_radar_TRs=list_radar_TRs
        self.list_radar_RRs=list_radar_RRs
        self.target=target
        self.delta=delta_t

        self.state= np.array([target.position.x,target.position.y,target.vel.x,target.vel.y]).reshape(-1,1)

    def getState(self):
        return self.state
    
    def randomState(self):
        p_x= np.random.uniform(low=0,high=1000)
        p_y= np.random.uniform(low=0,high=1000)

        vel_x=np.random.uniform(low=10,high=13)
        vel_y=np.random.uniform(low=10,high=13)
        self.state=np.array([p_x,p_y,vel_x,vel_y])

    def nextState(self):
        

        
    
    
        

