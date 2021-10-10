import numpy as np

class Env:
    def __init__(self,list_radar_TRs,list_radar_RRs,target,delta_t=0.1):
        self.list_radar_TRs=list_radar_TRs
        self.list_radar_RRs=list_radar_RRs
        self.target=target
        self.delta=delta_t

        self.state= np.array([target.position.x,target.position.y,target.vel.x,target.vel.y]).reshape(-1,1)

    # state methods
    def getState(self):
        return self.state
    
    def randomState(self):
        p_x= np.random.uniform(low=0,high=1000)
        p_y= np.random.uniform(low=0,high=1000)

        vel_x=np.random.uniform(low=10,high=12)
        vel_y=np.random.uniform(low=10,high=12)
        self.state=np.array([p_x,p_y,vel_x,vel_y])

    def nextState(self):
        """
        Move to a Next State
        """
        # s(t) = F(t, s)*s(t−1) + v(s)

        F_ts = np.array([1, 0, self.denta_t, 0, 0, 1, 0, self.denta_t, 0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 4)
        s_pre = self.state
        E_v = np.array([self.denta_t**3/3, 0, self.denta_t**2/2, 0, 0, self.denta_t**3/3, 0, self.denta_t**2/2, self.denta_t/2,
                        0, self.denta_t, 0, 0, self.denta_t/2, 0, self.denta_t]).reshape(4, 4)
        v_s = np.random.normal(0, np.std(E_v, axis=1)).reshape(4, 1)

        state_next = (np.dot(F_ts, s_pre) + v_s).reshape(4,1)
        self.state=state_next
    
    # action methods
    def sam_action(self):
        """
        select one action randomly
        action's format [a1, ..., aN,b1,...,bM]
        """
        a = np.random.randint(1, len(self.list_radar_TRs))
        b = np.random.randint(1, len(self.list_radar_RRs))

        a_vec = [0.0 for _ in range(len(self.list_radar_TRs))]
        b_vec = [0.0 for _ in range(len(self.list_radar_RRs))]

        sel_helper_idxs_a = np.random.permutation(len(self.list_radar_TRs))[0:a]
        sel_helper_idxs_b = np.random.permutation(len(self.list_radar_TRs))[0:b]

        for helper_idx in list(sel_helper_idxs_a):
            a_vec[helper_idx] = 1.0

        for helper_idx in list(sel_helper_idxs_b):
            b_vec[helper_idx] = 1.0

        action = np.array([a_vec,b_vec]).reshape(1,-1)
        return action




        
    
    
        

