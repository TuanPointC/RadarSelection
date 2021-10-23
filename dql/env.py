from random import getstate
import numpy as np
from caculator import * 

class Env:
    def __init__(self,list_radar_TRs,list_radar_RRs,target,delta_t=0.1):
        self.list_radar_TRs=list_radar_TRs
        self.list_radar_RRs=list_radar_RRs
        self.target=target
        self.delta_t=delta_t
        self.step_counter =0
        sum_ind_action = len(self.list_radar_TRs) + len(self.list_radar_RRs)
        self.act_dims =  [2 for _ in range(sum_ind_action)]
        self.state= np.array([target.position["x"],target.position["y"],target.vel["x"],target.vel["y"]]).reshape(4,1)

    # state methods
    def getState(self):
        return self.state
    
    def randomState(self):
        p_x= np.random.uniform(low=0,high=1000)
        p_y= np.random.uniform(low=0,high=1000)

        vel_x=np.random.uniform(low=10,high=12)
        vel_y=np.random.uniform(low=10,high=12)
        self.state=np.array([p_x,p_y,vel_x,vel_y]).reshape(4,1)

    def reset(self):
        p_x, p_y, vel_x, vel_y = 0, 0, 10, 10
        # p_x= np.random.uniform(low=0,high=1000)
        # p_y= np.random.uniform(low=0,high=1000)

        # vel_x=np.random.uniform(low=10,high=12)
        # vel_y=np.random.uniform(low=10,high=12)
        return np.array([p_x,p_y,vel_x,vel_y]).reshape(4,1)

    def nextState(self):
        """
        Move to a Next State
        """
        # s(t) = F(t, s)*s(t−1) + v(s)

        F_ts = np.array([1, 0, self.delta_t, 0, 0, 1, 0, self.delta_t, 0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 4)
        s_pre = self.state
        E_v = np.array([self.delta_t**3/3, 0, self.delta_t**2/2, 0, 0, self.delta_t**3/3, 0, self.delta_t**2/2, self.delta_t/2,
                        0, self.delta_t, 0, 0, self.delta_t/2, 0, self.delta_t]).reshape(4, 4)
        v_s = np.random.normal(0, np.std(E_v, axis=1)).reshape(4, 1)

        state_next = (np.dot(F_ts, s_pre) + v_s).reshape(4,1)
        return state_next
    
    # action methods
    def sam_action(self):
        """
        select one action randomly
        action's format [a1, ..., aN,b1,...,bM]
        """
        # print(len(self.list_radar_TRs), len(self.list_radar_RRs))

        a_vec = [0.0 for _ in range(len(self.list_radar_TRs) + len(self.list_radar_RRs))]
        for i in range(len(a_vec)):
            a_vec[i] += np.random.randint(0,2)
        return a_vec


    def action2index(self, action):
        """
        Convert action from nulti-dimension format to index format
        """
        # print(len(action))
        # print(len(self.act_dims))
        # if len(action) != len(self.act_dims):
        #     raise Exception("Shape Error")

        act_idx = 0

        for i in range(len(action)-1, -1, -1):
            act_idx += action[i]*2**(len(action)-i-1)

        return act_idx

    def index2action(self, act_idx):
        """
        Convert action from index format to multi-dimension format
        """
        action = []
        for i in range(len(self.act_dims), 0, -1):
            ai = act_idx % 2
            action.append(float(ai))
            act_idx = (act_idx - ai) / 2

        # action.append(float(act_idx))
        action.reverse()
        return action

    # reward
    def reward(self,q_performance,k):
        A= k*(len(self.list_radar_RRs)+len(self.list_radar_TRs))
        B=1
        sigma= 10
        countOfTR=0
        countOfRR=0
        for i in self.list_radar_TRs:
            if(i.a>0):
                countOfTR+=1

        for i in self.list_radar_RRs:
            if(i.b>0):
                countOfRR+=1
        if(countOfTR ==0 or countOfRR==0):
            return B*-2
        if(q_performance<= sigma and countOfRR>0 and countOfTR>0):
            return A/(countOfTR+countOfRR)
        if (q_performance>sigma and countOfRR>0 and countOfTR>0):
            return -B
    
    def step(self, action):
        a_action = action[:len(self.list_radar_TRs)]
        b_action = action[len(self.list_radar_TRs):]
        caculator = Caculator()
        Q_ab = caculator.Q_performance(self.list_radar_TRs, self.list_radar_RRs, self.target.position, a_action, b_action, h=1)
        sum_a = np.sum(a_action)
        sum_b = np.sum(b_action)
        reward = 0
        done = False
        B = 10
        A = 20
        sigma = 0.1
        if(sum_a==0 or sum_b==0):
            reward = -2*B
        elif (Q_ab > sigma):
            reward = (-1)*B
        else:
            reward = A*1.0/(sum_a+sum_b)
        
        self.step_counter += 1
        stateNext= self.nextState()
       
        if (self.step_counter == 100 or  stateNext[0] > 1000 or stateNext[0] < 0 or stateNext[1] > 1000 or stateNext[1] < 0):
            done = True
            self.step_counter=0

        return stateNext, reward, done
        