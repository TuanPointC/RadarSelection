import math
import numpy as np

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get(self):
        return self

class Transmitting_Radar:
    def __init__(self, position, p_m, beta):
        """
        params - x_m: float - X coordinate
        params - y_m: float - Y coordinate
        params - p_m: float - transmit power
        params - beta  : float - bandwidth
        params - a  : int - Is chosen radar
        """
        if(p_m < 0 and beta < 0):
            raise Exception(
                "Initial Values for Transmitting radar must be Positive!")

        self.position = position
        self.p_m = p_m
        self.bandwidth = beta

class Receiving_Radar:
    def __init__(self, position):
        """
        params - x_n: float - X coordinate
        params - y_n: float - Y coordinate
        params - p_n: float - transmit power
        """
        self.position = position

class Target:
    def __init__(self, position, v_x, v_y):
        self.position = position
        self.v_x = v_x
        self.v_y = v_y
        self.position_start = position
        self.v_x_start = v_x
        self.v_y_start = v_y

    def transit(self):
        """
        Move to a Next State
        """
        # s(t) = F(t, s)*s(t−1) + v(s)

        denta_t = 0.1
        F_ts = np.array([1, 0, denta_t, 0, 0, 1, 0, denta_t, 0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 4)
        s_pre = np.array([self.position.x, self.position.y, self.v_x, self.v_y]).reshape(4, 1)
        E_v = np.array([denta_t**3/3, 0, denta_t**2/2, 0, 0, denta_t**3/3, 0, denta_t**2/2, denta_t/2,
                        0, denta_t, 0, 0, denta_t/2, 0, denta_t]).reshape(4, 4)
        v_s = np.random.normal(0, np.std(E_v, axis=1)).reshape(4, 1)

        state_next = (np.dot(F_ts, s_pre) + v_s).reshape(1, 4)

        self.position = Position(state_next[0], state_next[1])
        self.v_x, self.v_y = state_next[2], state_next[3]

    def show_cur_state(self):
        print("x: {:.3f}, y: {:.3f}, v_x: {:.10f}, v_y: {:.10f}".format(self.position.x, self.position.y, self.v_x, self.v_y))

    def get_state(self):
        """
        Get the Current State of This Target
        """
        state = [self.position, self.v_x, self.v_y]
        return state

    def reset(self):
        """
        Create a New Instance
        """
        low = 0
        high = 1000
        self.position.x = np.random.uniform(low=low, high=high)
        self.position.y = np.random.uniform(low=low, high=high)
        v_min, v_max = 10, 12
        self.v_x = np.random.uniform(low=v_min, high=v_max)
        self.v_y = np.random.uniform(low=v_min, high=v_max)

class SubCaculator:
    def __init__(self, position_target, position_tr, position_rr, bandwidth, power, h):
        self.position_target = position_target
        self.position_tr = position_tr
        self.position_rr = position_rr
        self.bandwidth = bandwidth
        self.power = power
        self.h = h

    def distance(self,position_radar):
        return math.sqrt((self.position_target.x-position_radar.x)**2+(self.position_target.y-position_radar.y)**2)

    # alpha m, n
    def loss_efficients(self):
        distance_tr=self.distance(self.position_tr, self.position_target)
        distance_rr=self.distance(self.position_rr, self.position_target)
        return 1/(distance_rr*distance_tr)**2
    
    def e_m(self):
        c=3*10**8
        noise_2=-110
        return (8*math.pi**2*self.bandwidth**2)/(noise_2*c**2)

    # hệ số có công thức a_m,n * p_m * e_m * |h_m,n|^2
    def coefficient(self):
        return self.loss_efficients()*self.power*self.e_m()*self.h**2;

    # phân số cho biến x có công thức : (x_radar - x_target)/d_Tm
    def fractions_x(self, position_radar):
        return (position_radar.x - self.position_target.x)/self.distance(position_radar)

    # phân số cho biến y có công thức : (x_radar - x_target)/d_Tm
    def fractions_y(self, position_radar):
        return (position_radar.y - self.position_target.y)/self.distance(position_radar)

    # Công thức (4)
    def o_mn(self):
        return self.coefficient()*(self.fractions_x(self.position_tr) + self.fractions_x(self.position_rr))**2

    # Công thức (5)
    def q_mn(self):
        return self.coefficient()*(self.fractions_y(self.position_tr) + self.fractions_y(self.position_rr))**2

    # Công thức (6)
    def r_mn(self):
        ps_x = self.fractions_x(self.position_tr) + self.fractions_x(self.position_rr)
        ps_y = self.fractions_y(self.position_tr) + self.fractions_y(self.position_rr)
        return self.coefficient()*ps_x*ps_y

    # Ma trận J_mn
    def J_mn(self):
        matrix_J = np.array([self.o_mn(), self.q_mn(), self.q_mn(), self.r_mn()]).reshape(2, 2)
        return matrix_J

class C_ab:
    def __init__(self, list_radar_TR, list_radar_RR, target, action_a, action_b):
        self.radar_TRs = list_radar_TR
        self.radar_RRs = list_radar_RR
        self.position_target = target.position
        self.action_a = action_a
        self.action_b = action_b

    def update_state(self, target, action_a, action_b):
        self.position_target = target.position
        self.action_a = action_a
        self.action_b = action_b

    def result(self):
        Tr_C_ab = np.array([0, 0, 0, 0]).reshape(2, 2)
        index_i = 0
        for i in self.radar_TRs:
            index_j = 0
            for j in self.radar_RRs:
                tmp = SubCaculator(position_target=self.position_target, position_tr=i.position, position_rr=j.position,
                                   bandwidth=i.bandwidth, power=i.p_m, h=1)
                J_mn = tmp.J_mn()
                Tr_C_ab += self.action_a[index_i]*self.action_b[index_j]*J_mn
                index_j += 1

            index_i += 1

        return Tr_C_ab


class RadarSelectionEnv:
    def __init__(self, RRs, TRs, target, alpha1, alpha2, seed=1):
        self.RRs = RRs
        self.TRs = TRs
        self.target = target
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.step_counter = 0

        np.random.seed(seed)

    def get_state(self):
        """
        Get Environment State
        """
        state_target = self.target

        return state_target

    def reset(self):
        """
        Create a New Instance
        """
        self.target.reset()
