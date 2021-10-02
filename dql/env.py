import math
import numpy as np


class Transmitting_Radar:
    def __init__(self, position, p_m, beta, a):
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
        self.a = a


class Receiving_Radar:
    def __init__(self, position, b):
        """
        params - x_n: float - X coordinate
        params - y_n: float - Y coordinate
        params - p_n: float - transmit power
        params - a  : int - Is chosen radar
        """

        self.position = position
        self.b = b


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
    def __init__(self, list_radar_TR, list_radar_RR, position_target, action_a, action_b):
        self.radar_TRs = list_radar_TR
        self.radar_RRs = list_radar_RR
        self.position_target = position_target
        self.action_a = action_a
        self.action_b = action_b

    def result(self):
        Tr_C_ab = np.array([0, 0, 0, 0]).reshape(2, 2)
        for i in self.radar_TRs:
            for j in self.radar_RRs:
                tmp = SubCaculator(position_target=self.position_target, position_tr=i.position, position_rr=j.position,
                                   bandwidth=i.bandwidth, power=i.p_m, h=1)
                J_mn = tmp.J_mn()
                Tr_C_ab += i.a*j.a*J_mn

        return Tr_C_ab