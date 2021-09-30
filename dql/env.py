import math


class Transmitting_Radar:
    def __init__(self, x_m, y_m, p_m, beta, a):
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

        self.x_m = x_m
        self.y_m = y_m
        self.p_m = p_m
        self.beta = beta
        self.a = a


class Receiving_Radar:
    def __init__(self, x_n, y_n, b):
        """
        params - x_n: float - X coordinate
        params - y_n: float - Y coordinate
        params - p_n: float - transmit power
        params - a  : int - Is chosen radar
        """

        self.x_m = x_n
        self.y_m = y_n
        self.b = b


class FunctionTest:

    def distance(self,position_radar, position_target):
        return math.sqrt((position_target.x-position_radar.x)**2+(position_target.y-position_radar.y)**2)

    def loss_efficients(self,position_target,position_tr,position_rr):
        distance_tr=self.distance(position_tr,position_target)
        distance_rr=self.distance(position_rr,position_target)
        return 1/(distance_rr*distance_tr)**2
    
    def e_m(bandwidth):
        c=3*10**8
        noise_2=-110
        return (8*math.pi**2*bandwidth**2)/(noise_2*c**2)
    

    
    def o_mn(self,power,h,bandwidth):
        return self.loss_efficients()*power*self.e_m(bandwidth)*h**2*()

    #def TRC(action, position_tr, position_rr, position_target, m, n):
