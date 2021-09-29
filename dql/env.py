class Transmitting_Radar:
    def __init__(self,x_m,y_m,p_m,beta,a):
        """
        params - x_m: float - X coordinate
        params - y_m: float - Y coordinate
        params - p_m: float - transmit power
        params - beta  : float - bandwidth
        params - a  : int - Is chosen radar
        """
        if(p_m<0 and beta<0):
            raise Exception("Initial Values for Transmitting radar must be Positive!")
        
        self.x_m=x_m
        self.y_m=y_m
        self.p_m=p_m
        self.beta=beta
        self.a=a

class Receiving_Radar:
    def __init__(self,x_n,y_n,b):
        """
        params - x_n: float - X coordinate
        params - y_n: float - Y coordinate
        params - p_n: float - transmit power
        params - a  : int - Is chosen radar
        """
    
        self.x_m=x_n
        self.y_m=y_n
        self.b=b

