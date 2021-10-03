class Transmitting_Radar:
    def __init__(self, position, p_m, beta, a):
        """
        params - position: float - coordinate
        params - p_m: float - transmit power
        params - beta  : float - bandwidth
        params - a  : int - Is chosen radar
        """

        self.position = position
        self.p_m = p_m
        self.bandwidth = beta
        self.a = a


class Receiving_Radar:
    def __init__(self, position, b):
        """
        params - position: float - coordinate
        params - p_n: float - transmit power
        params - a  : int - Is chosen radar
        """

        self.position = position
        self.b = b


class Target:

    def __init__(self, position, vel):
        self.position = position
        self.vel = vel
