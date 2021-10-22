import json
import numpy as np
from env import Env
from algorithm import Backbone,DQL
import json
from entities import *


def main():

    #read config file
    with open('./config/config1.json', "r") as file:
        config = json.load(file)

    #create list radars
    M = config["M"]
    N = config["N"]
    listRadarRRs=[]
    listRadarTRs=[]
    for i in range(1,M+1):
        position = {"x":config["xT{}".format(i)],"y":config["yT{}".format(i)]}
        p_m= config["p{}".format(i)]
        beta = config["beta{}".format(i)]
        a=0
        newTranmittingRadar= Transmitting_Radar( position,p_m,beta,a)
        listRadarTRs.append(newTranmittingRadar)

    for i in range(1,N+1):
        position = {"x":config["xR{}".format(i)],"y":config["yR{}".format(i)]}
        b=0
        newRecievingRadar= Receiving_Radar( position,b)
        listRadarRRs.append(newRecievingRadar)

    #create target radar
    position = {"x": np.random.uniform(0,10000),"y": np.random.uniform(0,1000)}
    vel = {"x": np.random.uniform(10,12),"y": np.random.uniform(10,12)}
    target = Target(position,vel)

    #create env
    env=Env(listRadarTRs,listRadarRRs,target)

    #create model
    input_dims = 4
    num_actions= 2**(M+N)
    output_dims = M+N
    backbone= Backbone(output_dims)
    model = DQL(input_dims,num_actions,backbone)
    print(model)


if __name__ == "__main__":
    main()
