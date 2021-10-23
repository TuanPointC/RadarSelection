import json
import os
import numpy as np
from relay import Memory
from env import Env
from algorithm import Backbone,DQL
import json
from entities import *
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import matplotlib.pyplot as plt


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
    position = {"x": np.random.uniform(0,1000),"y": np.random.uniform(0,1000)}
    vel = {"x": np.random.uniform(10,12),"y": np.random.uniform(10,12)}
    target = Target(position,vel)

    #create env
    env=Env(listRadarTRs,listRadarRRs,target)

    #create model
    input_dims = 4
    num_actions= 2**(M+N)
    output_dims = 2**(M+N)
    backbone= Backbone(output_dims)
    model = DQL(input_dims,num_actions,backbone)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_idx"])
    gpu_option = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_option))
    sess.run(tf.compat.v1.global_variables_initializer())

    container = Memory(config["buffer_size"])
    log_total_reward, log_comp_reward, log_cost_reward = [], [], []
    epsilon, lr = config["max_eps"], config["learning_rate"]
    r=[]
    for episode in range(config["num_episodes"]):
        if episode % 50 == 0:
            avg_total = []
            for game in range(10):
                state = env.reset()
                done = False
                total_reward = 0
                    
                while not done:
                    action = model.sel_action(env=env, model= model, sess= sess, state= state, epsilon= 0)

                    next_state, reward, done = env.step(action)
                    state = next_state
                        
                    total_reward += reward
                    # print(done)                  
                avg_total.append(total_reward)
                    
            avg_total = np.mean(avg_total)
                
            log_total_reward.append(avg_total)
            # if (episode) % 100 == 0:
            print("Episode {} - Total Reward {:.5f}".\
                format(episode, avg_total))
            r.append(avg_total)





        env.randomState()
        state = env.getState()
        done = False
        if (episode + 1) % 50 == 0:
            epsilon *= 0.99
        
        epsilon = max(epsilon, config["min_eps"])
        
        while not done:
            action = model.sel_action(env, model, sess, state, epsilon)
            next_state, reward, done = env.step(action)
            container.insert_samples({'s': [state],
                                    'a':[env.action2index(action)],
                                    'ns':[next_state],
                                    'r':[float(reward)],
                                    'd':[float(0.0)]})
            state = next_state
        
        batch_data = container.sel_samples(config["batch_size"])
        states = np.array(batch_data["s"]).reshape(-1,4)
        actions = np.array(batch_data["a"])
        next_states = np.array(batch_data["ns"]).reshape(-1,4)
        rewards = np.array(batch_data["r"])
        dones = np.array(batch_data["d"])

        [_, loss] = sess.run([model.opt, model.loss],
                            feed_dict={
                                model.states: states,
                                model.next_states:next_states,
                                model.actions: actions,
                                model.rewards: rewards,
                                model.dones: dones,
                                model.gamma: config["gamma"],
                                model.lr: lr
                            })
    print("ok")
    plt.plot(r)
    plt.show()


if __name__ == "__main__":
    main()
