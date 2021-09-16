import math 
import random
import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd
from tensorflow import keras
import tensorflow as tf
class RobotArmEnvV2():

    def __init__(self):
 
        self.set_link_properties([100,100,100,100])
        self.set_increment_rate(0.0174533)
        self.p = 0
        self.r = 0
        self.re = []
        self.target_pos = self.generate_random_pos()
        self.action = {0:"0000",
                    1:"0001",
                    2:"0002",
                    3:"0010",
                    4:"0011",
                    5:"0012",
                    6:"0020",
                    7:"0021",
                    8:"0022",
                    9:"0100",
                    10:"0101",
                    11:"0102",
                   12:"0110",
                    13:"0111",
                    14:"0112",
                    15:"0120",
                    16:"0121",
                    17:"0122",
                    18:"0200",
                    19:"0201",
                    20:"0202",
                    21:"0210",
                    22:"0211",
                    23:"0212",
                    24:"0220",
                    25:"0221",
                    26:"0222",
                    27:"1000",
                    28:"1001",
                    29:"1002",
                    30:"1010",
                    31:"1011",
                    32:"1012",
                    33:"1020",
                    34:"1021",
                    35:"1022",
                    36:"1100",
                    37:"1101",
                    38:"1102",
                    39:"1110",
                    40:"1111",
                    41:"1112",
                    42:"1120",
                    43:"1121",
                    44:"1122",
                    45:"1200",
                    46:"1201",
                    47:"1202",
                    48:"1210",
                    49:"1211",
                    50:"1212",
                    51:"1220",
                    52:"1221",
                    53:"1222",
                    54:"2000",
                    55:"2001",
                    56:"2002",
                    57:"2010",
                    58:"2011",
                    59:"2012",
                    60:"2020",
                    61:"2021",
                    62:"2022",
                    63:"2100",
                    64:"2101",
                    65:"2102",
                    66:"2110",
                    67:"2111",
                    68:"2112",
                    69:"2120",
                    70:"2121",
                    71:"2122",
                    72:"2200",
                    73:"2201",
                    74:"2202",
                    75:"2210",
                    76:"2211",
                    77:"2212",
                    78:"2220",
                    79:"2221",
                    80:"2222"}



        

        self.current_error = -math.inf

        self.viewer = None

    def set_link_properties(self, links):
        self.links = links
        self.n_links = len(self.links)
        self.min_theta1 = math.radians(0)
        self.min_theta2= math.radians(50)
        self.min_theta3 = math.radians(75)
        self.min_theta4 = math.radians(20)

        self.max_theta1 = math.radians(300)
        self.max_theta2 = math.radians(130)
        self.max_theta3 = math.radians(150)
        self.max_theta4 = math.radians(210)

        self.theta = self.generate_random_angle()
        self.max_length = sum(self.links)

    def set_increment_rate(self, rate):
        self.rate = rate



    def forward_kinematics(self, theta):

        theta_1= math.degrees(theta[0])
        theta_2= math.degrees(theta[1])
        theta_3= math.degrees(theta[2])
        theta_4= math.degrees(theta[3])
        a1 = 108 # Length of link 1
        a2 = 146 # Length of link 2
        a3 = 40 # Length of link 3
        a4 = 146
        a5 = 160 # Length of link 4

                #(theta,alpha,a,d)
        d_h_table = np.array([[np.deg2rad(theta_1), np.deg2rad(-90), 0, a1],
                            [np.deg2rad(theta_2-180), np.deg2rad(90), a2, 0],
                            [0, np.deg2rad(90), 0, -a3],
                            [np.deg2rad(theta_3 - 175), 0, a4,0],
                            [np.deg2rad(theta_4 - 105), 0, a5,0]]) 

        
        # Homogeneous transformation matrix from frame 0 to frame 1
        i = 0
        homgen_0_1 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                            [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                            [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                            [0, 0, 0, 1]])  
        
        # Homogeneous transformation matrix from frame 1 to frame 2
        i = 1
        homgen_1_2 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                            [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                            [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                            [0, 0, 0, 1]])  
        
        # Homogeneous transformation matrix from frame 2 to frame 3
        i = 2
        homgen_2_3 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                            [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                            [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                            [0, 0, 0, 1]])  
        
        # Homogeneous transformation matrix from frame 3 to frame 4
        i = 3
        homgen_3_4 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                            [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                            [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                            [0, 0, 0, 1]])  

        i = 4
        homgen_4_5 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                        [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                        [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                        [0, 0, 0, 1]])  
        
        homgen_0_5 = homgen_0_1 @ homgen_1_2 @ homgen_2_3 @ homgen_3_4 @ homgen_4_5 
        
        
        # Print the homogeneous transformation matrices

        return [homgen_0_5[0][3]/10, homgen_0_5[1][3]/10, homgen_0_5[2][3]/10]

    
    def generate_random_angle(self):
        theta = np.zeros(self.n_links)
        theta[0] = math.radians(random.uniform(50, 250))
        theta[1] = math.radians(90)
        theta[2] = math.radians(100)
        theta[3] = math.radians(random.uniform(50,120))
        return theta

    def generate_random_pos(self):
        theta = self.generate_random_angle()
        self.theta = theta
        self.p = random.uniform(0,300)
        self.r = random.uniform(15,30)
        pos = np.array([self.r* math.cos(math.radians(self.p)), self.r* math.sin(math.radians(self.p)), 0])
        return pos


    def step(self, action):
        a = list(self.action[action])
        for n in range(len(a)):
            if a[n] == '2':
                self.theta[n] = self.theta[n] - self.rate
            elif a[n] == '1':
                self.theta[n] = self.theta[n] + self.rate
            else:
                continue


        P = self.forward_kinematics(self.theta)
        tip_pos = P
  

        pos = tip_pos

        target = self.target_pos
        pt = self.p
        pp = math.degrees(self.theta[0])

        diffp = math.radians(abs(pp-pt))

        rt = math.sqrt(target[1]**2 + target[0]**2)  
        rp = math.sqrt(pos[1]**2 + pos[0]**2) 
        diffs = min(rt,rp)*diffp
        diffz = abs(pos[2]-target[2])
        diffr = abs(rp-rt)
        diffh = math.sqrt(diffz**2 + diffr**2) 
        distance_error = diffh + diffs

        reward = 0
        if distance_error >= self.current_error:
            reward = -1
        epsilon = 10
 
        if distance_error < self.current_error:
            reward = 1

        if len(self.re) == 10:
            f = np.roll(self.re.copy(),-1)
            f[-1] = reward
            self.re = f
        else:
            self.re.append(reward)
        self.current_error = distance_error
        self.current_score += reward

        if self.current_score == -50 or self.current_score == 300 or self.current_error <= 1 or sum(self.re) == -10:
            done = True
        else:
            done = False

        observation = np.hstack((self.target_pos, self.theta))
        info = {
            'distance_error': distance_error,
            'target_position': self.target_pos,
            'current_position': tip_pos
        }
        return observation, reward, done, info

    def reset(self):
        self.target_pos = self.generate_random_pos()
        self.current_score = 0
        self.re = []
        observation = np.hstack((self.target_pos, self.theta))
        return observation
class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = np.empty(max_size, dtype=np.object)
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.randint(self.size, size=batch_size)
        return self.buffer[indices]
class QNetwork(tf.Module):
    def __init__(self, lr=1e-3, input_shape = [7], n_outputs = 81):
        super(QNetwork, self).__init__()
        #self.optimizer = keras.optimizers.Adam(lr=lr)
        #self.loss = keras.losses.mean_squared_error
        self.optimizer = keras.optimizers.Adam(lr=1e-3)
        self.loss = keras.losses.Huber()
        self.model = self.get_model()

    def get_model(self):
        input_shape = [7]
        n_outputs =  81
        K = keras.backend
        input_states = keras.layers.Input(shape=[7])
        hidden1 = keras.layers.Dense(128, activation="elu")(input_states)
        hidden2 = keras.layers.Dense(128, activation="elu")(hidden1)
        hidden3 = keras.layers.Dense(128, activation="elu")(hidden2)
        hidden4 = keras.layers.Dense(128, activation="elu")(hidden3)
        hidden5 = keras.layers.Dense(128, activation="elu")(hidden4)
        state_values = keras.layers.Dense(1)(hidden5)
        raw_advantages = keras.layers.Dense(n_outputs)(hidden5)
        advantages = raw_advantages - K.max(raw_advantages, axis=1, keepdims=True)
        Q_values = state_values + advantages
        model = keras.models.Model(inputs=[input_states], outputs=[Q_values])
        return model
class DQNAgent(object):
    def __init__(self):
        self.epsilon = 1
        self.replay_memory = ReplayMemory(max_size=10000000)
        self.batch_size = 128
        self.discount_rate = 0.95
        self.network = QNetwork()
        self.model = self.network.model
        self.target = keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())
        self.n_outputs = 81
        self.input_shape = [7]
        self.optimizer = self.network.optimizer
        self.loss_fn = self.network.loss

    def limit_check(self,df,env):
        theta = env.theta.copy()
        a = list(env.action[df['index']])
        for n in range(len(a)):
            if a[n] == '2':
                theta[n] = theta[n] - env.rate
            elif a[n] == '1':
                theta[n] = theta[n] + env.rate
            else:
                continue
        if theta[0] > 300 or theta[0] < 0 : 
            return True
        return False

    def limit_check2(self,df,env):
        theta = env.theta.copy()
        a = list(env.action[df])
        for n in range(len(a)):
            if a[n] == '2':
                theta[n] = theta[n] - env.rate
            elif a[n] == '1':
                theta[n] = theta[n] + env.rate
            else:
                continue
        if theta[0] > 300 or theta[0] < 0: 
            return True
        return False

    def check_action(self,dfx, env):
        collision = True
        i = 0
        dfx = dfx.sort_values(by = 0,ascending=False)
        dfx = dfx.reset_index()

        while collision == True:
            collision = self.limit_check(dfx.iloc[i], env)
            if collision == True:
                i +=1
        return dfx.iloc[i]['index']

    def random_action(self,ex):
        rans = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
                60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80]
        for r in ex:
          rans.pop(rans.index(r))
        return random.choice(rans)

    def epsilon_greedy_policy(self, state, env, epsilon=0):
        
        if np.random.rand() < epsilon:
            collision = True
            actions = []
            while collision == True:
                action = self.random_action(actions)
                collision = self.limit_check2(action,env)
                if collision == True:
                    actions.append(action)

            return int(action)
        else:
            Q_values = self.model.predict(state[np.newaxis])

            dfx = pd.DataFrame(Q_values[0])
            action = self.check_action(dfx, env)

            return int(action)


    def sample_experiences(self, batch_size):
        indices = self.replay_memory.sample(batch_size)
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in indices])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones  

    def play_one_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state,env, epsilon )
        next_state, reward, done, info = env.step(action)
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, info
    
    def training_step(self, batch_size):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = self.model.predict(next_states)
        best_next_actions = np.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()
        next_best_Q_values = (self.target.predict(next_states) * next_mask).sum(axis=1)
        target_Q_values = (rewards + 
                        (1 - dones) * self.discount_rate * next_best_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
env = RobotArmEnvV2()
np.random.seed(42)
tf.random.set_seed(42)
input_shape = [7]
n_outputs =  81
training_rewards = []
best_score = -1000
agent = DQNAgent()


def fl(f):
    floats_list = []
    for item in f.split():
        floats_list.append(float(item))

    return floats_list
action = {0:"0000",
                    1:"0001",
                    2:"0002",
                    3:"0010",
                    4:"0011",
                    5:"0012",
                    6:"0020",
                    7:"0021",
                    8:"0022",
                    9:"0100",
                    10:"0101",
                    11:"0102",
                   12:"0110",
                    13:"0111",
                    14:"0112",
                    15:"0120",
                    16:"0121",
                    17:"0122",
                    18:"0200",
                    19:"0201",
                    20:"0202",
                    21:"0210",
                    22:"0211",
                    23:"0212",
                    24:"0220",
                    25:"0221",
                    26:"0222",
                    27:"1000",
                    28:"1001",
                    29:"1002",
                    30:"1010",
                    31:"1011",
                    32:"1012",
                    33:"1020",
                    34:"1021",
                    35:"1022",
                    36:"1100",
                    37:"1101",
                    38:"1102",
                    39:"1110",
                    40:"1111",
                    41:"1112",
                    42:"1120",
                    43:"1121",
                    44:"1122",
                    45:"1200",
                    46:"1201",
                    47:"1202",
                    48:"1210",
                    49:"1211",
                    50:"1212",
                    51:"1220",
                    52:"1221",
                    53:"1222",
                    54:"2000",
                    55:"2001",
                    56:"2002",
                    57:"2010",
                    58:"2011",
                    59:"2012",
                    60:"2020",
                    61:"2021",
                    62:"2022",
                    63:"2100",
                    64:"2101",
                    65:"2102",
                    66:"2110",
                    67:"2111",
                    68:"2112",
                    69:"2120",
                    70:"2121",
                    71:"2122",
                    72:"2200",
                    73:"2201",
                    74:"2202",
                    75:"2210",
                    76:"2211",
                    77:"2212",
                    78:"2220",
                    79:"2221",
                    80:"2222"}
alist = list(action.values())


import pandas as pd
df = pd.read_csv("history_positions.csv")
del df["Unnamed: 0"]
df = df[df['0'] != df['1']]
df = df.reset_index(drop=True)
df['0'] = df['0'].apply(lambda x: x.replace('\n', ' '))
df['0'] = df['0'].apply(lambda x: x.replace('[', ' '))
df['0'] = df['0'].apply(lambda x: x.replace(']', ' '))
df['1'] = df['1'].apply(lambda x: x.replace('\n', ' '))
df['1'] = df['1'].apply(lambda x: x.replace('[', ' '))
df['1'] = df['1'].apply(lambda x: x.replace(']', ' '))
df['start'] = df['0'].apply(lambda x: fl(x))
df['end'] = df['1'].apply(lambda x: fl(x))
df['a1'] = df['start'].apply(lambda x: x[3:])
df['a2'] = df['end'].apply(lambda x: x[3:])
df['action'] = df.apply(lambda row: [   int((row['a2'][0]- row['a1'][0])/max(abs(row['a2'][0]- row['a1'][0]),0.001))  , 
int((row['a2'][1]-row['a1'][1])/max(abs(row['a2'][1]-row['a1'][1]),0.001)), 
int((row['a2'][2]-row['a1'][2])/max(abs(row['a2'][2]-row['a1'][2]),0.001)), 
int((row['a2'][3]-row['a1'][3])/max(abs(row['a2'][3]-row['a1'][3]),0.001))], 
axis = 1)
df['step'] = df['action'].apply(lambda x: ''.join(str(e) for e in x))
df['step'] = df['step'].apply(lambda x: x.replace("-1","2"))
df['action'] = df['step']
df['action'] = df['action'].apply(lambda x: alist.index(x))
df['start'] = df['start'].apply(lambda x: [x[0],x[1],x[2],math.radians(x[3]),math.radians(x[4]),math.radians(x[5]),math.radians(x[6]) ])
df['end'] = df['end'].apply(lambda x: [x[0],x[1],x[2],math.radians(x[3]),math.radians(x[4]),math.radians(x[5]),math.radians(x[6]) ])
df['reward'] = 1
df['done'] = False
df['start'] = df['start'].apply(lambda x: np.array(x).astype("float32"))
df['end'] = df['end'].apply(lambda x: np.array(x).astype("float32"))
df['reward'] = df['reward'].apply(lambda x: np.array(x).astype("float32"))
df['action'] = df['action'].apply(lambda x: np.array(x).astype("float32"))
df['done'] = df['done'].apply(lambda x: np.array(x).astype("float32"))
df = df.loc[df.index.repeat(25)].sample(frac=1)
df.apply(lambda row: agent.replay_memory.append((row['start'], row['action'], row['reward'], row['end'], row['done'])), axis = 1 )










for episode in range(5000):
    obs = env.reset()    
    e_reward = 0

    for step in range(300):
        agent.epsilon = max(1 - episode / 4000, 0.01)
        obs, reward, done, info = agent.play_one_step(env, obs, agent.epsilon)
        e_reward += reward
        if done:
            break
    training_rewards.append(e_reward)
    if e_reward > best_score:
        best_weights = agent.model.get_weights()
        agent.model.save('human_model_43021_727')
        best_score = e_reward
    print("\rEpisode: {}, Reward: {}, eps: {:.3f}".format(episode, e_reward, agent.epsilon), end="")
    if episode > 500:
        agent.training_step(agent.batch_size)
    if episode % 500 == 0 and episode >= 500:
        agent.target.set_weights(agent.model.get_weights())

agent.model.set_weights(best_weights)

df = pd.DataFrame(training_rewards)
df.to_csv("training_human_model_43021_727.csv")