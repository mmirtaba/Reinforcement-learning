from matplotlib import pyplot as plt
import numpy as np

##### set parameters #####
plane_size = 7
plane = [[1 for j in range(plane_size) ] for i in range(plane_size)]
Goal = [4,6]
epoch  = 1000
Gamma = 0.9
Epsilon = 0.99
alpha = 0.1

##### define grid world map #####
plane[0][1] = 0
plane[1][3:5] = [0,0]
plane[1][6] = 0
plane[2][0:3] = [0,0,0]
plane[2][6] = 0
plane[3][4:6] = [0,0]
plane[4][1:4] = [0,0,0]
plane[5][1] = 0
plane[6][3] = 0

print("The plane is:")
for j in range(plane_size):
    print(plane[j])

print("The goal is:")
All_states = []
for i in range(plane_size):
    for j in range(plane_size):
        if plane[i][j] == 1:
            All_states.append([i,j])
print("Free states are:")
print(All_states)

n_states = len(All_states)
print("Number of free states: ")
print(n_states)
All_actions = [[1,0], [-1,0], [0,1], [0,-1]] # Down Up Right Left
n_actions = len(All_actions)
print("Number of actions: ")
print(n_actions)
policy = [0 for i in range(n_states)]

def environment(all_states, state, action, goal):
    new_state = [0,0]
    new_state[0] = state[0] + action[0]
    new_state[1] = state[1] + action[1]
    if new_state in all_states:
       if new_state == goal:
           reward = 10
           return new_state, reward
       else:
           reward = -1
           return new_state, reward
    else:
        new_state = state
        reward = -10
        return new_state, reward



def find_action(all_sa,state,q,epsilon,all_a):
    if np.random.rand() < epsilon:
       index_action = np.random.randint(0,4)
       return all_a[index_action]
    else:
        index = all_sa.index([state,[1,0]])
        my_set_of_actions = [q[index],q[index+1],q[index+2],q[index+3]]
        index_action = my_set_of_actions.index(max(my_set_of_actions))
        return all_a[index_action]


all_action_state_pairs = []
for i in All_states:
    for j in All_actions:
        all_action_state_pairs.append([i,j])
# print("All state-actions are: ")
# print(all_action_state_pairs)
Q = [0 for i in all_action_state_pairs]

##### Q learning #####
for _ in range(epoch):
    index_of_initial_state = np.random.randint(0, n_states)
    initial_state = All_states[index_of_initial_state]
    new_st = initial_state

    while new_st!=Goal:
          action_to_take = find_action(all_action_state_pairs, initial_state, Q, Epsilon, All_actions)
          new_st, reward = environment(All_states, initial_state, action_to_take, Goal)
          index_of_q = all_action_state_pairs.index([initial_state, action_to_take])
          index_of_q_new_st = all_action_state_pairs.index([new_st, [1,0]])
          my_set = [Q[index_of_q_new_st],Q[index_of_q_new_st+1],Q[index_of_q_new_st+2],Q[index_of_q_new_st+3]]
          Max_q = max(my_set)
          Q[index_of_q] = Q[index_of_q] + alpha*(reward + Gamma*Max_q - Q[index_of_q])
          initial_state = new_st
    Epsilon = Epsilon * 0.99



##### find optimal policy with Q #####

for i in range(n_states-1):
    index = all_action_state_pairs.index([All_states[i], [1, 0]])
    my_set_of_actions = [Q[index], Q[index + 1], Q[index + 2], Q[index + 3]]
    index_action = my_set_of_actions.index(max(my_set_of_actions))
    policy[i] = index_action



##### plot results #####

all_actions_str = ['Down','Up','Right','Left']

print("Policy for each state is:")
for i in range(n_states-1):
    print(f'{All_states[i]}--{all_actions_str[policy[i]]}')

plane2 = plane
plane2[Goal[0]][Goal[1]] = 5
stt = [0,0]
for j in range(40):


    plane2[stt[0]][stt[1]] = 3
    plt.imshow(plane2, extent=[0, 7, 7, 0])
    if stt == Goal:
        break
    idx2 = All_states.index(stt)
    a = All_actions[policy[idx2]]
    stt, ree = environment(All_states, stt, a, Goal)
    plt.pause(.3)
    plt.draw()

plt.show()
