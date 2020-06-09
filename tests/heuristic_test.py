from mpl_toolkits.mplot3d import Axes3D
from gym.spaces import Discrete, Tuple
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage.measurements import label
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy

class PackEnv2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_shape = (5, 5, 5), input_shapes=[],max_moves=100, replacement=True):
        self.counter = 0
        self.max_moves = max_moves
        self.done = False
        self.reward = 0
        self.board_shape = board_shape
        self.observation_space = np.zeros((board_shape[0], board_shape[1], board_shape[2]*2))
        self.action_space = Discrete(board_shape[0]*board_shape[1]*board_shape[2]+1)
        self.state = [np.zeros(board_shape),np.zeros(board_shape)]
        self.return_state = np.concatenate((self.state[0], self.state[1]), axis=2)
        self.replace = replacement

        self.num_possible_moves = board_shape[0]*board_shape[1]*board_shape[2]

        if len(input_shapes) == 0:
            mat = np.zeros(board_shape)
            mat[0][0][0] = 1
            self.shapes = [mat]
        else:
            self.shapes = []
            for shape in input_shapes:
                base_mat = np.zeros(board_shape)
                for i in range(len(shape)):
                    for j in range(len(shape[0])):
                        for k in range(len(shape[0][0])):
                            base_mat[i][j][k] = shape[i][j][k]
                self.shapes.append(base_mat)
        self.remaining_shapes = copy.deepcopy(self.shapes)
        val = random.choice(range(len(self.shapes)))
        self.state[1] = self.shapes[val]
        if not self.replace:
            self.remaining_shapes.pop(val)

    def reset(self):
        val = random.choice(range(len(self.shapes)))
        random_shape = self.shapes[val]
        self.counter = 0
        self.done = False
        self.reward = 0
        self.state = [np.zeros(self.board_shape), random_shape]
        self.return_state = np.concatenate((self.state[0], self.state[1]), axis=2)
        self.remaining_shapes = copy.deepcopy(self.shapes)
        if not self.replace:
            self.remaining_shapes.pop(val)
        return self.return_state

    def move_to_vec(self,target):
        h = self.board_shape[0]
        w = self.board_shape[1]
        d = self.board_shape[2]
        return (int(target % (h*w) / w), (target % (h*w)) % w,int(target/(h*w)))
    def vec_to_move(self,vec):
        h = self.board_shape[0]
        w = self.board_shape[1]
        d = self.board_shape[2]
        return h*w*vec[2] + w*vec[0] + vec[1]
    def get_neighbors(self,vec):
        neighbors = []
        h = self.board_shape[0]
        w = self.board_shape[1]
        d = self.board_shape[2]
        if vec[0] -1 >= 0:
            neighbors.append([vec[0]-1, vec[1], vec[2]])
        if vec[0] +1 < h:
            neighbors.append([vec[0]+1, vec[1], vec[2]])
        if vec[1] -1 >= 0:
            neighbors.append([vec[0], vec[1]-1, vec[2]])
        if vec[1] +1 < w:
            neighbors.append([vec[0], vec[1]+1, vec[2]])
        if vec[2] -1 >= 0:
            neighbors.append([vec[0], vec[1], vec[2]-1])
        if vec[2] +1 < d:
            neighbors.append([vec[0], vec[1], vec[2]+1])
        #print(neighbors)
        return neighbors


    def valid_move(self, target):
        state = self.state
        board = state[0]
        #print("board", board)
        piece = state[1]
        #print("piece", piece)
        h = self.board_shape[0]
        w = self.board_shape[1]
        d = self.board_shape[2]

        #do nothing
        if target == h * w * d:
            return True

        if target > h*w*d or target < 0:
            return False

        d_offset = int(target/(h*w))
        h_offset = int(target % (h*w) / w)
        w_offset = (target % (h*w)) % w
        #print(h_offset, w_offset,d_offset)
        #print(board)

        for H in range(len(piece)):
            for W in range(len(piece[0])):
                for D in range(len(piece[0][0])):
                    if piece[H][W][D] == 1:
                        if (h_offset + H >= h) or (w_offset + W  >= w) or (d_offset + D >= d):
                            return False
                        if board[H+h_offset][W+w_offset][D+d_offset] == 1:
                            return False
        return True


    def calculate_reward(self, target, divisor=20):
        state = self.state
        board = state[0]
        h = self.board_shape[0]
        w = self.board_shape[1]
        d = self.board_shape[2]
        if target == self.num_possible_moves:
            return -.5



        #connection structure
        #structure = np.ones((3, 3), dtype=np.int)
        structure = [[[0, 0, 0],[0, 1, 0],[0, 0, 0]], [[0, 1, 0],[1, 1, 1],[0, 1, 0]],[[0, 0, 0],[0, 1, 0],[0, 0, 0]]]
        labeled, ncomponents = label(board, structure)
        vec = self.move_to_vec(target)
        component_num = labeled[vec[0]][vec[1]][vec[2]]
        if component_num == 0:
            #invalid
            return -1
        component = list(list(elt) for elt in np.array(np.where(labeled == 1)).T)

        size = len(component)
        max_h = max([pair[0] for pair in component])
        min_h = min([pair[0] for pair in component])
        max_w = max([pair[1] for pair in component])
        min_w = min([pair[1] for pair in component])
        max_d = max([pair[2] for pair in component])
        min_d = min([pair[2] for pair in component])


        perimeter = 0
        for elt in component:
            neighborhood = self.get_neighbors(elt)
            for neighbor in neighborhood:
                if neighbor not in component:
                    perimeter +=1
        perimeter = max(1, perimeter)
        #print("perimiter is" ,perimeter)
        block_size = abs(max_h-min_h + 1)*abs(max_w-min_w + 1)*abs(max_d-min_d + 1)
        return size**4/block_size/divisor/perimeter
    def pseudo_reward(self,board, target, divisor=20):
        h = self.board_shape[0]
        w = self.board_shape[1]
        d = self.board_shape[2]
        if target == self.num_possible_moves:
            return -.5


        structure = [[[0, 0, 0],[0, 1, 0],[0, 0, 0]], [[0, 1, 0],[1, 1, 1],[0, 1, 0]],[[0, 0, 0],[0, 1, 0],[0, 0, 0]]]
        labeled, ncomponents = label(board, structure)
        vec = self.move_to_vec(target)
        component_num = labeled[vec[0]][vec[1]][vec[2]]
        if component_num == 0:
            #invalid
            return -1
        component = list(list(elt) for elt in np.array(np.where(labeled == 1)).T)

        size = len(component)
        max_h = max([pair[0] for pair in component])
        min_h = min([pair[0] for pair in component])
        max_w = max([pair[1] for pair in component])
        min_w = min([pair[1] for pair in component])
        max_d = max([pair[2] for pair in component])
        min_d = min([pair[2] for pair in component])


        perimeter = 0
        for elt in component:
            neighborhood = self.get_neighbors(elt)
            for neighbor in neighborhood:
                if neighbor not in component:
                    perimeter +=1
        perimeter = max(1, perimeter)
        #print("perimiter is" ,perimeter)
        block_size = abs(max_h-min_h + 1)*abs(max_w-min_w + 1)*abs(max_d-min_d + 1)
        return size**4/block_size/divisor/perimeter

    def merge(self, target):
        state = self.state
        board = copy.deepcopy(state[0])
        piece = state[1]
        h = self.board_shape[0]
        w = self.board_shape[1]
        d = self.board_shape[2]

        #do nothing
        if target == h * w*d:
            return state[0]

        h_offset = int(target / h)
        w_offset = target % w
        vec = self.move_to_vec(target)

        for H in range(len(piece)):
            for W in range(len(piece[0])):
                for D in range(len(piece[0][0])):
                    if piece[H][W][D] == 1:
                        #print("HIIIIIII")
                        board[H+vec[0]][W+vec[1]][D+vec[2]] = 1
        return board

    def final_reward(self, multiplier=10):
        h = self.board_shape[0]
        w = self.board_shape[1]
        d = self.board_shape[2]
        state = self.state
        board = state[0]
        return np.sum(board)*multiplier/(h*w*d)
        #if np.sum(board) == h*w:
        #    return 1
        #else:
        #    return -1


    def step(self, target):
        h = self.board_shape[0]
        w = self.board_shape[1]
        d = self.board_shape[2]
        if self.done == True:
            self.reward = self.final_reward()
            #print("It's over")
            return [self.return_state, self.reward, self.done, {}]
        elif target > self.num_possible_moves:
            print("Impossible. Invalid position")
            return [self.return_state, self.reward, self.done, {}]
        else:
            self.counter+=1
            #print("counter", self.counter)
            if (self.counter == self.max_moves):
                self.done = True
                self.reward = self.final_reward()
                return [self.return_state, self.reward, self.done, {}]
            #self.state[0][int(target/h)][target%k] = 1
            if not self.valid_move(target):
                self.reward = -1
                return [self.return_state, self.reward, self.done, {}]

            updated_board = self.merge(target)
            self.state[0] = updated_board
            self.reward = self.calculate_reward(target)


            #do nothing so same state
            if (target == h*w*d):
                val = random.choice(range(len(self.remaining_shapes)))
                self.state[1] = self.remaining_shapes[val]
                if not self.replace:
                    self.remaining_shapes.pop(val)
                self.return_state = np.concatenate((self.state[0], self.state[1]), axis=2)
                return [self.return_state, self.reward, self.done, {}]
            #no pieces left so we're done
            if len(self.remaining_shapes) == 0:
                #print("hi")
                self.state[1] = np.zeros(self.board_shape)
                self.return_state = np.concatenate((self.state[0], self.state[1]), axis=2)
                self.done = True
                self.reward = self.final_reward()
                return [self.return_state, self.reward, self.done, {}]
            else:
                val = random.choice(range(len(self.remaining_shapes)))
                self.state[1] = self.remaining_shapes[val]
                if not self.replace:
                    self.remaining_shapes.pop(val)
                self.return_state = np.concatenate((self.state[0], self.state[1]), axis=2)
                return [self.return_state, self.reward, self.done, {}]

    def pseudo_step(self, target):
        h = self.board_shape[0]
        w = self.board_shape[1]
        d = self.board_shape[2]
        if self.done == True:
            reward = self.final_reward()
            #print("It's over")
            return [self.return_state, reward, self.done, {}]
        elif target > self.num_possible_moves:
            print("Impossible. Invalid position")
            return [self.return_state, self.reward, self.done, {}]
        else:
            counter = self.counter+1
            if (counter == self.max_moves):
                done = True
                reward = self.final_reward()
                return [self.return_state, reward, done, {}]
            #self.state[0][int(target/h)][target%k] = 1
            #print("valid", self.valid_move(target))
            if not self.valid_move(target):
                reward = -1
                return [self.return_state, reward, self.done, {}]

            updated_board = self.merge(target)
            #print("board should be 0",self.state)
            reward = self.pseudo_reward(updated_board,target)
            state_0 = updated_board

            #do nothing so same state
            if (target == h*w*d):
                return [self.return_state, reward, self.done, {}]
            #no pieces left so we're done
            if len(self.remaining_shapes) == 0:
                #print("hi")
                state_1 = np.zeros(self.board_shape)
                return_state = np.concatenate((state_0, state_1), axis=2)
                done = True
                reward = self.final_reward()
                return [return_state, reward, done, {}]
            else:
                val = random.choice(range(len(self.remaining_shapes)))
                state_1 = self.remaining_shapes[val]
                #if not self.replace:
                #    self.remaining_shapes.pop(val)
                return_state = np.concatenate((state_0, state_1), axis=2)
                return [return_state, reward, self.done, {}]




    def render(self, mode='human'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # your real data here - some 3d boolean array
        x, y, z = np.indices((5, 5, 5))
        #voxels = (x == 5) | (y == 5)
        voxels = self.state[0]

        ax.voxels(voxels)

        plt.show()
    def render_piece(self, mode='human'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # your real data here - some 3d boolean array
        x, y, z = np.indices((5, 5, 5))
        #voxels = (x == 5) | (y == 5)
        voxels = self.state[1]

        ax.voxels(voxels)

        plt.show()

def generate_pieces(max_h, max_w, max_d):
    pieces = []
    for i in range(1,max_h):
        for j in range(1, max_w):
            for k in range(1, max_d):
                shape = np.ones(i*j*k).reshape(i, j, k).astype(int).tolist()
                pieces.append(shape)
    return pieces

def max_heuristic_move(env, verbose=False):
    max_reward = -100
    max_move = None
    for move in range(env.num_possible_moves + 1):
        reward = env.pseudo_step(move)[1]
        if verbose:
            print (reward)
        if reward > max_reward:
            max_reward = reward
            max_move = move
    return max_move

def simulate_env(env):
    volume_left = env.num_possible_moves
    while(True):
        print(volume_left)
        piece_size = np.sum(env.state[1])
        opt_move = max_heuristic_move(env)
        env.step(opt_move)
        volume_left -= piece_size
        if volume_left <= 0:
            return env

def trial_accuracy(env):
    volume_left = env.num_possible_moves
    while(True):
        print("volume left", volume_left)
        #print(volume_left)
        piece_size = np.sum(env.state[1])
        opt_move = max_heuristic_move(env)
        env.step(opt_move)
        volume_left -= piece_size
        if volume_left <= 0:
            return np.sum(env.state[0])/env.num_possible_moves

def simulate_step(env):
    opt_move = max_heuristic_move(env, verbose=True)
    env.step(opt_move)
    env.render()

env = PackEnv2(board_shape=(8,8,8), input_shapes=generate_pieces(4,4,4), max_moves = 200)
x = trial_accuracy(env)

print("trail accuracy is {}".format(x))
