import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

#learning rate
LR = 1e-3
#choose environment
env = gym.make('CartPole-v1')
env.reset()
#how many steps to play in environment
goal_steps = 500
#mininum score requirement to be added to training data
score_requirement = 50
#how many games to play to generate training data
#only games that meet the score requirement will be used to train
initial_games = 10000

'''
def some_random_games_first():
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				break

some_random_games_first()
'''

def initial_population():
	#data that will be used to train
	training_data = []
	#all scores
	scores = []
	#unformatted (not one_hot) scores that meet the requirement
	accepted_scores = []

	#run games
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(goal_steps):
			#generates either 0 or 1
			action = random.randrange(0, 2)
			observation, reward, done, info = env.step(action)

			#if there was a previous observation, append to game memory with current action
			#since action occurs before current observation, action is based on previous observation
			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])

			prev_observation = observation
			#add reward (0 for loss, else 1) to score
			score += reward
			if done:
				break


		if score >= score_requirement:
			#append acceptable scores to array
			accepted_scores.append(score)
			#format action into one_hot array
			#data[0] = prev_observation
			#data[1] = action
			for data in game_memory:
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]

				#add prev_observation, action (as one_hot array) to training data
				training_data.append([data[0], output])

		env.reset()
		#add score to array of all scores
		scores.append(score)

	#save training data as np.array()
	training_data_save = np.array(training_data)
	np.save('nn_data/saved.npy', training_data_save)

	print('Average accepted score: ', mean(accepted_scores))
	print('Median accepted score: ', median(accepted_scores))
	print(Counter(sorted(accepted_scores)))
	return training_data

def neural_network_model(input_size):
	#input_size = 4 for CartPole
	network = input_data(shape=[None, input_size, 1], name='input')

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=LR, 
		loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(network, tensorboard_dir='log')

	return model

def train_model(training_data, model=False):
	#X = input data = observations
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	#y = expected output = action (one_hot array)
	y = [i[1] for i in training_data]

	#create model if it doesn't exist
	if not model:
		model = neural_network_model(input_size = len(X[0]))

	#train the model
	model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True,
		run_id='reinforcement_learning_openai')
	return model

training_data = initial_population()

#TRAIN MODEL
'''
model = train_model(training_data)
model.save('nn_data/goodmodel.tflearn')
'''

#LOAD TRAINED MODEL


temp_X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
model = neural_network_model(input_size = len(temp_X[0]))
model.load('nn_data/goodmodel.tflearn')


#test the model
scores = []
choices = []

for each_game in range(10):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_steps):
		env.render()
		#start with a random action
		if len(prev_obs) == 0:
			action = random.randrange(0, 2)
		#perform the action chosen by the model based on the previous observation
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
		#add the action to the list of actions
		choices.append(action)

		new_observation, reward, done, info = env.step(action)
		#set the previous observation to the current one
		prev_obs = new_observation
		#add data to game memory (for retraining if wanted)
		game_memory.append([new_observation, action])
		#add reward to score
		score += reward
		if done:
			print("Trial: {}, Ended after {} frames".format(each_game+1, int(score)))
			break

	scores.append(score)

print('Average Score:', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))