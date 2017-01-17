import pandas as pd 
import numpy as np 


filename = 'seeds.txt'
LEARNING_RATE = 0.001
NUM_INPUTS = 3
NUM_HIDDEN = 4
NUM_OUTPUTS = 3

def generate_dataset(filename):

	 data = pd.read_csv('seeds.txt',sep='\s+',header = None, names = ['Area','Perimeter','Compactness','Klength', \
												'Kwidth','Asym_coeff','length of kernel groove','class_label'])                                    

	 data.drop(data.columns[[3,4,5,6]] ,axis=1, inplace=True)
	 
	# Shuffling the dataset
	 data = data.sample(frac=1).reset_index(drop=True)

	 # Putting rows into list
	 dataset = data.values.tolist()
	 
	 # Dividing the dataset into training and test
	 train_dataset = dataset[:180]
	 test_dataset  = dataset[180:]

	 return train_dataset, test_dataset

def normalize_dataset(dataset,minmax):
	for row in dataset:
		for i in xrange(len(row)-1):
			row[i] = (row[i] -  minmax[i][0]) / (minmax[i][1] - minmax[i][0])


class NN(object):

	def __init__(self,n_inputs,n_hidden,n_output):

		self.number_of_inputs = n_inputs
		self.number_of_hidden = n_hidden
		self.number_of_outputs = n_output

		self.W1 = np.random.rand(self.number_of_inputs,self.number_of_hidden)
		self.W2 = np.random.rand(self.number_of_hidden,self.number_of_outputs)

	def sigmoid(self,x):
		return 1/(1+(np.exp(x))) 

	def sigmoid_prime(self,x):
		return self.sigmoid(x) * (1.0 - self.sigmoid(x))

	def feed_forward(self,X): # X being the input 

		self.hidden_layer_sums = np.dot(X,self.W1)
		self.hidden_layer_activation = self.sigmoid(self.hidden_layer_sums)
		self.output_sum = np.dot(self.hidden_layer_activation,self.W2)
		yP = self.sigmoid(self.output_sum)

		return yP

	def cost_dericative_function(self,y,yP):

		# Here y is the actual value and yP is the predicted value

		delta3 = np.multiply(-(y - yP), \
										self.sigmoid_prime(self.output_sum))
		dJdW2 = np.dot(self.hidden_layer_activation.T,delta3)
		delta2 = np.dot(delta3,self.W2.T) * self.sigmoid_prime(self.hidden_layer_sums)
		dJdW1 = np.dot(X.T,delta2)

		return dJdW1 , dJdW2

	def train(self,learning_rate,y,yP):

		# Actual Training of the network

		# for e in xrange(epochs):

		dJdW1,dJdW2 = self.cost_dericative_function(y,yP)
		
		#Updating the weights
		self.W1 = self.W1 + learning_rate * dJdW1
		self.W2 = self.W2 + learning_rate * dJdW2



def main():

		train_dataset, test_dataset = generate_dataset(filename)

		# Creating a neural network object
		nn = NN(NUM_INPUTS,NUM_HIDDEN,NUM_OUTPUTS)
		
		#dataset_dictionary = {train_dataset[i][:]}
		for i in xrange(len(train_dataset)):

			X =  train_dataset[i][:3]
			yP = nn.feed_forward(X)
			y = train_dataset[i][-1]
			nn.train(LEARNING_RATE,y,yP)

		
if __name__=="__main__":

		main()
		



