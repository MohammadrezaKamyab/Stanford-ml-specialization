#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math, copy

def generate_dataset(SAMPLES, random_seed=42):
	# Define Features of the Dataset
	hours_studied = np.random.uniform(5,30, SAMPLES)
	attendence_rate = np.clip(np.random.normal(85,10,SAMPLES),50,100)
	ave_assignment_score = np.clip(np.random.normal(75,15,SAMPLES),40,100)

	# Define a Noise in to the Dataset
	#noise = np.random.normal(0,5,SAMPLES)
	noise = np.zeros(SAMPLES)

	# Define Weights
	w1 = 0.3
	w2 = 0.5
	w3 = 0.2

	# Define Target Variable
	final_score = (
		w1 * hours_studied +
		w2 * attendence_rate +
		w3 * ave_assignment_score +
		noise
	)

	df = pd.DataFrame({
		"Hours_Studied": hours_studied,
		"Attendence_Rate": attendence_rate,
		"Averate_Assignment_Score": ave_assignment_score,
		"Final_Score": final_score
	})

	df.to_csv('./Data/Student_Data.csv')
	return df
#############################################################################3
def print_dataset(df, number = -1, * , randomize=False, frac=1, seed=50):
	if randomize == True:
		try:
			return df.sample(frac=frac, ignore_index=False, random_state=seed)
		except:
			raise ValueError("Frac is a number between 0 and 1")
	else:
		try:
			number = df.shape[0] if number == -1 else number
			return df.head(number)
		except:
			raise ValueError(f"Your dataset has {df.shape[0]} Data")
	
##############################################################################	
def draw_correlation_matrix (df):
	corr_matrix = df.corr()
	fig, ax = plt.subplots(figsize=(12,8))
	sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
	ax.set_title("Correlation Matrix")

	position = np.arange(len(df.columns)) + 0.5
	ax.set_xticks(position)
	ax.set_yticks(position)

	ax.set_xticklabels(["Study(H)", "AR", "MAS", "Final"], rotation=45, ha='center')
	ax.set_yticklabels(["Study(H)", "AR", "MAS", "Final"], rotation=45, va='center')

	plt.tight_layout()
	plt.show()
	del fig, ax

################################################################################
def plot_target(df):
	fig, ax = plt.subplots()
	ax.plot(df.iloc[:, -1], marker='o', markerfacecolor='red', markeredgecolor='red', markersize=5, lw=1.5, color='black')
	ax.grid(axis='y', linestyle='--', lw=0.5, alpha=0.5)
	
	plt.tight_layout()
	plt.show()
	del fig, ax

################################################################################
def compute_output(x, w, b):
	"""
	Args:
		x (ndarray) : shape(m, n) m is the number of training examples and n is number of features
		w (ndarray) : shape(n,) weight paramters for each feature
		b scalar : bias paramter
	Returns:
		f_wb (ndarray) : shape(m) output based on given w & b
	"""
	return np.dot(x, w) + b

#################################################################################
def compute_cost (x, y, w, b):
	"""
	Args:
		x (ndarray) : shape(m, n) m number of training example with n features
		y (ndarray) : shape(m, ) target variable
		w (ndarray) : shape(n), weight parameter for each feature
		b (scalary): bias
	Returns:
		J_wb (scalar) : How far our predictions are from the actual value
	"""
	m = y.shape[0]
	n = x.shape[0]
	return (1 * (2 / m)) * np.sum(np.square(compute_output(x,w,b) - y))

#################################################################################
def gradient(x,y,w,b):
	"""
	Args:
		x (ndarray) : Shape(m, n) m traning example with n number of features
		y (ndarray) : shape(m,) target variable
		w (ndarray) : shape(n,) weight paramter for each feature
		b (scalar) : bias - parameter
	Returns:
		dj_dw (ndarray) : shape(n,)
		dj_db (scalar)
	"""
	m = y.shape[0]
	dj_dw = np.dot(x.T,(compute_output(x,w,b) - y)) * (1 / m)
	dj_db = np.sum((compute_output(x,w,b) - y)) * (1/m)
	return dj_dw, dj_db
#################################################################################
def gradient_descent(x, y, w_in, b_in, alpha, num_iter, cost_function, gradient_function, tolerance=1e-6):
	w = copy.deepcopy(w_in)
	b = b_in
	J_history = list()


	for i in range(num_iter):
		dj_dw , dj_db = gradient_function(x,y,w,b)
		w = w - alpha * dj_dw
		b = b - alpha * dj_db

		if i <= 100000:
			J_history.append(cost_function(x,y,w,b))
		
		if i % math.ceil(num_iter / 10) == 0:
			print(f"Iteration {i:4d} : Cost {J_history[-1]:8.2f}")
		
		if i > 0 and abs(J_history[-1] - J_history[-2]) < tolerance:
			print(f"Convergence at {i:4d}")
			break
	return w, b, J_history



#################################################################################
def plot_prediction (df, w, b):
	predict = compute_output(df.iloc[:, :-1], w, b)

	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(df.iloc[:, -1], marker='o', markerfacecolor='r', markeredgecolor='r', markersize=7,color='r', alpha=0.5, lw=2, label='Actual Score')
	ax.plot(predict, marker='o', markerfacecolor='blue', markeredgecolor='blue', color='blue' ,markersize=7, alpha=0.5, lw=2, label='Predicted Score')
	ax.fill_between(df.index, df.iloc[:, -1], predict, color='grey', alpha=0.2, label='Error')
	ax.set_title("Prediction VS Actual Value")
	ax.legend(loc='best', fontsize='small')
	ax.set_ylim(df.iloc[:, -1].min() - 5, predict.max() + 35)
	ax.grid(axis='y', linestyle='--', lw=0.5, alpha=0.5)
	
	plt.tight_layout()
	plt.show()
	del fig, ax

#################################################################################
def rescaling_df (df):
	return df.iloc[:, :-1].apply(lambda x : (x - x.min()) / (x.max() - x.min()))
#################################################################################
def learning_curve (J_history, tolerance=1e-6):
	plt.plot(J_history)
	plt.title("Iteration VS Cost")
	plt.xlabel("Iteration")
	plt.ylabel("Cost")
	plt.show()
#################################################################################


def main():
	# Generate Data
	SAMPLES = int(input("How many Data to Generate : "))
	SEED = int(input("Set Random Seed : "))
	df = generate_dataset(SAMPLES, SEED)
	print_dataset(df, randomize=True, frac=0.5, seed=SEED)

	# Show Final Score rate
	plot_target(df)

	# Show Correlation Between Features in the dataset
	#draw_correlation_matrix(df)

	# Get weights for user_prediction
	w_hours_studied = float(input("Enter weight for Hours of Studied : "))
	w_attendence_rate = float (input("Enter weight for Attendence Rate : "))
	w_ave_assignment_score = float (input("Enter weight for Average Assignemnt Score :"))
	b = float(input("Enter Bias : "))
	w = np.array([w_hours_studied, w_attendence_rate, w_ave_assignment_score])
	
	# Show Plot of User Prediction Weights
	plot_prediction(df, w, b)
	print(f"Cost = {compute_cost(df.iloc[:, :-1], df.iloc[:, -1], w, b):.3e}")

	# Gradient Descent to Find the w, b Paramters
	print("--------------------------------")
	print(f"w = {w} , b = {b}")

	w_in = w
	b_in = b
	alpha =  1e-4
	iterations = 1000000
	r_df = df.copy()
	#r_df.iloc[:, :-1] = rescaling_df(df)
	w_final , b_final, J_hist = gradient_descent(r_df.iloc[:, :-1], r_df.iloc[:, -1], w_in, b_in, alpha, iterations, compute_cost, gradient, 1e-10)

	learning_curve(J_hist)
	print(f"W_Final = {w_final} , b_Final = {b_final:.3f}")
	print(f"Cost = {compute_cost(r_df.iloc[:, :-1], r_df.iloc[:, -1], w_final, b_final)}")
	plot_prediction(r_df, w_final, b_final)



if __name__ == '__main__': main()
