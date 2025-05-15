import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

num_sample = 100

# Feature variable Size and Target variable Price
size = np.random.randint(500,1000,num_sample)
price = size * np.random.random()*3 + np.random.randint(-50,50,num_sample)

# Create A DataFrame
house_df = pd.DataFrame({"Size": size, "Price": price})
print(house_df)

# Plot Size VS Price to see any correlations
plt.scatter(house_df.Size, house_df.Price, c='r', s=10)
plt.title("Correlation between Size and Price")
plt.xlabel("Size (feet^2)")
plt.ylabel("Price (1000$)")
plt.grid(axis='y', linestyle='--', lw=0.5, alpha=0.5)
plt.show()

# Define Linear Regression Model
def f (x, w, b):
	return w*x + b

# Plot Actual Price VS Predicted Price
w , b = map(float, input("Enter W,b in respect : ").split())
plt.scatter(house_df.Size, house_df.Price, c='r', s=10, label='Actual Price')
plt.plot(house_df.Size, f(house_df.Size, w, b), linestyle='solid', lw=1.5, label='Predicted Price')
plt.title("Prediction Price based on W & b")
plt.xlabel("Size (feet^2)")
plt.ylabel("Price (1000$)")
plt.grid(axis='y', linestyle='--', lw=0.5, alpha=0.5)
plt.legend(loc='best', shadow=True)
plt.show()

# Define Squared Error Cost Function
def j (x, y , w , b):
	return np.sum(np.square(f(x, w , b) - y)) * (1 / (2 * x.shape[0]))

# Define Gradient Function 
def gradient (x,y,w,b):
	"""
	Args:
		x (ndarray, (m,)) : Feature variable with m examples
		y (ndarray, (m,)) : Target variable
		w,b (Scalars) : Model Parameter
	Returns:
		dj_dw : Derivative of cost function W.R.T. parameter W
		dj_db : Derivative of cost function W.R.T. parameter b
	 """
	m = x.shape[0]
	dj_dw = (1 / m) * np.sum((f(x,w,b) - y) * x)
	dj_db = (1 / m) * np.sum(f(x,w,b) - y)
	return dj_dw, dj_db
# Define Gradient Descent Algorithm
def gradient_descent (x, y, w_in, b_in, alpha, num_iter, cost_function, gradient_function):
	"""
	Args:
		x (ndarray (m,)) Feature variable with m examples
		y (ndarray (m,)) Target variable
		w_in, b_in (scalar) : initialization of w, b
		alpha (float) : learning rate
		num_iter (int): number of iteration for gradient descent
		cost_function : squared error cost function
		gradient function : generate dj_dw, dj_db
	Returns:
		w, b (scalar): parameters after the gradient descent 
		J_history (list) : list of cost function history
		p_history (list) : list of (w,b) history
	"""
	J_history = list()
	p_history = list()
	w = w_in
	b = b_in

	for i in range(num_iter):
		dj_dw , dj_db = gradient_function(x,y,w,b)

		w = w - alpha * dj_dw
		b = b - alpha * dj_db

		if i < 1000000:
			J_history.append(cost_function(x,y,w,b))
			p_history.append([w,b])
		if i % math.ceil(num_iter * 0.1) == 0:
			print(f"Iteration {i} : cost function = {J_history[-1]:.3e}",
				f"dj_dw = {dj_dw:0.4e}, dj_db = {dj_db:.4e}",
				f"w = {w:.3f}, b = {b:.5f}")
	return w, b , J_history, p_history

# Training using gradient descent
w_init = w
b_init = b
alpha = 0.000001
iterations = 100000

w_final, b_final, j_hist, p_hist = gradient_descent (house_df.Size, house_df.Price, w_init, b_init, alpha, iterations, j, gradient)
print(f"w = {w_final:.3f}, b = {b_final:.5f} , cost function = {j_hist[-1]}")

# Plot Number of Iterations  VS Cost Function
plt.plot(j_hist[:])
plt.title("Iteration VS Cost function (first 100 Iterations)")
plt.xlabel("Iteration")
plt.ylabel("J(w,b)")
plt.grid(axis='both', linestyle='--', lw=0.5, alpha=0.5)
plt.show()


# Plot a Linear Regression Line using  final_w and final_b
plt.scatter(house_df.Size, house_df.Price, c='r', s=10, label='Actual Price')
plt.plot(house_df.Size, f(house_df.Size, w_final, b_final), linestyle='solid', lw=1.5, alpha=0.7, label='Predicted Price')
plt.title("House Price Prediction")
plt.xlabel("Size (feet^2)")
plt.ylabel("Price (1000$)")
plt.grid(axis='both', linestyle='--', lw=0.5, alpha=0.5)
plt.legend(shadow=True)
plt.show()
