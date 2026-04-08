import jax.numpy as jnp
from jax import grad

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def predict(self, w, b, X):
        return jnp.dot(X, w) + b
    
    def loss(self,w,b,X, y):
        y_pred = self.predict(w,b,X)
        return jnp.mean(jnp.square(y - y_pred))
    
    def fit(self,X,y):
        self.weights = jnp.zeros(X.shape[1])
        self.bias = 0.0

        grad_fn = grad(self.loss, argnums=(0,1))

        for _ in range(self.n_iterations):
            dw,db = grad_fn(self.weights, self.bias, X,y)
            self.weights -= self.learning_rate* dw
            self.bias -= self.learning_rate*db
        
        return self.weights, self.bias
# Example usage:
if __name__ == "__main__":
    # Sample data
    X = jnp.array([[1,3], [2,4], [3,5], [4,6], [5,7]])
    y = jnp.array([2, 3, 5, 7, 11])

    # Create and fit the model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    # Predict and evaluate
    predictions = model.predict(model.weights, model.bias, X)
    print("Predictions:", predictions)
    print("Mean Squared Error:", model.loss(model.weights, model.bias, X, y))