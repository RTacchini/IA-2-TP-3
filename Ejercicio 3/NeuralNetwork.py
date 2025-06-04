import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.initialize()

    def initialize(self):
    
        np.random.seed(42)  # semilla para reproducibilidad
        n_inputs = 3
        n_hidden1 = 5
        n_hidden2 = 5
        n_outputs = 3

        # pesos y sesgos primera capa oculta
        self.W1 = np.random.randn(n_inputs, n_hidden1) * (1/np.sqrt(n_inputs))
        self.b1 = np.zeros((1, n_hidden1))

        # pesos y sesgos segunda capa oculta
        self.W2 = np.random.randn(n_hidden1, n_hidden2) * (1/np.sqrt(n_hidden1))
        self.b2 = np.zeros((1, n_hidden2))

        # pesos y sesgos capa salida
        self.W3 = np.random.randn(n_hidden2, n_outputs) * (1/np.sqrt(n_hidden2))
        self.b3 = np.zeros((1, n_outputs))

    def relu(self, x):  # función de activación ReLU, entre entrada y capa oculta
        return np.maximum(0, x)

    def softmax(self, x): # función de activación softmax, entre capa oculta y salida
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    def think(self, inputs):
        x = np.array(inputs).reshape(1, -1)  # Asegura formato (1, n_inputs)
        z1 = np.dot(x, self.W1) + self.b1
        h1 = self.relu(z1)
        
        z2 = np.dot(h1, self.W2) + self.b2
        h2 = self.relu(z2)

        z3 = np.dot(h1, self.W3) + self.b3

        output = self.softmax(z3)  # Salida de la red neuronal

        #para ver los datos
        #action = self.act(output) 
        #print(f"Inputs: {inputs} -> Salidas softmax: {output} -> Acción: {action}")

        return self.act(output)

    def act(self, output):
        action = np.argmax(output)
        if action == 0:
            return "JUMP"
        elif action == 1:
            return "DUCK"
        elif action == 2:
            return "RUN"