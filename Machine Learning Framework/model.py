import random, math

class Model:
    def __init__(self, inputs, hidden, outputs):
        self.w_i_h = [[random.random() - 0.5 for i in range(inputs)] for h in range(hidden)]
        self.w_h_o = [[random.random() - 0.5 for h in range(hidden)] for o in range(outputs)]

        self.b_i_h = [random.random() - 0.5 for h in range(hidden)]
        self.b_h_o = [random.random() - 0.5 for o in range(outputs)]

        self.leanring_rate = 0.01
        self.epochs = 1
    def softmax(self, predictions):
        n = max(predictions)
        temp = [math.exp(p - n) for p in predictions]
        total = sum(temp)
        return [t / total for t in temp]
    def log_loss(self, activations, targets):
        losses = [-t * math.log(a) - (1 - t) * math.log(1 - a) for a, t in zip(activations, targets)]
        return sum(losses)
    def predict(self, inputs):
        pred_h = [sum([w * i for w, i in zip(weight, inputs)]) + b 
                        for weight, b in zip(self.w_i_h, self.b_i_h)]
        act_h = [max(0, p) for p in pred_h]

        pred_o = [sum([w * i for w, i in zip(weight, act_h)]) + b 
                for weight, b in zip(self.w_h_o, self.b_h_o)]
        print(pred_o)
        return self.softmax(pred_o)
    def train(self, inputs, targets):
        for epochs in range(self.epochs):
            pred_h = [[sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(self.w_i_h, 
                                                                            self.b_i_h)] for inp in inputs]
        
            act_h = [[max(0, p) for p in pred] for pred in pred_h] # apply ReLU
            pred_o = [[sum([w * a for w, a in zip(weights, inp)]) + 
            bias for weights, bias in zip(self.w_h_o, self.b_h_o)] for inp in act_h]
            act_o = [self.softmax(predictions) for predictions in pred_o]

            cost = sum([self.log_loss(a, t) for a, t in zip(act_o, targets)]) / len(act_o)
            print("Epochs: {}, Cost: {}".format(epochs, cost))

            #Error derivatives
            errors_d_o = [[a - t for a, t in zip(ac, ta)] for ac, ta in zip(act_o, targets)]
            w_h_o_T = list(zip(*self.w_h_o))
            errors_d_h = [[sum([d * w for d, w in zip(deltas, weights)]) * (0 if p <= 0 else 1)
                for weights, p in zip(w_h_o_T, pred)] for deltas, pred in zip(errors_d_o, pred_h)]

            
            #Gradient Hidden -> output
            act_h_T = list(zip(*act_h))
            errors_d_o_T = list(zip(*errors_d_o))
            w_h_o_d = [[sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_o_T] for act in act_h_T]
            b_h_o_d = [sum([d for d in deltas]) for deltas in errors_d_o_T]

            #Gradient inputs -> hidden
            inputs_T = list(zip(*inputs))
            errors_d_h_T = list(zip(*errors_d_h))
            w_i_h_d = [[sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_h_T]
            for act in inputs_T]
            b_i_h_d = [sum([d for d in deltas]) for deltas in errors_d_h_T]

            # Update Weights and biases for all layers
            w_h_o_d_T = list(zip(*w_h_o_d))
            for y in range(len(w_h_o_d_T)):
                for x in range(len(w_h_o_d_T[0])):
                    self.w_h_o[y][x] -= self.learning_rate * w_h_o_d_T[y][x] / len(inputs)
                self.b_h_o[y] -= self.learning_rate * b_h_o_d[y] / len(inputs)

            w_i_h_d_T = list(zip(*w_i_h_d))
            for y in range(len(w_i_h_d_T)):
                for x in range(len(w_i_h_d_T[0])):
                    self.w_i_h[y][x] -= self.learning_rate * w_i_h_d_T[y][x] / len(inputs)
                self.b_i_h[y] -= self.learning_rate * b_i_h_d[y] / len(inputs)
    def test(self, test_inputs, test_targets):
        pred = [self.predict(i) for i in test_inputs]
        correct = 0

        for a, t in zip(pred, test_targets):
            if a.index(max(a)) == t.index(max(a)):
                correct += 1
        print(f"Correct: {correct}/{len(pred)} ({correct / len(pred)}%)")

