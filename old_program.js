const seedrandom = require("seedrandom");

class Neuron {
    constructor(weight) {
        this.weights = weight;
    }

    activate(value) {
        return value^2;
    }

    forward(input) {
        let result = 0;
        for(const key in this.weights) {
            if (this.weights.hasOwnProperty(key) && input.hasOwnProperty(key)) {
                result += this.weights[key] * input[key];
            }
        }

        return this.activate(result);
    }

    backpropagate(deltas) {
        for (const deltaKey in deltas) {
            if (deltas.hasOwnProperty(deltaKey) && this.weights.hasOwnProperty(deltaKey)) {
                this.weights[deltaKey] -= deltas[deltaKey];
            }
        }
        console.log(`updated weights: ${this.weights}`)
    }
}

class Trainer {
    static meanSquare(x, y) {
        console.log(`Loss: ${Math.pow(x - y, 2)}`);
    }

    static gradient(weights, inputs, expected, result, learningRate = 0.0001) {
        const newWeights = [];
        for (const weightKey in weights) {
            if (weights.hasOwnProperty(weightKey) && inputs.hasOwnProperty(weightKey)) {
                newWeights.push((weights[weightKey] - inputs[weightKey]) * 2 * learningRate * (expected - result));
            }
        }
        // console.log(`old weights: ${weights},\ninputs: ${inputs},\nexpected: ${expected},\nresult: ${result},\nnew weights: ${newWeights}`)
        return newWeights;
    }

    static trainingLoop(neuron, loops = 10) {
        const rng = seedrandom("oui");
        for (let i = 0; i < loops; i++) {
            const vector = getRandomVector(neuron.weights.length, rng, 25);
            const result = neuron.forward(vector);
            let expected = 0;
            for(const key in neuron.weights) {
                if (neuron.weights.hasOwnProperty(key) && vector.hasOwnProperty(key)) {
                    expected += vector[key];
                }
            }
            this.meanSquare(expected, result);
            neuron.backpropagate(this.gradient(neuron.weights, vector, expected, result));
        }
    }
}


function getRandomVector(size, rng, max = 1) {
    const vec = [];
    for (let i = 0; i < size; i++) {
        vec.push(rng() * max);
    }
    return vec;
}

const neuron = new Neuron(getRandomVector(2, seedrandom()));
Trainer.trainingLoop(neuron, 1000);
