const seedrandom = require('seedrandom')

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

function getRandomVector(size, rng, max = 1) {
    const vec = [];
    for (let i = 0; i < size; i++) {
        vec.push(rng() * max);
    }
    return vec;
}

neurons = [
    new Neuron(getRandomVector(1, seedrandom())),
    new Neuron(getRandomVector(1, seedrandom())),
    new Neuron(getRandomVector(1, seedrandom()))
]

console.log(neurons)