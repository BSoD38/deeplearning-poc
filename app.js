const seedrandom = require("seedrandom");
const alphabet = require("./alphabet");


class Neuron {
    constructor(weight) {
        this.weights = weight;
    }

    activate(value) {
        // ReLu
        return value < 0 ? 0 : value;
    }

    forward(input) {
        input.map(i => i * this.weights[0])
        return this.activate(input.reduce((a, b) => a + b));
    }

    backpropagate(deltas) {
        for (const deltaKey in deltas) {
            if (deltas.hasOwnProperty(deltaKey) && this.weights.hasOwnProperty(deltaKey)) {
                this.weights[deltaKey] -= deltas[deltaKey];
            }
        }
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
        return newWeights;
    }
}

class Layer {

    constructor(name, neurons, input = []) {
        this.name = name
        this.neurons = neurons
        this.inputs = input
        this.output = []
    }

    /**
     * Ajoute les inputs d'un neurone dans le layer
     * @param {Array<number>} input 
     */
    addInput(input) {
        this.inputs.push(input)
    }

    /**
     * Ajoute les outs d'un neurone dans le layer
     * @param {Array<number>} input 
     */
     addOutput(output) {
        this.output.push(output)
    }
}

class NeuralNetwork {

    constructor(layers, input) {
        this.layers = layers;
        // Flatten la matrice et multiplie chaque valeur par une valeur entre 0 et 1 aléatoire
        this.input = [].concat(...input).map(x => x * Math.random())
        this.expected = [].concat(...input)
    }

    backPropagate() {
        this.layers.reverse().forEach((layer) => {

            layer.neurons.forEach((neuron) => {

                const grad = Trainer.gradient(
                    neuron.weights, layer.inputs || this.input,
                    this.input,
                    layer.output
                )

                neuron.weights -= grad

            })

        })
    }

    start() {

        var last_layer_output = []

        this.layers.forEach((layer) => {

            layer.addInput(last_layer_output)

            const current_layer_output = []

            layer.neurons.forEach((neuron, i) => {

                const neuron = layer.neurons[i]
                if(!last_layer_output.length) { // 1 ere couche

                    current_layer_output.push(
                        neuron.forward([this.input[i]])
                    )

                } else {
                    current_layer_output.push(neuron.forward(last_layer_output))
                }
            })
            
            layer.addOutput(current_layer_output)
            last_layer_output = current_layer_output;

        })
    }
    // TODO: Fonction qui active le réseau
    // TODO: Fonction de rétropropagation 
}

function getRandomVector(size, rng, max = 1) {
    const vec = [];
    for (let i = 0; i < size; i++) {
        vec.push(rng() * max);
    }
    return vec;
}

layer_input_neurons = [] // 25 neurones en entrée (correspond à une matrice 5x5)
layer_middle_1_neurons = [] // 15 neurones
layer_middle_2_neurons = [] // 12 neurones
layer_output_neurons = [] // 8 neurones en sortie (1 neurone par lettre à reconnaître)

for(let i = 0; i < 2; i++)
    layer_input_neurons.push(new Neuron(getRandomVector(1, seedrandom())))

for(let i = 0; i < 5; i++)
    layer_middle_1_neurons.push(new Neuron(getRandomVector(1, seedrandom())))

for(let i = 0; i < 4; i++)
    layer_middle_2_neurons.push(new Neuron(getRandomVector(1, seedrandom())))

for(let i = 0; i < 1; i++)
    layer_output_neurons.push(new Neuron(getRandomVector(1, seedrandom())))

layers = [
    new Layer("Input", layer_input_neurons, alphabet[0]),
    new Layer("Layer 1", layer_middle_1_neurons),
    new Layer("Layer 2", layer_middle_2_neurons),
    new Layer("Output", layer_output_neurons),
]

neural_network = new NeuralNetwork(layers, alphabet[0])
neural_network.start()
