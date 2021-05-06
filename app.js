const seedrandom = require("seedrandom");
const alphabet = require("./alphabet");


class Neuron {
    constructor(weight) {
        this.weights = weight;
    }
    
    /**
     * Met à jour le poid en soustrayant le poid avec le delta de la couche précédente
     * @param {float} delta 
     */
    updateWeights(delta) {
        for(let i = 0; i < this.weights.length; i++) {
            this.weights[i] -= delta;
        }
    }

    /**
     * Fonction d'activation Relu
     * @param {*} value 
     * @returns float
     */
    activate(value) {
        return value <= 0 ? 0 : value;
    }

    /**
     * @param {Array<Float>} input 
     * @returns 
     */
    forward(input) {
        let computed = input.map(i => i * this.weights[0]);
        return this.activate(computed.reduce((a, b) => a + b));
    }

}

class Trainer {
    static meanSquare(x, y) {
        console.log(`Loss: ${Math.pow(x - y, 2)}`);
    }

    /**
     * Compute le delta
     * @param {Array<Float>} weights 
     * @param {Array<Float>} inputs 
     * @param {Array<Float>} expected 
     * @param {Float} result 
     * @param {Float} learningRate 
     * @returns {Array<Float>} delta
     */
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

    constructor(layers) {
        this.input = []
        this.expected = []
        this.layers = layers;

        /**
         * Critère qui arrête l'apprentissage si n fois de suite
         * il a plus de 90% de reconnaissance de la lettre (ou n = nombre de lettre de l'alphabet)
         */
        this.criteria = 0;
    }

    backPropagate() {

        let delta_from_last_layer = ""

        this.layers.reverse().forEach((layer, indice) => {

            let current_layer_grad = [];

            layer.neurons.forEach((neuron) => {

                // Calcule le delta pour les neurones du layer
                current_layer_grad.push(
                    Trainer.gradient(
                        neuron.weights,
                        layer.inputs || this.input,
                        this.input,
                        layer.output
                    )
                )
            })


            // Si on est dans la première couche (donc la dernière vu qu'on a reverse la liste)
            if(indice === 0) {
                // On passe à la couche suivante
            }

            if(indice > 0) {

                // On applique le nouveau poid aux neuronnes de la couche
                layer.neurons.forEach((neuron) => {
                    neuron.updateWeights(delta_from_last_layer)
                })
            }
            
            // Fait une moyenne des gradients du layer et propage le nouveau poid à la prochaine couche 
            delta_from_last_layer = current_layer_grad.reduce((a, b) => (a + b) / current_layer_grad.length)

        }, this)
    }

    /**
     * @param {Array<Array<Float>>} alphabet 
     */
    fit(alphabet) {
        
        this.train = true;

        /*
        * Propage pour chaque lettre et rétropropage pour modifier les poids
        * Tant que les poids ne sont pas stables (se modifient à chaque rétropropagation)
        * on continue d'entrainer le neurone
        */
        while(this.criteria < alphabet.length) {
            alphabet.forEach(function(input) {

                // Flatten la matrice et mulitplie chaque valeur par -> 0 >= coefficient aléatoire <= 1
                this.input = [].concat(...input).map(x => x * Math.random()) 
                this.expected = [].concat(...input)
            
                this.propagate() // Propage pour chaque lettre
                this.backPropagate() // Rétropropage pour chaque lettre

            }, this);
        }


    }
    /**
     * Active le réseau de neurone
     */
    propagate() {

        var last_layer_output = []

        this.layers.forEach((layer, indice) => {

            layer.addInput(last_layer_output)

            const current_layer_output = []

            layer.neurons.forEach((neuron, i) => {

                if(!last_layer_output.length) { // 1 ere couche
                    current_layer_output.push(
                        neuron.forward([this.input[i]])
                    )

                } else {
                    //console.log(last_layer_output)
                    current_layer_output.push(
                        neuron.forward(last_layer_output)
                    )
                }
            }, this)
            
            layer.addOutput(current_layer_output)
            last_layer_output = current_layer_output;

            if(indice === this.layers.length - 1) {
                console.log("output " + current_layer_output + "\n")
                // TODO: Si les poids n'ont pas changé, ajouter +1 à this.criteria
            }
            
        }, this)
    }
}

/**
 * Crée des poids aléatoire de la taille "size"
 * @param {number} size 
 * @param {float} rng 
 * @param {int} max 
 * @returns 
 */
function getRandomVector(size, rng, max = 1) {
    const vec = [];
    for (let i = 0; i < size; i++) {
        vec.push(rng() * max);
    }
    return vec;
}

layer_input_neurons = [] // 25 neurones en entrée (correspond à une matrice 5x5)
layer_middle_1_neurons = [] // 20 neurones (valeur aléatoire, à changer pour mieux reconnaitre)
layer_middle_2_neurons = [] // 34 neurones (valeur aléatoire, à changer pour mieux reconnaitre)
layer_output_neurons = [] // 26 neurones en sortie (1 neurone par lettre à reconnaître)

for(let i = 0; i < 25; i++)
    layer_input_neurons.push(new Neuron(getRandomVector(1, seedrandom())))

for(let i = 0; i < 20; i++)
    layer_middle_1_neurons.push(new Neuron(getRandomVector(1, seedrandom())))

for(let i = 0; i < 34; i++)
    layer_middle_2_neurons.push(new Neuron(getRandomVector(1, seedrandom())))

for(let i = 0; i < 26; i++)
    layer_output_neurons.push(new Neuron(getRandomVector(1, seedrandom())))


layers = [
    new Layer("Input", layer_input_neurons, alphabet[0]),
    new Layer("Layer 1", layer_middle_1_neurons),
    new Layer("Layer 2", layer_middle_2_neurons),
    new Layer("Output", layer_output_neurons),
]

neural_network = new NeuralNetwork(layers)
neural_network.fit(alphabet)
