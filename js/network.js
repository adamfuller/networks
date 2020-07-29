class Network{

    constructor(){
        this.neurons = [];
        this.neuronOrder = [];
        // Neuron selected for connections
        this.selected = null;
    }

    toString(){
        return JSON.stringify({
            "neurons": this.neurons.map(n=> n.toJSON()),
            "neuronOrder": this.neuronOrder.map(na=>na.map((n)=>n.id)),
        },null, 2);
        // return this.neurons.map((r,n) => r+=n.toString(),"");
    }

    attachNeuron(neuron){
        this.neurons.push(neuron);
    }

    /**
     * Build the neuron order, so that forward and backward 
     * propagation can be performed.
     */
    buildNeuronOrder(){
        // This needs to be updated for building with the GUI
        console.log("Building network order");
        this.neuronOrder = [[]];

        // Find all input neurons
        for (let i = 0; i<this.neurons.length; i++){
            if (this.neurons[i].depth == -1 && this.neurons[i].isInput){
                // Neuron is part of the input layer
                this.neurons[i].assignDepth(0);
                this.neuronOrder[0].push(this.neurons[i]);
            }
        }
        

        // Make a copy of the neurons
        let maxDepth = -1;
        this.neurons.forEach((n) => maxDepth = Math.max(n.depth, maxDepth));
        if (maxDepth == -1) maxDepth = 1;
        // Presize
        this.neuronOrder.length = maxDepth;
        this.neurons.forEach((n) => {
            // Don't re-add the inputs
            if (n.depth == 0) return;
            if (this.neuronOrder[n.depth]){
                this.neuronOrder[n.depth].push(n);
            } else {
                this.neuronOrder[n.depth] = [n];
            }
        });
        // console.log(this.neuronOrder);
    }

    forwardProp(input){

        if (this.neuronOrder.length == 0){
            console.log("Neuron Order not build! call buildNeuronOrder()");
            return;
        }

        // Start the propagation with the input layer
        this.neuronOrder[0].forEach((n) => n.forwardProp(input));

        for (let i = 1; i<this.neuronOrder.length; i++){
            // They should fetch the output of their parent neurons
            this.neuronOrder[i].forEach((n) => n.forwardProp());
        }

        return this.neuronOrder[this.neuronOrder.length-1].reduce((result, n) => {
            if (n.children.length == 0){
                result.push(n.output);
            }
            return result;
        }, []); 
    }

    backwardProp(expectedOutput){
        let outputLayer = this.neuronOrder[this.neuronOrder.length-1];

        for (let i = 0; i<outputLayer.length; i++){
            // If the element has children it is not an output
            if (outputLayer[i].children.length > 0){
                continue;
            }
            outputLayer[i].backPropOutput(expectedOutput[i]);
        }

        for (let i = this.neuronOrder.length-2; i>=0; i--){
            this.neuronOrder[i].forEach((n) => n.backPropHidden());
        }

        // Apply all the adjustments
        this.neurons.forEach((n) => n.applyAdjustments());
    }

    draw(canvasId){
        let canvas = document.getElementById(canvasId);
        let ctx = canvas.getContext("2d");
        ctx.clearRect(0,0, canvas.clientWidth, canvas.clientHeight);

        this.neurons.forEach((n) => n.draw(canvasId, false));

        for (let i = this.neurons.length-1; i>=0;i--){
            this.neurons[i].drawArrows(canvasId);
        }
    }

}

class Neuron {
    sigmoid(x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    sigmoidDerivative(x) {
        let s = this.sigmoid(x);
        return s * (1.0 - s);
    }
    relu(x) {
        if (x > 0.0)
            return x;

        return 0.0;
    }
    reluDerivative(x) {
        if (x > 0.0)
            return 1.0;

        return 0.0;
    }
    /**
     * @param {number} x - input to leakyReLU
     */
    leakyRelu(x) {
        if (x > 0.0)
            return x;

        return 0.1 * x;
    }
    leakyReluDerivative(x) {
        if (x > 0.0)
            return 1;

        return 0.1;
    }

    constructor(learningRate=0.033, id=null, isInput=false) {

        // Location on the canvas, -1 means not placed on a canvas
        this.x = -1;
        this.y = -1;
        this.size = 10;
        this.color = "blue"; // blue for normal, red for selected

        this.inputs = [];
        this.parents = [];
        this.children = [];
        this.nextLayerIndices = [];
        this.depth = -1;
        this.isInput=isInput;

        this.id = id || Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);

        /**
         * Current output of the neuron
         */
        this.output = 0.0;
        this.delta = 0.0;
        this.error = 0.0;

        /**
         * Weights for inputs
         */
        this.weights = [];
        this.weightAdj = [];
        this.inputs = [];

        // for (let i = 0; i < inputCount; i++) {
        //     this.weights.push(2.0 * Math.random() - 1.0);
        //     this.weightAdj.push(0.0);
        // }

        this.learningRate = learningRate;
        this.activationFunction = this.leakyRelu;
        this.activationFunctionDerivative = this.leakyReluDerivative;

    }

    updatePos(x,y){
        this.x = x;
        this.y = y;
    }

    toJSON(){
        return {
            "id": this.id,
            "weights": this.weights,
            "parents": JSON.stringify(this.parents.map((n)=>n.id)),
            "depth": this.depth,
        };
    }

    toString(){
        return JSON.stringify({
            "id": this.id,
            "weights": this.weights,
            "parents": JSON.stringify(this.parents.map((n)=>n.id)),
            "depth": this.depth,
        }, null, 2);
    }

    connectParent(parent){
        if (this.parents.includes(parent))
            return;

        this.weights.push(2.0 * Math.random() - 1.0);
        this.weightAdj.push(0.0);

        // this.depth = Math.max(parent.depth + 1, this.depth);

        this.parents.push(parent);
        parent.connectChild(this, true);
    }

    connectChild(child, isCalledFromParent=false){
        if (this.children.includes(child))
            return;

        if (isCalledFromParent){
            this.nextLayerIndices.push(child.parents.length - 1);
        } else {
            this.nextLayerIndices.push(child.parents.length);
        }

        this.children.push(child);
        child.connectParent(this);
    }

    assignDepth(depth){
        this.depth = depth;
        this.children.forEach((n) =>{
            if (n.depth < 0){
                n.assignDepth(depth+1);
            }
        });
    }

    /**
     * Re-randomize the neuron's weights
     */
    reset() {
        for (let i = 0; i < this.weights.length; i++) {
            weights[i] = (2.0 * Math.random() - 1.0);
        }
    }

    /**
     * Adjust the size of the weights for a given input
     * @param {number[]} input - input to be accomodated
     */
    adjustForInput(input) {
        this.inputs = [1.0].concat(input);

        if (this.weights.length == this.inputs.length) return;

        // Increase the size of weights
        if (this.inputs.length >= this.weights.length) {
            for (let i = this.weights.length; i < this.inputs.length; i++) {
                this.weights.push(2.0 * Math.random() - 1.0);
                this.weightAdj.push(0.0);
            }
            // Don't go further
            return;
        }


        // Reduce the size of weights
        if (this.inputs.length < this.weights.length) {
            this.weights = this.weights.slice(0, this.inputs.length);
            this.weightAdj = this.weightAdj.slice(0, this.inputs.length);
        }
    }

    /**
     * Returns a number representing this neurons output for a given input
     * @param {number[]} input - input to the neuron
     */
    forwardProp(input=null) {
        // The input to this neuron is the output of the parents
        if (!input){
            input = this.parents.map((n) => n.output);
        }
        this.output = 0;

        // Adjust the current weights based on the given input
        this.adjustForInput(input);

        // console.log(this.id, "Input:", this.inputs);

        // Calculate the output
        for (let i = 0; i < this.inputs.length; i++) {
            this.output += this.inputs[i] * this.weights[i];
        }

        // Apply the activation function and set the output as the result
        this.output = this.activationFunction(this.output);

        // return this neurons output
        return this.output;
    }

    /**
     * Computes adjustments to the weights based on an expected output
     * @param {number} expectedOutput 
     */
    backPropOutput(expectedOutput) {
        // Difference between expected and calculated output
        this.error = expectedOutput - this.output;

        // Adjust for input based on error * gradient of output
        this.delta = this.error * this.activationFunctionDerivative(this.output);

        if (!!!this.delta) {
            console.log("BAD DELTA!!");
        }

        // For each input calculate the new corresponding weight
        this.weightAdj = this.inputs.map((i) => i * this.delta);
    }

    /**
     * Performs back propagation after getting the next layers weights and deltas
     */
    backPropHidden() {
        let nextLayerDeltas = [];
        let nextLayerWeights = [];

        for (let i = 0; i<this.children.length; i++){
            nextLayerDeltas[i] = this.children[i].delta;
            // +1 for the offset of the bias
            nextLayerWeights[i] = this.children[i].weights[this.nextLayerIndices[i]+1];
        }

        this.delta = 0.0;

        // Adjust based on each neuron that pulls from this one
        for (let i = 0; i < nextLayerDeltas.length; i++) {
            this.delta += nextLayerDeltas[i] * nextLayerWeights[i];
        }

        this.delta *= this.activationFunctionDerivative(this.output);

        // Assign the adjustments
        for (let i = 0; i < this.inputs.length; i++) {
            this.weightAdj[i] = this.delta * this.inputs[i];
        }
    }

    applyAdjustments() {
        // let maxWeight = 0.0;
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += this.weightAdj[i] * this.learningRate;
            // if (Math.abs(this.weights[i]) > maxWeight) maxWeight = Math.abs(this.weights[i]);
                // this.weightAdj[i] = 0.0;
        }
        // Normalize the weights
        // this.weights = this.weights.map((w) => w / maxWeight);
    }

    drawArrows(canvasId){
        let canvas = document.getElementById(canvasId);
        let ctx = canvas.getContext("2d");
        var arms = 10;
        ctx.strokeStyle = "white";
        ctx.beginPath();
        this.children.forEach((child)=>{
            let dx = child.x - this.x;
            let dy = child.y - this.y;
            let angle = Math.atan2(dy, dx);
            let left = angle - Math.PI/6;
            let right = angle + Math.PI/6;

            ctx.moveTo(this.x, this.y);
            ctx.lineTo(child.x,child.y);
            ctx.lineTo(child.x-arms*Math.cos(left), child.y - arms * Math.sin(left));
            ctx.moveTo(child.x,child.y);
            ctx.lineTo(child.x-arms*Math.cos(right), child.y - arms * Math.sin(right))
        });
        ctx.stroke();
    }

    draw(canvasId, drawArrows=true){
        let canvas = document.getElementById(canvasId);
        let ctx = canvas.getContext("2d");
        ctx.fillStyle = this.color;
        // Draw the neuron
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, 2 * Math.PI);
        ctx.fill();
        if (drawArrows){
            // Draw its connections, TODO: make colorful
            ctx.strokeStyle = "white";
            this.drawArrows(canvasId);
        }
    }

}

/**
 * 
 * @param {canvas} canvasId 
 * @param {Network} network 
 */
function addListeners(canvasId, network) {
    var canvas = document.getElementById(canvasId);


    document.addEventListener('keyup', (e) => {
        if (e.key == "c"){
            network.neurons = [];
            network.neuronOrder = [];
            network.draw(canvasId);
        } else if (e.key == "b"){
            network.buildNeuronOrder();
        }
    });

    canvas.addEventListener("mousedown", function (evt) {
        let rect = canvas.getBoundingClientRect();
        var mouseLoc = {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top,
        };
        // console.log(mouseLoc);
        
        for (var index in network.neurons){
            let n = network.neurons[index];
            // Check if the neuron was clicked
            let dist = Math.sqrt((Math.pow(n.x-mouseLoc.x,2)+Math.pow(n.y-mouseLoc.y,2)));
            // console.log("Distance", dist, n.x, n.y, mouseLoc.x, mouseLoc.y);
            // Click was within the neuron 
            if (dist <= n.size){
                if (n == network.selected){
                    // If the same one is double clicked, cancel it
                    network.selected.color = "blue";
                    // network.selected.draw(canvasId);
                    network.selected = null;
                    network.draw(canvasId);
                    return;
                }
                // Neuron was selected
                if (network.selected){
                    // Connect it as a child of the currently selected neuron
                    n.connectParent(network.selected);
                    // Re-color the selected one
                    network.selected.color = "blue";
                    network.selected = null;
                } else {
                    // Make it the currently selected neuron
                    network.selected = n;
                    n.color = "red";
                }
                network.draw(canvasId);
                return; // Stop looking through
            }
        }
        if (network.selected){
            // Update the selected to no longer show as selected
            network.selected.color = "blue";
            network.draw(canvasId);
            network.selected = null;
            // network.selected.draw(canvasId);
            return;
        }
        // Clear the network's selected neuron
        network.selected = null;

        // Make a new neuron at the given location
        var neu = new Neuron();
        neu.updatePos(mouseLoc.x, mouseLoc.y);
        // console.log("Position:", neu.x, neu.y);

        // Attach the neuron
        network.attachNeuron(neu);

        // neu.draw(canvasId);
        network.draw(canvasId);
    })

}

let lf = 0.133;

// Input neurons
let n1 = new Neuron(lf, "N1", true);
let n2 = new Neuron(lf, "N2", true);

// Hidden neurons
let n3 = new Neuron(lf, "N3");
let n4 = new Neuron(lf, "N4");

// Output neurons
let n5 = new Neuron(lf, "N5");


n1.connectChild(n3);
n1.connectChild(n4);

n2.connectChild(n3);
n2.connectChild(n4);

n3.connectChild(n5);
n4.connectChild(n5);

let net = new Network();

net.attachNeuron(n1);
net.attachNeuron(n2);
net.attachNeuron(n3);
net.attachNeuron(n4);
net.attachNeuron(n5);

net.buildNeuronOrder();

// net.neurons.forEach((n) => console.log(n.id, n.depth));

console.log("Before Training:");
console.log("(0,0 -> 0): " , net.forwardProp([0,0]));
console.log("(0,1 -> 1): " , net.forwardProp([0,1]));
console.log("(1,0 -> 1): " , net.forwardProp([1,0]));
console.log("(1,1 -> 0): " , net.forwardProp([1,1]));

for (let i = 0; i<10000; i++){
    net.forwardProp([0, 0]);
    net.backwardProp([0.0]);
    // 0,1
    net.forwardProp([0, 1]);
    net.backwardProp([1.0]);
    // 1,0
    net.forwardProp([1, 0]);
    net.backwardProp([1.0]);
    // 1,1
    net.forwardProp([1, 1]);
    net.backwardProp([0.0]);
    
    // if (i%500 == 0){
    //     console.log("(0,0 -> 0): " , net.forwardProp([0,0]));
    //     console.log("(0,1 -> 1): " , net.forwardProp([0,1]));
    //     console.log("(1,0 -> 1): " , net.forwardProp([1,0]));
    //     console.log("(1,1 -> 0): " , net.forwardProp([1,1]));
    //     console.log("\n");
    // }
}

console.log("\nAfter Training:");
console.log("(0,0 -> 0): " , net.forwardProp([0,0]));
console.log("(0,1 -> 1): " , net.forwardProp([0,1]));
console.log("(1,0 -> 1): " , net.forwardProp([1,0]));
console.log("(1,1 -> 0): " , net.forwardProp([1,1]));
// console.log(net.toString());