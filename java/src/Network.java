import java.util.ArrayList;
import java.util.stream.Collectors;

public class Network {
    private ArrayList<Neuron> neurons;
    private ArrayList<ArrayList<Neuron>> neuronOrder;
    // private Neuron selected;
    private boolean isScaled = false;

    public Network() {
        this.neurons = new ArrayList<>();
        this.neuronOrder = new ArrayList<>();
        // this.selected = null;
    }

    /**
     * Returns a JSON representation as a string
     */
    @Override
    public String toString() {
        // System.out.println("toString() needs to be implemented!");
        StringBuilder sb = new StringBuilder();
        this.neurons.forEach((n) -> sb.append(n.toString()).append('\n'));
        return sb.toString();
    }

    /**
     * Prints out the Layers and the associated neurons in them.
     * Layer 0 is the input layer.
     */
    public void printNeuronOrder(){
        System.out.println("Neuron Order:");
        for (int i = 0; i<this.neuronOrder.size(); i++){
            System.out.print(i + " : ");
            System.out.println(this.neuronOrder.get(i).stream().map((n) -> n.getId()).collect(Collectors.joining(",")));
        }
    }

    public int size(){
        return this.neurons.size();
    }

    public int depth(){
        return this.neuronOrder.size();
    }

    public void attachNeuron(Neuron n) {
        if (this.neurons == null) {
            this.neurons = new ArrayList<>();
        }
        this.neurons.add(n);
    }

    public void buildNeuronOrder() {

        // Start with a new empty order
        this.neuronOrder = new ArrayList<>();
        this.neuronOrder.add(new ArrayList<>());

        for (int i = 0; i<this.neurons.size(); i++){
            for (Neuron n : this.neurons.get(i).getChildren()){
                // If the child isn't present for some reason, add it, check that one's children as well
                if (!this.neurons.contains(n)){
                    System.out.println("Missing Neuron Found: " + n.getId());
                    this.neurons.add(n);
                }
            }
            for (Neuron n : this.neurons.get(i).getParents()){
                // If the child isn't present for some reason, add it, check that one's children as well
                if (!this.neurons.contains(n)){
                    System.out.println("Missing Neuron Found: " + n.getId());
                    this.neurons.add(n);
                }
            }
        }

        int maxDepth = -1;

        // Build out the depth values of each neuron from the input layer
        for (Neuron n : this.neurons) {
            maxDepth = n.getDepth() > maxDepth ? n.getDepth() : maxDepth;
            if (n.getDepth() <= 0 && n.isInput()) {
                // Neuron is part of the input layer
                n.assignDepth(0);
                this.neuronOrder.get(0).add(n);
            }
        }

        // Fill in the remaining levels
        for (int i = 0; i < maxDepth; i++) {
            this.neuronOrder.add(new ArrayList<>());
        }

        for (Neuron n : this.neurons) {
            if (n.getDepth() == 0)
                continue;
            // Add to the corresponding depth
            this.neuronOrder.get(n.getDepth()).add(n);
        }
    }

    /**
     * Scales the input layer to accept the input.
     * 
     * Must be called before using forwardProp
     * @param input
     */
    public void scaleToInput(double[] input){
        this.neurons.forEach((n) -> n.adjustForInput(input));
        this.isScaled = true;
    }


    /**
     * Performs forward propagation on the network with a given input
     * @param input - dataset input to the network
     * @return An array representing the output of the network
     */
    public double[] forwardProp(double[] input) {

        if (!this.isScaled){
            this.scaleToInput(input);
        }

        // System.out.println("Forward Prop Input Layer Size: " + this.neuronOrder.get(0).size());
        for (Neuron n : this.neuronOrder.get(0)){
            n.forwardProp(input);
        }

        // Iterate over each layer and forward propagate
        for (int i = 1; i<this.neuronOrder.size(); i++){
            this.neuronOrder.get(i).forEach(Neuron::forwardProp);
        }

        // Outputs the values from the last row
        return this.neuronOrder.get(this.neuronOrder.size()-1).stream().filter((n) -> n.childCount() == 0).map((n)->n.getOutput()).mapToDouble(Double::doubleValue).toArray();
    }

    public void backwardProp(double[] expectedOutput) throws Exception {
        ArrayList<Neuron> outputLayer = this.neuronOrder.get(this.neuronOrder.size()-1);

        if (expectedOutput.length > outputLayer.size()){
            throw new Exception("Output Layer is not properly sized (" + outputLayer.size() +") for an output of size: " + expectedOutput.length);
        }

        // Back propagate the output layer
        for (int i = 0; i<expectedOutput.length;i++){
            if (outputLayer.get(i).childCount() > 0){
                // This neuron isn't actually part of the output layer
                continue;
            }
            // Use the corresponding expected output
            outputLayer.get(i).backPropOutput(expectedOutput[i]);
        }

        // Back propagate hidden layers
        for (int i = this.neuronOrder.size()-2; i>=0; i--){
            this.neuronOrder.get(i).forEach((n) -> n.backPropHidden());
        }

        this.neurons.forEach((n)->n.applyAdjustments());
    }

}