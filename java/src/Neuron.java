import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.function.Function;

public class Neuron {
    private static double defaultLearningRate = 0.033;
    public static Function<Double, Double> sigmoid = (x) -> 1.0 / (1.0 + Math.exp(-x));
    public static Function<Double, Double> sigmoidDerivative = (x) -> {
        double s = Neuron.sigmoid.apply(x);
        return s * (1.0 - s);
    };
    public static Function<Double, Double> relu = (x) -> x > 0 ? x : 0.0;
    public static Function<Double, Double> reluDerivative = (x) -> x > 0 ? 1 : 0.0;

    public static Function<Double, Double> leakyRelu = (x) -> x > 0.0 ? x : (0.1 * x);
    public static Function<Double, Double> leakyReluDerivative = (x) -> x > 0.0 ? 1.0 : 0.1;

    // Computational parameters
    private double[] inputs, weights, weightAdj;
    private ArrayList<Neuron> children, parents;
    private ArrayList<Integer> nextLayerIndicies;
    /** Depth of the neuron in it's network */
    private int depth;
    private boolean isInput = false;
    private String id;
    private double output, delta, error, learningRate;
    /**
     * Activation function for this neuron
     */
    private Function<Double, Double> actFunc;
    /**
     * Derivative of the activation function for this neuron
     */
    private Function<Double, Double> actFuncDer;

    public Neuron(double learningRate, String id, boolean isInput) {
        this.id = id;
        this.isInput = isInput;

        this.depth = -1;

        this.output = 0.0;
        this.delta = 0.0;
        this.error = 0.0;

        this.weights = new double[0];
        this.weightAdj = new double[0];
        this.inputs = new double[0];

        this.parents = new ArrayList<Neuron>();
        this.children = new ArrayList<Neuron>();

        this.nextLayerIndicies = new ArrayList<Integer>();

        this.learningRate = learningRate;
        this.actFunc = Neuron.leakyRelu;
        this.actFuncDer = Neuron.leakyReluDerivative;
    }

    public Neuron() {
        this(Neuron.defaultLearningRate, Neuron.randId(), false);
    }

    public Neuron(double learningRate) {
        this(learningRate, Neuron.randId(), false);
    }

    public Neuron(double learningRate, String id) {
        this(learningRate, id, false);
    }

    public Neuron(double learningRate, boolean isInput) {
        this(learningRate, Neuron.randId(), isInput);
    }

    private static String randId() {
        StringBuilder randStr = new StringBuilder();
        Random rnd = new Random();

        // Generate a random lowercase string of length 'len'
        for (int len = 0; len < 14; len++) {
            randStr.append((char) ('a' + rnd.nextInt(26)));
        }

        return randStr.toString();
    }

    public String toString() {
        HashMap<String, String> map = new HashMap<>();
        map.put("id", this.id);
        map.put("depth", String.valueOf(this.depth));
        map.put("weights", Arrays.toString(this.weights));
        map.put("output", String.valueOf(this.output));
        // map.put("parents", this.parents.toString());
        // map.put("children", this.children.toString());
        return map.toString();
    }

    public boolean connectParent(Neuron parent) {
        // Check if parents contains the input
        for (Neuron n : this.parents){
            if (n == parent){
                return false;
            }
        }
        
        // Maybe adjust weights and weightAdj here, 
        //  but should be possible in adjustForInput
        this.parents.add(parent);
        // Have the parent connect this as a child
        parent.connectChild(this, true);
        return true;
    }
    
    public boolean connectNeighbor(Neuron neighbor) {
        // Check if parents contains the input
        for (Neuron n : this.parents){
            if (n == neighbor){
                System.out.println("Attempted to connect parent: " + n.getId());
                return false;
            }
        }

        for (Neuron n : this.children){
            if (n == neighbor){
                System.out.println("Attempted to connect child: " + n.getId());
                return false;
            }
        }

        if (this.isInput){
            neighbor.isInput = true;
        }

        for (Neuron n : this.parents){
            n.connectChild(neighbor);
        }

        for (Neuron n : this.children){
            n.connectParent(neighbor);
        }

        return true;
    }

    public boolean connectChild(Neuron child, boolean isFromParent) {
        // Check if children contains the input
        for (Neuron n : this.parents){
            if (n == child){
                return false;
            }
        }

        // Add to the children list
        this.children.add(child);

        if (isFromParent){
            this.nextLayerIndicies.add(child.parents.size() -1);
        } else {
            this.nextLayerIndicies.add(child.parents.size());
            // Tell the parent to connect this as a child.
            child.connectParent(this);
        }

        return true;
    }

    public void connectChild(Neuron child) {
        this.connectChild(child, false);
    }

    public void assignDepth(int depth) {
        this.depth = depth;
        // Cascade depth to children
        this.children.forEach((n) -> {
            if (n.depth < 0) {
                n.assignDepth(depth + 1);
            }
        });
    }

    public int getDepth(){
        return this.depth;
    }

    public double getOutput(){
        return this.output;
    }

    public int childCount(){
        return this.children.size();
    }

    public boolean isInput(){
        return this.isInput;
    }

    public void reset() {
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = (2.0 * Math.random()) - 1.0;
            this.weightAdj[i] = 0.0;
        }
    }

    public void adjustForInput(double[] input) {

        if (this.inputs.length != input.length) {
            this.inputs = new double[input.length + 1];
            // Set the bias
            this.inputs[0] = 1.0;
            System.arraycopy(input, 0, this.inputs, 1, input.length);
        }

        if (this.weights.length == this.inputs.length)
            return;

        System.out.println("Adjusting for Input (" +  this.id + ") existing weights: " + Arrays.toString(this.weights));

        // if (this.weights.length > 0)
        //     System.out.println("New input did not match previous input size");

        // Make them two new arrays, account for the bias
        this.weights = new double[this.inputs.length];
        // Defaults to all 0.0
        this.weightAdj = new double[this.inputs.length];

        // Create new weights and reset the adjustments
        for (int i = 0; i < this.inputs.length; i++) {
            this.weights[i] = 2.0 * Math.random() - 1.0;
        }
    }

    /**
     * Performs forward propagation using the output of the parent neurons
     */
    public double forwardProp(){
        return this.forwardProp(null);
    }

    public double forwardProp(double[] input) {

        if (input == null) {
            // System.out.println("NULL INPUT for neuron: " + this.id);
            // Overwrite the input using this neuron's parent outputs
            input = this.parents.stream().map((n) -> n.output).mapToDouble(Double::doubleValue).toArray();
            // System.out.println(this.id + " Using Input: " + Arrays.toString(input));
        } else {
            // System.out.println(this.id + " Received input: " + Arrays.toString(input));
        }
        this.output = 0.0;
        this.inputs = new double[input.length+1];
        this.inputs[0] = 1.0;
        System.arraycopy(input, 0, this.inputs, 1, input.length);

        // Don't adjust here anymore, use an external call to adjustForInput
        // this.adjustForInput(input);
        // System.out.print(this.id + " output calculation: ");
        for (int i = 0; i < this.inputs.length; i++) {
            this.output += this.inputs[i] * this.weights[i];
            // if (i > 0){
            //     System.out.print(" + ");
            // }
            // System.out.print(this.inputs[i] + " * " + this.weights[i]);
        }
        // System.out.println(" = " + this.output);

        // System.out.println(this.id + " before Activation Function: " + this.output);
        // Apply the activation function to the output
        this.output = this.actFunc.apply(this.output);
        // System.out.println(this.id + " after Activation Function: " + this.output);

        return this.output;
    }

    public void backPropOutput(double expectedOutput) {
        this.error = expectedOutput - this.output;

        this.delta = this.error * this.actFuncDer.apply(this.output);

        for (int i = 0; i < this.inputs.length; i++) {
            this.weightAdj[i] = this.inputs[i] * this.delta;
        }

    }

    public void backPropHidden() {
        double[] nextLayerDeltas = new double[this.children.size()];
        double[] nextLayerWeights = new double[this.children.size()];

        for (int i = 0; i < this.children.size(); i++) {
            nextLayerDeltas[i] = this.children.get(i).delta;
            nextLayerWeights[i] = this.children.get(i).weights[this.nextLayerIndicies.get(i) + 1];
        }

        this.delta = 0.0;

        for (int i = 0; i < nextLayerDeltas.length; i++) {
            this.delta += nextLayerDeltas[i] * nextLayerWeights[i];
        }

        this.delta *= this.actFuncDer.apply(this.output);

        for (int i = 0; i < this.inputs.length; i++) {
            this.weightAdj[i] = this.delta * this.inputs[i];
        }
    }

    public void applyAdjustments() {
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = this.weights[i] + this.weightAdj[i] * this.learningRate;
        }
    }

    public String getId() {
        return this.id;
    }

    public ArrayList<Neuron> getChildren(){
        return new ArrayList<>(this.children);
    }

    public ArrayList<Neuron> getParents(){
        return new ArrayList<>(this.parents);
    }

}