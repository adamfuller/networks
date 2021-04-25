import java.util.Arrays;

public class App {
    public static void main(String[] args) throws Exception {
        Network net = new Network();

        double lf = 0.033;

        // Input neurons
        Neuron n1 = new Neuron(lf, "N1", true);
        Neuron n2 = new Neuron(lf, "N2", true);

        // Hidden neurons
        Neuron n3 = new Neuron(lf, "N3");
        Neuron n4 = new Neuron(lf, "N4");

        // Output neurons
        Neuron n5 = new Neuron(lf, "N5");

        n1.connectChild(n3);
        n1.connectChild(n4);

        n2.connectChild(n3);
        n2.connectChild(n4);

        n3.connectChild(n5);
        n4.connectChild(n5);

        // Attach the neurons to the network
        net.attachNeuron(n1);
        net.attachNeuron(n2);
        net.attachNeuron(n3);
        net.attachNeuron(n4);
        net.attachNeuron(n5);

        net.buildNeuronOrder();

        // System.out.println("Before Training:");
        // System.out.println("(0,0 -> 0): " + Arrays.toString(net.forwardProp(new double[] { 0.0, 0.0 })));
        // System.out.println("(0,1 -> 1): " + Arrays.toString(net.forwardProp(new double[] { 0.0, 1.0 })));
        // System.out.println("(1,0 -> 1): " + Arrays.toString(net.forwardProp(new double[] { 1.0, 0.0 })));
        // System.out.println("(1,1 -> 0): " + Arrays.toString(net.forwardProp(new double[] { 1.0, 1.0 })));

        System.out.println("To String: \n" + net.toString());
        net.forwardProp(new double[] { 0.0, 0.0 });
        System.out.println("To String {0.0, 0.0}: \n" + net.toString());

        net.forwardProp(new double[] { 1.0, 0.0 });
        System.out.println("To String {1.0, 0.0}: \n" + net.toString());
        
        net.forwardProp(new double[] { 1.0, 1.0 });
        System.out.println("To String {1.0, 1.0}: \n" + net.toString());
        
        for (int i = 0; i < 1000; i++) {
            net.forwardProp(new double[] { 0.0, 0.0 });
            net.backwardProp(new double[] { 0.0 });
            // 0,1
            net.forwardProp(new double[] { 0.0, 1.0 });
            net.backwardProp(new double[] { 1.0 });
            // 1,0
            net.forwardProp(new double[] { 1.0, 0.0 });
            net.backwardProp(new double[] { 1.0 });
            // 1,1
            net.forwardProp(new double[] { 1.0, 1.0 });
            net.backwardProp(new double[] { 0.0 });

            // System.out.println("Round: " + i);
            // System.out.println("N1: " + n1.getOutput());
            // System.out.println("N2: " + n2.getOutput());
            // System.out.println("N3: " + n3.getOutput());
            // System.out.println("N4: " + n4.getOutput());
            // System.out.println("N5: " + n5.getOutput());
            
        }
        
        System.out.println("After Training:");
        System.out.println("(0,0 -> 0): " + Arrays.toString(net.forwardProp(new double[] { 0.0, 0.0 })));
        System.out.println("(0,1 -> 1): " + Arrays.toString(net.forwardProp(new double[] { 0.0, 1.0 })));
        System.out.println("(1,0 -> 1): " + Arrays.toString(net.forwardProp(new double[] { 1.0, 0.0 })));
        System.out.println("(1,1 -> 0): " + Arrays.toString(net.forwardProp(new double[] { 1.0, 1.0 })));
        System.out.println("To String: \n" + net.toString());

        // System.out.println("N1: " + n1.getOutput());
        // System.out.println("N2: " + n2.getOutput());
        // System.out.println("N3: " + n3.getOutput());
        // System.out.println("N4: " + n4.getOutput());
        // System.out.println("N5: " + n5.getOutput());
    }
}
