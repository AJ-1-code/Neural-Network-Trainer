import java.io.*;
import java.util.ArrayList;
import java.util.Random;

/** A small fully-connected classifier for 64 x 64 greyscale drawings. */
public class NeuralNetwork implements Serializable {

    private static final long serialVersionUID = 3L;

    private static final int INPUT_SIZE = 4096;
    private static final int HIDDEN_1_SIZE = 128;
    private static final int HIDDEN_2_SIZE = 64;

    private final int classCount;

    private final double[] hiddenLayer1 = new double[HIDDEN_1_SIZE];
    private final double[] hiddenLayer2 = new double[HIDDEN_2_SIZE];
    private final double[] outputLayer;

    private final double[][] weightsInputToHidden1 = new double[INPUT_SIZE][HIDDEN_1_SIZE];
    private final double[][] weightsHidden1ToHidden2 = new double[HIDDEN_1_SIZE][HIDDEN_2_SIZE];
    private final double[][] weightsHidden2ToOutput;

    private final double[] biasHidden1 = new double[HIDDEN_1_SIZE];
    private final double[] biasHidden2 = new double[HIDDEN_2_SIZE];
    private final double[] biasOutput;

    private final double learningRate = 0.02;

    private final ArrayList<double[]> trainingInputs = new ArrayList<>();
    private final ArrayList<Integer> trainingLabels = new ArrayList<>();

    private final Random random = new Random();

    // These let the visualizer show what happened during training.
    private int epoch;
    private double lastLoss;
    private double lastAccuracy;

    private transient Runnable trainingUpdate;

    public NeuralNetwork() {
        this(3);
    }

    public NeuralNetwork(int classCount) {
        if(classCount < 2)
            throw new IllegalArgumentException("A classifier needs at least two classes.");

        this.classCount = classCount;
        outputLayer = new double[classCount];
        weightsHidden2ToOutput = new double[HIDDEN_2_SIZE][classCount];
        biasOutput = new double[classCount];

        initializeWeights();
    }

    private void initializeWeights() {
        initializeLayer(weightsInputToHidden1, INPUT_SIZE, HIDDEN_1_SIZE);
        initializeLayer(weightsHidden1ToHidden2, HIDDEN_1_SIZE, HIDDEN_2_SIZE);
        initializeLayer(weightsHidden2ToOutput, HIDDEN_2_SIZE, classCount);
    }

    private void initializeLayer(double[][] weights, int fanIn, int fanOut) {
        double standardDeviation = Math.sqrt(2.0 / (fanIn + fanOut));

        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights[i].length; j++) {
                weights[i][j] = random.nextGaussian() * standardDeviation;
            }
        }
    }

    private double relu(double x) {
        return Math.max(0.0, x);
    }

    private double reluDerivative(double activatedValue) {
        return activatedValue > 0.0 ? 1.0 : 0.0;
    }

    public double[] predict(double[] input) {
        validateInput(input);

        for(int j = 0; j < HIDDEN_1_SIZE; j++) {
            double sum = biasHidden1[j];

            for(int i = 0; i < INPUT_SIZE; i++)
                sum += input[i] * weightsInputToHidden1[i][j];

            hiddenLayer1[j] = relu(sum);
        }

        for(int j = 0; j < HIDDEN_2_SIZE; j++) {
            double sum = biasHidden2[j];

            for(int i = 0; i < HIDDEN_1_SIZE; i++)
                sum += hiddenLayer1[i] * weightsHidden1ToHidden2[i][j];

            hiddenLayer2[j] = relu(sum);
        }

        double largestLogit = Double.NEGATIVE_INFINITY;

        for(int j = 0; j < classCount; j++) {
            double sum = biasOutput[j];

            for(int i = 0; i < HIDDEN_2_SIZE; i++)
                sum += hiddenLayer2[i] * weightsHidden2ToOutput[i][j];

            outputLayer[j] = sum;
            largestLogit = Math.max(largestLogit, sum);
        }

        double total = 0.0;

        for(int j = 0; j < classCount; j++) {
            outputLayer[j] = Math.exp(outputLayer[j] - largestLogit);
            total += outputLayer[j];
        }

        for(int j = 0; j < classCount; j++)
            outputLayer[j] /= total;

        return outputLayer.clone();
    }

    public int guess(double[] input) {
        double[] output = predict(input);
        int best = 0;

        for(int i = 1; i < output.length; i++) {
            if(output[i] > output[best])
                best = i;
        }

        return best;
    }

    public void addExample(double[] input, int label) {
        validateInput(input);
        validateLabel(label);

        trainingInputs.add(input.clone());
        trainingLabels.add(label);
    }

    public int getTrainingExampleCount() {
        return trainingInputs.size();
    }

    public void train(int epochs) {
        if(epochs < 1)
            throw new IllegalArgumentException("Epochs must be at least 1.");

        if(trainingInputs.isEmpty())
            throw new IllegalStateException("Add drawings before training.");

        for(int i = 0; i < epochs; i++)
            trainOneEpoch();
    }

    public void trainOneEpoch() {
        if(trainingInputs.isEmpty())
            throw new IllegalStateException("Add drawings before training.");

        int[] order = new int[trainingInputs.size()];

        for(int i = 0; i < order.length; i++)
            order[i] = i;

        shuffle(order);

        double loss = 0.0;
        int correct = 0;

        for(int index : order) {
            int label = trainingLabels.get(index);
            double[] probabilities = predict(trainingInputs.get(index));

            loss -= Math.log(Math.max(probabilities[label], 1e-12));

            if(argMax(probabilities) == label)
                correct++;

            backpropagate(trainingInputs.get(index), label);
        }

        lastLoss = loss / order.length;
        lastAccuracy = 100.0 * correct / order.length;
        epoch++;

        System.out.printf("Epoch %d - loss: %.4f, training accuracy: %.1f%%%n", epoch, lastLoss, lastAccuracy);

        if(trainingUpdate != null)
            trainingUpdate.run();
    }

    public void setTrainingUpdate(Runnable trainingUpdate) {
        this.trainingUpdate = trainingUpdate;
    }

    public double[] getHiddenLayer1() {
        return hiddenLayer1;
    }

    public double[] getHiddenLayer2() {
        return hiddenLayer2;
    }

    public double[] getOutputLayer() {
        return outputLayer;
    }

    public double[][] getWeightsInputToHidden1() {
        return weightsInputToHidden1;
    }

    public double[][] getWeightsHidden1ToHidden2() {
        return weightsHidden1ToHidden2;
    }

    public double[][] getWeightsHidden2ToOutput() {
        return weightsHidden2ToOutput;
    }

    public double[] getBiasHidden1() {
        return biasHidden1;
    }

    public double[] getBiasHidden2() {
        return biasHidden2;
    }

    public double[] getBiasOutput() {
        return biasOutput;
    }

    public int getEpoch() {
        return epoch;
    }

    public double getLastLoss() {
        return lastLoss;
    }

    public double getLastAccuracy() {
        return lastAccuracy;
    }

    private void backpropagate(double[] input, int label) {
        double[] outputError = outputLayer.clone();
        outputError[label] -= 1.0;

        double[] hidden2Error = new double[HIDDEN_2_SIZE];

        for(int i = 0; i < HIDDEN_2_SIZE; i++) {
            double error = 0.0;

            for(int j = 0; j < classCount; j++)
                error += outputError[j] * weightsHidden2ToOutput[i][j];

            hidden2Error[i] = error * reluDerivative(hiddenLayer2[i]);
        }

        double[] hidden1Error = new double[HIDDEN_1_SIZE];

        for(int i = 0; i < HIDDEN_1_SIZE; i++) {
            double error = 0.0;

            for(int j = 0; j < HIDDEN_2_SIZE; j++)
                error += hidden2Error[j] * weightsHidden1ToHidden2[i][j];

            hidden1Error[i] = error * reluDerivative(hiddenLayer1[i]);
        }

        for(int i = 0; i < HIDDEN_2_SIZE; i++) {
            for(int j = 0; j < classCount; j++) {
                weightsHidden2ToOutput[i][j] -= learningRate * outputError[j] * hiddenLayer2[i];
            }
        }

        for(int i = 0; i < HIDDEN_1_SIZE; i++) {
            for(int j = 0; j < HIDDEN_2_SIZE; j++) {
                weightsHidden1ToHidden2[i][j] -= learningRate * hidden2Error[j] * hiddenLayer1[i];
            }
        }

        for(int i = 0; i < INPUT_SIZE; i++) {
            for(int j = 0; j < HIDDEN_1_SIZE; j++) {
                weightsInputToHidden1[i][j] -= learningRate * hidden1Error[j] * input[i];
            }
        }

        for(int i = 0; i < classCount; i++)
            biasOutput[i] -= learningRate * outputError[i];

        for(int i = 0; i < HIDDEN_2_SIZE; i++)
            biasHidden2[i] -= learningRate * hidden2Error[i];

        for(int i = 0; i < HIDDEN_1_SIZE; i++)
            biasHidden1[i] -= learningRate * hidden1Error[i];
    }

    private void shuffle(int[] values) {
        for(int i = values.length - 1; i > 0; i--) {
            int other = random.nextInt(i + 1);
            int temporary = values[i];
            values[i] = values[other];
            values[other] = temporary;
        }
    }

    private int argMax(double[] values) {
        int best = 0;

        for(int i = 1; i < values.length; i++) {
            if(values[i] > values[best])
                best = i;
        }

        return best;
    }

    private void validateInput(double[] input) {
        if(input == null || input.length != INPUT_SIZE)
            throw new IllegalArgumentException("Each drawing must contain exactly " + INPUT_SIZE + " pixels.");
    }

    private void validateLabel(int label) {
        if(label < 0 || label >= classCount)
            throw new IllegalArgumentException("Invalid label.");
    }

    public void save(String file) throws IOException {
        try(ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file))) {
            out.writeObject(this);
        }
    }

    public static NeuralNetwork load(String file) throws IOException, ClassNotFoundException {
        try(ObjectInputStream in = new ObjectInputStream(new FileInputStream(file))) {
            return (NeuralNetwork) in.readObject();
        }
    }
}