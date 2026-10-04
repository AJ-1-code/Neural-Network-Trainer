import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;

public class NN_Visable {

    private static final int INPUT_SIZE = 4096;
    private static final int HIDDEN_1_SIZE = 128;
    private static final int HIDDEN_2_SIZE = 64;

    private static final int NEURON_SIZE = 14;
    private static final int GAP = 5;

    private static final int H1_COLS = 16;
    private static final int H2_COLS = 8;

    private double[] input;
    private double[] hidden1;
    private double[] hidden2;
    private double[] output;

    private double[][] weights1;
    private double[][] weights2;
    private double[][] weights3;

    private double[] bias1;
    private double[] bias2;
    private double[] bias3;

    private int selectedLayer = -1;
    private int selectedNeuron = -1;

    private int hoverLayer = -1;
    private int hoverNeuron = -1;

    private int hidden1X;
    private int hidden1Y;
    private int hidden2X;
    private int hidden2Y;
    private int outputX;
    private int outputY;

    private boolean networkView;

    public void update(double[] input, double[] hidden1, double[] hidden2, double[] output, double[][] weights1, double[][] weights2, double[][] weights3, double[] bias1, double[] bias2, double[] bias3) {
        this.input = input;
        this.hidden1 = hidden1;
        this.hidden2 = hidden2;
        this.output = output;

        this.weights1 = weights1;
        this.weights2 = weights2;
        this.weights3 = weights3;

        this.bias1 = bias1;
        this.bias2 = bias2;
        this.bias3 = bias3;
    }

    public void setNetworkView(boolean networkView) {
        this.networkView = networkView;

        if(!networkView)
            clearSelection();
    }

    public boolean isNetworkView() {
        return networkView;
    }

    public void clearSelection() {
        selectedLayer = -1;
        selectedNeuron = -1;
    }

    public void mouseMoved(int mouseX, int mouseY) {
    if(!networkView)
        return;

    hoverLayer = -1;
    hoverNeuron = -1;

    int neuron = findNeuron(mouseX, mouseY, hidden1X, hidden1Y, HIDDEN_1_SIZE, H1_COLS);

    if(neuron != -1) {
        hoverLayer = 1;
        hoverNeuron = neuron;
        return;
    }

    neuron = findNeuron(mouseX, mouseY, hidden2X, hidden2Y, HIDDEN_2_SIZE, H2_COLS);

    if(neuron != -1) {
        hoverLayer = 2;
        hoverNeuron = neuron;
        return;
    }

    // Output neurons use their actual positions from drawOutput().
    for(int i = 0; i < output.length; i++) {
        int y = outputY + i * 40;

        if(mouseX >= outputX - 10 && mouseX <= outputX + 120 &&
           mouseY >= y - 10 && mouseY <= y + 25) {
            hoverLayer = 3;
            hoverNeuron = i;
            return;
        }
    }
}

    public void mousePressed(int mouseX, int mouseY) {
    if(!networkView)
        return;

    mouseMoved(mouseX, mouseY);

    if(hoverLayer != -1) {
        selectedLayer = hoverLayer;
        selectedNeuron = hoverNeuron;
    } else {
        clearSelection();
    }
}

    public void draw(Graphics2D g2) {
        if(!networkView || input == null || hidden1 == null || hidden2 == null || output == null)
            return;

        hidden1X = 300;
        hidden1Y = 125;

        hidden2X = 590;
        hidden2Y = 180;

        outputX = 800;
        outputY = 245;

        drawTitle(g2);

        drawInput(g2, 55, 220);

        if(selectedLayer == -1)
            drawNormalConnections(g2);
        else
            drawSelectedConnections(g2);

        drawLayer(g2, hidden1X, hidden1Y, hidden1, HIDDEN_1_SIZE, H1_COLS, "HIDDEN 1");
        drawLayer(g2, hidden2X, hidden2Y, hidden2, HIDDEN_2_SIZE, H2_COLS, "HIDDEN 2");

        drawOutput(g2);

        if(selectedLayer != -1)
            drawInspector(g2);
        else
            drawInstructions(g2);
    }

    private void drawTitle(Graphics2D g2) {
        g2.setColor(Color.WHITE);
        g2.drawString("NEURAL NETWORK", 30, 30);
        g2.drawString("Click a neuron to inspect it", 30, 50);
    }

    private void drawInput(Graphics2D g2, int x, int y) {
        int size = 128;

        g2.setColor(Color.WHITE);
        g2.drawString("INPUT", x, y - 12);

        for(int row = 0; row < 64; row++) {
            for(int col = 0; col < 64; col++) {
                int index = row * 64 + col;
                int shade = 255 - (int)(clamp(input[index]) * 255);

                g2.setColor(new Color(shade, shade, shade));
                g2.fillRect(x + col * 2, y + row * 2, 2, 2);
            }
        }

        g2.setColor(Color.WHITE);
        g2.drawRect(x, y, size, size);
    }

    private void drawNormalConnections(Graphics2D g2) {
        // The first layer has 4096 inputs, so drawing all of those lines is just noise.
        // Instead, the input is shown separately and the interesting layer-to-layer
        // connections are shown here.
        drawLayerConnections(g2, hidden1X, hidden1Y, hidden2X, hidden2Y, weights2, HIDDEN_1_SIZE, HIDDEN_2_SIZE, H1_COLS, H2_COLS, 4);
        drawLayerConnections(g2, hidden2X, hidden2Y, outputX, outputY, weights3, HIDDEN_2_SIZE, output.length, H2_COLS, 1, 1);
    }

    private void drawLayerConnections(Graphics2D g2, int x1, int y1, int x2, int y2, double[][] weights, int inputCount, int outputCount, int inputCols, int outputCols, int sample) {
    if(weights == null)
        return;

    for(int out = 0; out < outputCount; out++) {

        int outX;
        int outY;

        if(outputCount == 3) {
            // Output neurons are spaced 40 pixels apart.
            outX = x2 + 7;
            outY = y2 + out * 40 + 7;
        } else {
            outX = x2 + getCol(out, outputCols) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;
            outY = y2 + getRow(out, outputCols) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;
        }

        for(int in = 0; in < inputCount; in += sample) {

            double weight = weights[in][out];
            double strength = Math.min(1.0, Math.abs(weight) * 2.0);

            int inX = x1 + getCol(in, inputCols) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;
            int inY = y1 + getRow(in, inputCols) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;

            if(outputCount == 3) {

                int alpha = 35 + (int)(strength * 150);

                if(weight >= 0)
                    g2.setColor(new Color(255, 255, 255, alpha));
                else
                    g2.setColor(new Color(100, 100, 100, alpha));

                g2.setStroke(new BasicStroke(0.7f + (float)strength * 1.5f));
                g2.drawLine(inX, inY, outX, outY);

            } else {

                if(strength < 0.12)
                    continue;

                drawConnection(g2, inX, inY, outX, outY, weight, strength, 35);
            }
        }
    }

    g2.setStroke(new BasicStroke(1));
}

    private void drawSelectedConnections(Graphics2D g2) {
        if(selectedLayer == 1)
            drawH1Selection(g2);
        else if(selectedLayer == 2)
            drawH2Selection(g2);
        else if(selectedLayer == 3)
            drawOutputSelection(g2);
    }

    private void drawH1Selection(Graphics2D g2) {
        int targetX = hidden1X + getCol(selectedNeuron, H1_COLS) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;
        int targetY = hidden1Y + getRow(selectedNeuron, H1_COLS) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;

        for(int i = 0; i < INPUT_SIZE; i += 4) {
            double weight = weights1[i][selectedNeuron];
            double strength = Math.min(1.0, Math.abs(weight) * 2.0);

            if(strength < 0.15)
                continue;

            int row = i / 64;
            int col = i % 64;

            drawConnection(g2, 183 + col * 2, 220 + row * 2, targetX, targetY, weight, strength, 80);
        }
    }

    private void drawH2Selection(Graphics2D g2) {
        int targetX = hidden2X + getCol(selectedNeuron, H2_COLS) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;
        int targetY = hidden2Y + getRow(selectedNeuron, H2_COLS) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;

        for(int i = 0; i < HIDDEN_1_SIZE; i++) {
            double weight = weights2[i][selectedNeuron];
            double strength = Math.min(1.0, Math.abs(weight) * 2.0);

            if(strength < 0.08)
                continue;

            int sourceX = hidden1X + getCol(i, H1_COLS) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;
            int sourceY = hidden1Y + getRow(i, H1_COLS) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;

            drawConnection(g2, sourceX, sourceY, targetX, targetY, weight, strength, 130);
        }

        for(int i = 0; i < output.length; i++) {
            double weight = weights3[selectedNeuron][i];
            double strength = Math.min(1.0, Math.abs(weight) * 2.0);

            if(strength < 0.05)
                continue;

            int targetOutputY = outputY + i * 40 + 7;

            drawConnection(g2, targetX, targetY, outputX + 7, targetOutputY, weight, strength, 130);
        }
    }

    private void drawOutputSelection(Graphics2D g2) {
        int targetX = outputX + 7;
        int targetY = outputY + selectedNeuron * 40 + 7;

        for(int i = 0; i < HIDDEN_2_SIZE; i++) {
            double weight = weights3[i][selectedNeuron];
            double strength = Math.min(1.0, Math.abs(weight) * 2.0);

            if(strength < 0.05)
                continue;

            int sourceX = hidden2X + getCol(i, H2_COLS) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;
            int sourceY = hidden2Y + getRow(i, H2_COLS) * (NEURON_SIZE + GAP) + NEURON_SIZE / 2;

            drawConnection(g2, sourceX, sourceY, targetX, targetY, weight, strength, 150);
        }
    }

    private void drawConnection(Graphics2D g2, int x1, int y1, int x2, int y2, double weight, double strength, int baseAlpha) {
        int alpha = Math.min(220, baseAlpha + (int)(strength * 100));

        if(weight > 0)
            g2.setColor(new Color(255, 255, 255, alpha));
        else
            g2.setColor(new Color(90, 90, 90, alpha));

        g2.setStroke(new BasicStroke(0.5f + (float)strength * 1.5f));
        g2.drawLine(x1, y1, x2, y2);
        g2.setStroke(new BasicStroke(1));
    }

    private void drawLayer(Graphics2D g2, int x, int y, double[] values, int count, int cols, String name) {
        g2.setColor(Color.WHITE);
        g2.drawString(name, x, y - 15);

        for(int i = 0; i < count; i++) {
            int nx = x + getCol(i, cols) * (NEURON_SIZE + GAP);
            int ny = y + getRow(i, cols) * (NEURON_SIZE + GAP);

            drawNeuron(g2, nx, ny, values[i], i, 1 + (name.equals("HIDDEN 2") ? 1 : 0));
        }
    }

    private void drawNeuron(Graphics2D g2, int x, int y, double value, int index, int layer) {
        // ReLU activations can be larger than 1, so normalize them for display.
        double brightnessValue = value / (1.0 + value);
        int brightness = 25 + (int)(brightnessValue * 230);

        boolean selected = selectedLayer == layer && selectedNeuron == index;
        boolean hovered = hoverLayer == layer && hoverNeuron == index;

        g2.setColor(new Color(brightness, brightness, brightness));
        g2.fillOval(x, y, NEURON_SIZE, NEURON_SIZE);

        if(hovered || selected) {
            g2.setColor(Color.WHITE);
            g2.setStroke(new BasicStroke(selected ? 3 : 2));
            g2.drawOval(x - 3, y - 3, NEURON_SIZE + 6, NEURON_SIZE + 6);
            g2.setStroke(new BasicStroke(1));
        }
    }

    private void drawOutput(Graphics2D g2) {
        String[] names = {"LEFT", "RIGHT", "UP"};

        g2.setColor(Color.WHITE);
        g2.drawString("OUTPUT", outputX, outputY - 20);

        for(int i = 0; i < output.length; i++) {
            double value = clamp(output[i]);
            int currentY = outputY + i * 40;

            int brightness = 25 + (int)(value * 230);

            g2.setColor(new Color(brightness, brightness, brightness));
            g2.fillOval(outputX, currentY, 14, 14);

            if(selectedLayer == 3 && selectedNeuron == i || hoverLayer == 3 && hoverNeuron == i) {
                g2.setColor(Color.WHITE);
                g2.setStroke(new BasicStroke(2));
                g2.drawOval(outputX - 3, currentY - 3, 20, 20);
                g2.setStroke(new BasicStroke(1));
            }

            g2.setColor(Color.WHITE);
            g2.drawString(names[i], outputX + 25, currentY + 11);
            g2.drawString(String.format("%.3f", value), outputX + 90, currentY + 11);
        }
    }

    private void drawInspector(Graphics2D g2) {
        int x = 930;
        int y = 70;

        g2.setColor(Color.WHITE);
        g2.drawRect(x, y, 185, 480);

        g2.drawString("NEURON INFO", x + 12, y + 25);
        g2.drawString(getLayerName() + "[" + selectedNeuron + "]", x + 12, y + 48);

        double activation = getSelectedActivation();
        double bias = getSelectedBias();

        g2.drawString("Activation: " + String.format("%.4f", activation), x + 12, y + 75);
        g2.drawString("Bias: " + String.format("%.4f", bias), x + 12, y + 95);

        if(selectedLayer == 1) {
            drawWeightMap(g2, x + 28, y + 125);
            g2.drawString("Input weights", x + 12, y + 290);
            g2.drawString("white = positive", x + 12, y + 310);
            g2.drawString("gray = negative", x + 12, y + 330);
        } else {
            drawStrongestWeights(g2, x + 12, y + 130);
        }
    }

    private void drawWeightMap(Graphics2D g2, int x, int y) {
        double max = 0;

        for(int i = 0; i < INPUT_SIZE; i++)
            max = Math.max(max, Math.abs(weights1[i][selectedNeuron]));

        if(max == 0)
            max = 1;

        int size = 128;

        for(int row = 0; row < 64; row++) {
            for(int col = 0; col < 64; col++) {
                double weight = weights1[row * 64 + col][selectedNeuron];
                double value = weight / max;

                int shade = 128 + (int)(value * 127);
                shade = Math.max(0, Math.min(255, shade));

                g2.setColor(new Color(shade, shade, shade));
                g2.fillRect(x + col * 2, y + row * 2, 2, 2);
            }
        }

        g2.setColor(Color.WHITE);
        g2.drawRect(x, y, size, size);
    }

    private void drawStrongestWeights(Graphics2D g2, int x, int y) {
        double[][] weights = selectedLayer == 2 ? weights2 : weights3;
        int count = weights.length;

        double[] strongest = new double[6];
        int[] indexes = new int[6];

        for(int i = 0; i < count; i++) {
            double weight = selectedLayer == 2 ? weights[i][selectedNeuron] : weights[i][selectedNeuron];

            for(int slot = 0; slot < strongest.length; slot++) {
                if(Math.abs(weight) > Math.abs(strongest[slot])) {
                    for(int move = strongest.length - 1; move > slot; move--) {
                        strongest[move] = strongest[move - 1];
                        indexes[move] = indexes[move - 1];
                    }

                    strongest[slot] = weight;
                    indexes[slot] = i;
                    break;
                }
            }
        }

        g2.setColor(Color.WHITE);
        g2.drawString("Strongest weights:", x, y);

        for(int i = 0; i < strongest.length; i++) {
            g2.drawString(
                indexes[i] + ": " + String.format("%+.4f", strongest[i]),
                x,
                y + 25 + i * 25
            );
        }
    }

    private void drawInstructions(Graphics2D g2) {
        g2.setColor(Color.WHITE);
        g2.drawString("Click a neuron to inspect it.", 930, 100);
        g2.drawString("Selected H1 neurons show", 930, 125);
        g2.drawString("their 64x64 weight map.", 930, 145);
    }

    private int findNeuron(int mouseX, int mouseY, int x, int y, int count, int cols) {
        for(int i = 0; i < count; i++) {
            int nx = x + getCol(i, cols) * (NEURON_SIZE + GAP);
            int ny = y + getRow(i, cols) * (NEURON_SIZE + GAP);

            Rectangle area = new Rectangle(nx - 5, ny - 5, NEURON_SIZE + 10, NEURON_SIZE + 10);

            if(area.contains(mouseX, mouseY))
                return i;
        }

        return -1;
    }

    private int getCol(int index, int cols) {
        return index % cols;
    }

    private int getRow(int index, int cols) {
        return index / cols;
    }

    private String getLayerName() {
        if(selectedLayer == 1)
            return "HIDDEN 1";

        if(selectedLayer == 2)
            return "HIDDEN 2";

        return "OUTPUT";
    }

    private double getSelectedActivation() {
        if(selectedLayer == 1)
            return hidden1[selectedNeuron];

        if(selectedLayer == 2)
            return hidden2[selectedNeuron];

        return output[selectedNeuron];
    }

    private double getSelectedBias() {
        if(selectedLayer == 1)
            return bias1[selectedNeuron];

        if(selectedLayer == 2)
            return bias2[selectedNeuron];

        return bias3[selectedNeuron];
    }

    private double clamp(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }
}