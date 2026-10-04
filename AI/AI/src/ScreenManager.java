import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutionException;
import javax.swing.*;

public class ScreenManager extends JPanel implements MouseListener, MouseMotionListener {

    private static final int TILE_SIZE = 10;
    private static final int ROWS = 64;
    private static final int COLS = 64;
    private static final double INK_THRESHOLD = 0.05;
    private static final String MODEL_FILE = "shape-classifier.model";

    private final double[][] pixels = new double[COLS][ROWS];
    private final String[] labels = {"LEFT ARROW", "RIGHT ARROW", "UP ARROW"};

    private NeuralNetwork net;
    private final NN_Visable nn_vis;

    private final JButton modeButton = new JButton("Mode: TRAIN");
    private final JButton labelButton = new JButton("Label: LEFT ARROW");
    private final JButton trainButton = new JButton("Train (200 epochs)");
    private final JButton viewButton = new JButton("View: DRAWING");
    private final JLabel statusLabel = new JLabel("Draw a shape, then press Enter to add it.");

    private boolean mouseDown;
    private boolean trainMode = true;
    private boolean training;
    private boolean networkView;

    private int currentLabel;
    private int previousCol = -1;
    private int previousRow = -1;

    public ScreenManager(NeuralNetwork net) {
        this.net = net;
        nn_vis = new NN_Visable();

        net.setTrainingUpdate(() -> SwingUtilities.invokeLater(this::repaint));

        JFrame window = new JFrame("Shape Classifier");
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setResizable(false);

        JPanel buttons = new JPanel();

        buttons.add(modeButton);
        buttons.add(labelButton);
        buttons.add(trainButton);
        buttons.add(viewButton);
        buttons.add(statusLabel);

        JPanel main = new JPanel(new BorderLayout());

        main.add(buttons, BorderLayout.NORTH);
        main.add(this, BorderLayout.CENTER);

        window.add(main);

        setPreferredSize(new Dimension(1140, ROWS * TILE_SIZE));
        setBackground(Color.BLACK);

        addMouseListener(this);
        addMouseMotionListener(this);

        installKeyBindings();

        modeButton.addActionListener(e -> toggleMode());
        labelButton.addActionListener(e -> changeLabel());
        trainButton.addActionListener(e -> trainNetwork());
        viewButton.addActionListener(e -> toggleView());

        window.pack();
        window.setLocationRelativeTo(null);
        window.setVisible(true);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new ScreenManager(new NeuralNetwork()));
    }

    private void installKeyBindings() {
        InputMap inputMap = getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
        ActionMap actionMap = getActionMap();

        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, 0), "submit-drawing");
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_C, 0), "clear-drawing");
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_L, 0), "load-model");
        inputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_S, 0), "step-training");

        actionMap.put("submit-drawing", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent event) {
                submitDrawing();
            }
        });

        actionMap.put("clear-drawing", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent event) {
                clear();
            }
        });

        actionMap.put("load-model", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent event) {
                loadNetwork();
            }
        });

        actionMap.put("step-training", new AbstractAction() {
            @Override public void actionPerformed(ActionEvent event) {
                stepTraining();
            }
        });
    }

    private void toggleMode() {
        trainMode = !trainMode;

        modeButton.setText(trainMode ? "Mode: TRAIN" : "Mode: TEST");
        labelButton.setEnabled(trainMode);
        trainButton.setEnabled(trainMode);

        showDrawing();

        statusLabel.setText(
            trainMode
                ? "Training mode: choose a label and press Enter."
                : "Test mode: draw and press Enter."
        );

        clear();
    }

    private void toggleView() {
        if(training)
            return;

        if(networkView)
            showDrawing();
        else
            showNetwork();
    }

    private void showDrawing() {
        networkView = false;
        nn_vis.setNetworkView(false);
        viewButton.setText("View: DRAWING");
        repaint();
    }

    private void showNetwork() {
        networkView = true;
        nn_vis.setNetworkView(true);
        viewButton.setText("View: NETWORK");
        repaint();
    }

    private void changeLabel() {
        currentLabel = (currentLabel + 1) % labels.length;
        labelButton.setText("Label: " + labels[currentLabel]);
    }

    private void submitDrawing() {
        if(networkView || isBlank()) {
            if(isBlank())
                statusLabel.setText("Draw something first.");

            return;
        }

        double[] input = getInputArray();

        if(trainMode) {
            net.addExample(input, currentLabel);

            statusLabel.setText(
                "Added " + labels[currentLabel] +
                " (" + net.getTrainingExampleCount() +
                " samples total)."
            );
        } else {
            try {
                double[] probabilities = net.predict(input);
                int guess = net.guess(input);

                if(guess < 0 || guess >= labels.length) {
                    statusLabel.setText("Model returned an invalid class index: " + guess);
                } else {
                    statusLabel.setText(
                        String.format(
                            "Prediction: %s (%.0f%% confident)",
                            labels[guess],
                            probabilities[guess] * 100
                        )
                    );
                }
            } catch(Exception exception) {
                statusLabel.setText("Prediction failed: " + exception.getMessage());
                exception.printStackTrace();
            }
        }

        clear();
    }

    private void trainNetwork() {
        if(net.getTrainingExampleCount() == 0) {
            statusLabel.setText("Add labelled drawings before training.");
            return;
        }

        showNetwork();

        trainButton.setEnabled(false);
        modeButton.setEnabled(false);
        labelButton.setEnabled(false);
        viewButton.setEnabled(false);

        training = true;
        statusLabel.setText("Training " + net.getTrainingExampleCount() + " drawings...");

        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() {
                net.train(200);

                try {
                    net.save(MODEL_FILE);
                } catch(IOException exception) {
                    throw new IllegalStateException(
                        "Could not save model to " + MODEL_FILE,
                        exception
                    );
                }

                return null;
            }

            @Override
            protected void done() {
                training = false;

                trainButton.setEnabled(true);
                modeButton.setEnabled(true);
                labelButton.setEnabled(trainMode);
                viewButton.setEnabled(true);

                try {
                    get();

                    statusLabel.setText(
                        "Training complete and saved to " +
                        MODEL_FILE +
                        ". Press S to train one epoch."
                    );
                } catch(InterruptedException exception) {
                    Thread.currentThread().interrupt();
                    statusLabel.setText("Training was interrupted; model was not saved.");
                } catch(ExecutionException exception) {
                    statusLabel.setText(
                        "Training failed: " +
                        exception.getCause().getMessage()
                    );
                }

                repaint();
            }
        }.execute();
    }

    private void stepTraining() {
        if(training)
            return;

        if(net.getTrainingExampleCount() == 0) {
            statusLabel.setText("Add drawings before training.");
            return;
        }

        showNetwork();

        training = true;

        trainButton.setEnabled(false);
        modeButton.setEnabled(false);
        labelButton.setEnabled(false);
        viewButton.setEnabled(false);

        statusLabel.setText("Training one epoch...");

        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() {
                net.trainOneEpoch();

                try {
                    net.save(MODEL_FILE);
                } catch(IOException exception) {
                    throw new IllegalStateException(
                        "Could not save model.",
                        exception
                    );
                }

                return null;
            }

            @Override
            protected void done() {
                training = false;

                trainButton.setEnabled(true);
                modeButton.setEnabled(true);
                labelButton.setEnabled(trainMode);
                viewButton.setEnabled(true);

                try {
                    get();

                    statusLabel.setText(
                        "Epoch " + net.getEpoch() +
                        " complete. Press S to train another epoch."
                    );
                } catch(InterruptedException exception) {
                    Thread.currentThread().interrupt();
                    statusLabel.setText("Training interrupted.");
                } catch(ExecutionException exception) {
                    statusLabel.setText(
                        "Training failed: " +
                        exception.getCause().getMessage()
                    );
                }

                repaint();
            }
        }.execute();
    }

    private void loadNetwork() {
        if(training) {
            statusLabel.setText("Wait for training to finish before loading a model.");
            return;
        }

        File modelFile = new File(MODEL_FILE);

        if(!modelFile.isFile()) {
            statusLabel.setText("No saved model found. Train once first.");
            return;
        }

        try {
            net = NeuralNetwork.load(MODEL_FILE);

            net.setTrainingUpdate(() -> SwingUtilities.invokeLater(this::repaint));

            statusLabel.setText(
                "Loaded saved model. Switch to TEST and press Enter to guess."
            );

            repaint();
        } catch(IOException | ClassNotFoundException exception) {
            statusLabel.setText(
                "Could not load saved model: " +
                exception.getMessage()
            );
        }
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);

        Graphics2D g2 = (Graphics2D) graphics;

        if(!networkView) {
            drawCanvas(g2);
            return;
        }

        double[] input = getInputArray();

        nn_vis.update(
            input,
            net.getHiddenLayer1(),
            net.getHiddenLayer2(),
            net.getOutputLayer(),
            net.getWeightsInputToHidden1(),
            net.getWeightsHidden1ToHidden2(),
            net.getWeightsHidden2ToOutput(),
            net.getBiasHidden1(),
            net.getBiasHidden2(),
            net.getBiasOutput()
        );

        nn_vis.draw(g2);
    }

    private void drawCanvas(Graphics2D g2) {
        for(int row = 0; row < ROWS; row++) {
            for(int col = 0; col < COLS; col++) {
                int x = col * TILE_SIZE;
                int y = row * TILE_SIZE;

                g2.setColor(greyFor(pixels[col][row]));
                g2.fillRect(x, y, TILE_SIZE, TILE_SIZE);
            }
        }

        g2.setColor(new Color(0, 0, 0, 40));

        for(int row = 0; row <= ROWS; row++)
            g2.drawLine(0, row * TILE_SIZE, COLS * TILE_SIZE, row * TILE_SIZE);

        for(int col = 0; col <= COLS; col++)
            g2.drawLine(col * TILE_SIZE, 0, col * TILE_SIZE, ROWS * TILE_SIZE);
    }

    private Color greyFor(double intensity) {
        int shade = 255 - (int)Math.round(255 * clamp01(intensity));
        return new Color(shade, shade, shade);
    }

    private double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }

    public double[] getInputArray() {
        int minCol = COLS;
        int maxCol = -1;
        int minRow = ROWS;
        int maxRow = -1;

        for(int row = 0; row < ROWS; row++) {
            for(int col = 0; col < COLS; col++) {
                if(pixels[col][row] > INK_THRESHOLD) {
                    minCol = Math.min(minCol, col);
                    maxCol = Math.max(maxCol, col);
                    minRow = Math.min(minRow, row);
                    maxRow = Math.max(maxRow, row);
                }
            }
        }

        double[] input = new double[ROWS * COLS];

        if(maxCol == -1)
            return input;

        int colShift = Math.round((COLS - 1 - minCol - maxCol) / 2.0f);
        int rowShift = Math.round((ROWS - 1 - minRow - maxRow) / 2.0f);

        for(int row = 0; row < ROWS; row++) {
            for(int col = 0; col < COLS; col++) {
                int sourceCol = col - colShift;
                int sourceRow = row - rowShift;

                if(sourceCol >= 0 && sourceCol < COLS && sourceRow >= 0 && sourceRow < ROWS)
                    input[row * COLS + col] = pixels[sourceCol][sourceRow];
            }
        }

        return input;
    }

    private boolean isBlank() {
        for(double[] column : pixels) {
            for(double pixel : column) {
                if(pixel > INK_THRESHOLD)
                    return false;
            }
        }

        return true;
    }

    public void clear() {
        for(int row = 0; row < ROWS; row++) {
            for(int col = 0; col < COLS; col++)
                pixels[col][row] = 0.0;
        }

        repaint();
    }

    private void draw(MouseEvent event) {
        int col = event.getX() / TILE_SIZE;
        int row = event.getY() / TILE_SIZE;

        if(col < 0 || col >= COLS || row < 0 || row >= ROWS)
            return;

        if(previousCol >= 0)
            drawLine(previousCol, previousRow, col, row);
        else
            paintBrush(col, row);

        previousCol = col;
        previousRow = row;

        repaint();
    }

    private void drawLine(int startCol, int startRow, int endCol, int endRow) {
        int steps = Math.max(
            Math.abs(endCol - startCol),
            Math.abs(endRow - startRow)
        );

        for(int step = 0; step <= steps; step++) {
            int col = Math.round(
                startCol +
                (endCol - startCol) *
                step /
                (float)Math.max(steps, 1)
            );

            int row = Math.round(
                startRow +
                (endRow - startRow) *
                step /
                (float)Math.max(steps, 1)
            );

            paintBrush(col, row);
        }
    }

    private void paintBrush(int col, int row) {
        int radius = 2;

        for(int y = row - radius; y <= row + radius; y++) {
            for(int x = col - radius; x <= col + radius; x++) {
                if(x < 0 || x >= COLS || y < 0 || y >= ROWS)
                    continue;

                double distance = Math.sqrt(
                    (x - col) * (x - col) +
                    (y - row) * (y - row)
                );

                if(distance > radius)
                    continue;

                double falloff = 1.0 - distance / (radius + 1);
                pixels[x][y] = clamp01(
                    Math.max(pixels[x][y], falloff)
                );
            }
        }
    }

    @Override
    public void mousePressed(MouseEvent event) {
        if(networkView) {
            if(SwingUtilities.isLeftMouseButton(event))
                nn_vis.mousePressed(event.getX(), event.getY());

            repaint();
            return;
        }

        if(SwingUtilities.isRightMouseButton(event)) {
            clear();
            return;
        }

        if(SwingUtilities.isLeftMouseButton(event)) {
            mouseDown = true;
            previousCol = -1;
            previousRow = -1;
            draw(event);
        }
    }

    @Override
    public void mouseDragged(MouseEvent event) {
        if(!networkView && mouseDown)
            draw(event);
    }

    @Override
    public void mouseMoved(MouseEvent event) {
        if(networkView) {
            nn_vis.mouseMoved(event.getX(), event.getY());
            repaint();
        }
    }

    @Override
    public void mouseReleased(MouseEvent event) {
        mouseDown = false;
        previousCol = -1;
        previousRow = -1;
    }

    @Override public void mouseClicked(MouseEvent event) {}
    @Override public void mouseEntered(MouseEvent event) {}
    @Override public void mouseExited(MouseEvent event) {}
}