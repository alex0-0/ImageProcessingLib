package edu.umb.cs.imageprocessinglib.tensorflow;

import edu.umb.cs.imageprocessinglib.model.BoundingBox;
import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.GraphBuilder;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.apache.commons.math3.analysis.function.Sigmoid;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * YOLOClassifier class implemented in Java by using the TensorFlow Java API
 */
public class TensorFlowYoloDetector implements Classifier {
    private final static float OVERLAP_THRESHOLD = 0.5f;
    private final static double anchors[] = {1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52};
    private final static int SIZE = 13;
    private final static int MAX_RECOGNIZED_CLASSES = 24;
    private final static float THRESHOLD = 0.5f;
    private final static int MAX_RESULTS = 24;
    private final static int NUMBER_OF_BOUNDING_BOX = 5;
    private float MEAN = 255f;

    // Config values.
    private String inputName;
    private int inputSize;

    // Pre-allocated buffers.
//    private int[] intValues;
//    private float[] floatValues;
    private String[] outputNames;
    private String modelFileName;

    private int blockSize;

    private static final String[] LABELS = {
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
    };

    private TensorFlowYoloDetector() {}

    public static Classifier create(
            final String modelFilename,
            final int inputSize,
            final String inputName,
            final String outputName,
            final int blockSize) {
        TensorFlowYoloDetector d = new TensorFlowYoloDetector();
        d.inputName = inputName;
        d.inputSize = inputSize;

        // Pre-allocate buffers.
        d.outputNames = outputName.split(",");
//        d.intValues = new int[inputSize * inputSize];
//        d.floatValues = new float[inputSize * inputSize * 3];
        d.blockSize = blockSize;
        d.modelFileName = modelFilename;


        return d;
    }

    /**
     * Executes graph on the given preprocessed image
     * @param image preprocessed image
     * @return output tensor returned by tensorFlow
     */
    private float[] executeYOLOGraph(final Tensor<Float> image) {
        try (Graph graph = new Graph()) {
            byte[] model = new byte[0];
            model = ImageUtil.extractBytes(modelFileName, TensorFlowYoloDetector.class);
            graph.importGraphDef(model);
            try (Session s = new Session(graph);
                 Tensor<Float> result = s.runner().feed(inputName, image).fetch(outputNames[0]).run().get(0).expect(Float.class)) {
                float[] outputTensor = new float[getOutputSizeByShape(result)];
                FloatBuffer floatBuffer = FloatBuffer.wrap(outputTensor);
                result.writeTo(floatBuffer);
                return outputTensor;
            }
        }
    }

    /**
     * Gets the number of classes based on the tensor shape
     *
     * @param result - the tensorflow output
     * @return the number of classes
     */
    public int getOutputSizeByShape(Tensor<Float> result) {
        return (int) (result.shape()[3] * Math.pow(SIZE,2));
    }

    /**
     * It classifies the object/objects on the image
     *
     * @param tensorFlowOutput output from the TensorFlow, it is a 13x13x((num_class +1) * 5) tensor
     * 125 = (numClass +  Tx, Ty, Tw, Th, To) * 5 - cause we have 5 boxes per each cell
     * @param labels a string vector with the labels
     * @return a list of recognition objects
     */
    public List<Recognition> classifyImage(final float[] tensorFlowOutput, final String[] labels) {
        int numClass = (int) (tensorFlowOutput.length / (Math.pow(SIZE,2) * NUMBER_OF_BOUNDING_BOX) - 5);
        BoundingBox[][][] boundingBoxPerCell = new BoundingBox[SIZE][SIZE][NUMBER_OF_BOUNDING_BOX];
        PriorityQueue<Recognition> priorityQueue = new PriorityQueue(MAX_RECOGNIZED_CLASSES, new RecognitionComparator());

        int offset = 0;
        for (int cy=0; cy<SIZE; cy++) {        // SIZE * SIZE cells
            for (int cx=0; cx<SIZE; cx++) {
                for (int b=0; b<NUMBER_OF_BOUNDING_BOX; b++) {   // 5 bounding boxes per each cell
                    boundingBoxPerCell[cx][cy][b] = getModel(tensorFlowOutput, cx, cy, b, numClass, offset);
                    calculateTopPredictions(boundingBoxPerCell[cx][cy][b], priorityQueue, labels);
                    offset = offset + numClass + 5;
                }
            }
        }

        return getRecognition(priorityQueue);
    }

    private BoundingBox getModel(final float[] tensorFlowOutput, int cx, int cy, int b, int numClass, int offset) {
        BoundingBox model = new BoundingBox();
        Sigmoid sigmoid = new Sigmoid();
        model.setX((cx + sigmoid.value(tensorFlowOutput[offset])) * blockSize);
        model.setY((cy + sigmoid.value(tensorFlowOutput[offset + 1])) * blockSize);
        model.setWidth(Math.exp(tensorFlowOutput[offset + 2]) * anchors[2 * b] * blockSize);
        model.setHeight(Math.exp(tensorFlowOutput[offset + 3]) * anchors[2 * b + 1] * blockSize);
        model.setConfidence(sigmoid.value(tensorFlowOutput[offset + 4]));

        model.setClasses(new double[numClass]);

        for (int probIndex=0; probIndex<numClass; probIndex++) {
            model.getClasses()[probIndex] = tensorFlowOutput[probIndex + offset + 5];
        }

        return model;
    }

    private double[] getSoftMax(double[] params) {
        double sum = 0;

        for (int i=0; i<params.length; i++) {
            params[i] = Math.exp(params[i]);
            sum += params[i];
        }

        if (Double.isNaN(sum) || sum < 0) {
            for (int i=0; i<params.length; i++) {
                params[i] = 1.0 / params.length;
            }
        } else {
            for (int i=0; i<params.length; i++) {
                params[i] = params[i] / sum;
            }
        }

        return params;
    }

    private void calculateTopPredictions(final BoundingBox boundingBox, final PriorityQueue<Recognition> predictionQueue,
                                         final String[] labels) {
        for (int i=0; i<boundingBox.getClasses().length; i++) {
            double[] results = getSoftMax(boundingBox.getClasses());

            //get max result
            int maxIndex = 0;
            for (int k = 0; k < results.length; k++)
                if (results[k] > results[maxIndex])
                    maxIndex = k;

            double confidenceInClass = results[maxIndex] * boundingBox.getConfidence();
            if (confidenceInClass > THRESHOLD) {
                predictionQueue.add(new Recognition(maxIndex, labels[maxIndex], (float) confidenceInClass,
                        new BoxPosition((float) (boundingBox.getX() - boundingBox.getWidth() / 2),
                                (float) (boundingBox.getY() - boundingBox.getHeight() / 2),
                                (float) boundingBox.getWidth(),
                                (float) boundingBox.getHeight())));
            }
        }
    }

    private List<Recognition> getRecognition(final PriorityQueue<Recognition> priorityQueue) {
        List<Recognition> recognitions = new ArrayList();

        if (priorityQueue.size() > 0) {
            // Best recognition
            Recognition bestRecognition = priorityQueue.poll();
            recognitions.add(bestRecognition);

            for (int i = 0; i < Math.min(priorityQueue.size(), MAX_RESULTS); ++i) {
                Recognition recognition = priorityQueue.poll();
                boolean overlaps = false;
                for (Recognition previousRecognition : recognitions) {
                    overlaps = overlaps || (getIntersectionProportion(previousRecognition.getLocation(),
                            recognition.getLocation()) > OVERLAP_THRESHOLD);
                }

                if (!overlaps) {
                    recognitions.add(recognition);
                }
            }
        }

        return recognitions;
    }

    private float getIntersectionProportion(BoxPosition primaryShape, BoxPosition secondaryShape) {
        if (overlaps(primaryShape, secondaryShape)) {
            float intersectionSurface = Math.max(0, Math.min(primaryShape.getRight(), secondaryShape.getRight()) - Math.max(primaryShape.getLeft(), secondaryShape.getLeft())) *
                    Math.max(0, Math.min(primaryShape.getBottom(), secondaryShape.getBottom()) - Math.max(primaryShape.getTop(), secondaryShape.getTop()));

            float surfacePrimary = Math.abs(primaryShape.getRight() - primaryShape.getLeft()) * Math.abs(primaryShape.getBottom() - primaryShape.getTop());

            return intersectionSurface / surfacePrimary;
        }

        return 0f;

    }

    private boolean overlaps(BoxPosition primary, BoxPosition secondary) {
        return primary.getLeft() < secondary.getRight() && primary.getRight() > secondary.getLeft()
                && primary.getTop() < secondary.getBottom() && primary.getBottom() > secondary.getTop();
    }

    // Intentionally reversed to put high confidence at the head of the queue.
    private class RecognitionComparator implements Comparator<Recognition> {
        @Override
        public int compare(final Recognition recognition1, final Recognition recognition2) {
            return Float.compare(recognition2.getConfidence(), recognition1.getConfidence());
        }
    }

    /**
     * Pre-process input. It resize the image and normalize its pixels
     * @param imageBytes Input image
     * @return Tensor<Float> with shape [1][416][416][3]
     */
    private Tensor<Float> normalizeImage(final byte[] imageBytes) {
        int size = 416;
        try (Graph graph = new Graph()) {
            GraphBuilder graphBuilder = new GraphBuilder(graph);

            final Output<Float> output =
                    graphBuilder.div( // Divide each pixels with the MEAN
                            graphBuilder.resizeBilinear( // Resize using bilinear interpolation
                                    graphBuilder.expandDims( // Increase the output tensors dimension
                                            graphBuilder.cast( // Cast the output to Float
                                                    graphBuilder.decodeJpeg(
                                                            graphBuilder.constant("input", imageBytes), 3),
                                                    Float.class),
                                            graphBuilder.constant("make_batch", 0)),
                                    graphBuilder.constant("size", new int[]{inputSize, inputSize})),
                            graphBuilder.constant("scale", MEAN));

            try (Session session = new Session(graph)) {
                return session.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
            }
        }
    }

    @Override
    public List<Recognition> recognizeImage(BufferedImage img) {
        List<Recognition> recognitions;
        ByteArrayOutputStream baos=new ByteArrayOutputStream();
        try {
            ImageIO.write(img, "jpg", baos );
        } catch (IOException e) {
            e.printStackTrace();
        }
        byte[] data=baos.toByteArray();
        try (Tensor<Float> normalizedImage = normalizeImage(data)) {
            recognitions = classifyImage(executeYOLOGraph(normalizedImage), LABELS);
        }
        return recognitions;
    }
}

