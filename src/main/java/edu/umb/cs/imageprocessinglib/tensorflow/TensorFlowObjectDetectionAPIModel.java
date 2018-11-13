package edu.umb.cs.imageprocessinglib.tensorflow;

import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class TensorFlowObjectDetectionAPIModel implements Classifier {
  // Only return this many results.
  private static final int MAX_RESULTS = 100;

  // Config values.
//  private String inputName;
  private int inputSize;

  // Pre-allocated buffers.
  private String[] labels;
  private SavedModelBundle model;
  private String modelPath;
  private String labelPath;

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param modlePath The filepath of the model GraphDef protocol buffer.
   * @param lablePath The filepath of label file for classes.
   */
  public static Classifier create(
      final String modlePath,
      final String lablePath,
      final int inputSize) throws Exception {
    final TensorFlowObjectDetectionAPIModel d = new TensorFlowObjectDetectionAPIModel();
    d.modelPath = modlePath;
    d.labelPath = lablePath;
    d.labels = loadLabels(lablePath);
    d.model = SavedModelBundle.load(modlePath, "serve");
    d.inputSize = inputSize;
    return d;
  }


    private static String[] loadLabels(String filename) throws Exception {
        FileReader fileReader = new FileReader(filename);
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        List<String> lines = new ArrayList<String>();
        String line = null;
        while ((line = bufferedReader.readLine()) != null) {
            lines.add(line);
        }
        bufferedReader.close();
        return lines.toArray(new String[lines.size()]);
    }

    @Override
    public List<Recognition> recognizeImage(BufferedImage image) {
        if (image.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            System.out.printf("Expected 3-byte BGR encoding in BufferedImage, found %d. This code could be made more robust", image.getType());
            return new ArrayList<Recognition>();
        }
        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;
        long[] shape = new long[] {BATCH_SIZE, image.getHeight(), image.getWidth(), CHANNELS};
        byte[] data = ((DataBufferByte) image.getData().getDataBuffer()).getData();
        ImageUtil.BGR2RGB(data);
        List<Tensor<?>> outputs = null;
        try (Tensor<UInt8> input = Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data))) {
            outputs =
                    model
                            .session()
                            .runner()
                            .feed("image_tensor", input)
                            .fetch("detection_scores")
                            .fetch("detection_classes")
                            .fetch("detection_boxes")
                            .run();
        }
        try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
             Tensor<Float> classesT = outputs.get(1).expect(Float.class);
             Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
            // All these tensors have:
            // - 1 as the first dimension
            // - maxObjects as the second dimension
            // While boxesT will have 4 as the third dimension (2 sets of (x, y) coordinates).
            // This can be verified by looking at scoresT.shape() etc.
            int maxObjects = (int) scoresT.shape()[1];
            float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
            float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
            float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
            // Print all objects whose score is at least 0.5.
            final PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            1,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });
            for (int i = 0; i < scores.length; ++i) {
                Recognition r = new Recognition(i, labels[(int) classes[i]], scores[i],
                        new BoxPosition(boxes[i][1] * inputSize,
                                boxes[i][0] * inputSize,
                                (boxes[i][3] - boxes[i][1]) * inputSize,
                                (boxes[i][2] - boxes[i][0]) * inputSize),
                        inputSize);
                pq.add(r);
            }
            final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
            for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
                recognitions.add(pq.poll());
            }
            return recognitions;
        }
    }
}
