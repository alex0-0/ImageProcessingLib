package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.tensorflow.Classifier;
import edu.umb.cs.imageprocessinglib.tensorflow.TensorFlowObjectDetectionAPIModel;
import edu.umb.cs.imageprocessinglib.tensorflow.TensorFlowYoloDetector;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class ObjectDetector {
    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
    // or YOLO.
    public enum DetectorMode {
        TF_OD_API, YOLO;
    }

    private static final String YOLO_MODEL_FILE = "/YOLO/tiny-yolo-voc.pb";
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;

    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final String TF_OD_API_MODEL_FILE = "src/main/resources/ssd_inception_v2_coco/";
    private static final String TF_OD_API_LABELS_FILE = "src/main/resources/ssd_inception_v2_coco/coco_label_list.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private Classifier detector;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    //    private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
    private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;

    private float minConfidence = 0.5f;
    private int cropSize;


    public void init() {
        if (MODE == DetectorMode.YOLO) {
            detector = TensorFlowYoloDetector.create(
                    YOLO_MODEL_FILE,
                    YOLO_INPUT_SIZE,
                    YOLO_INPUT_NAME,
                    YOLO_OUTPUT_NAMES,
                    YOLO_BLOCK_SIZE);
            minConfidence = MINIMUM_CONFIDENCE_YOLO;
            cropSize = YOLO_INPUT_SIZE;
        } else {
            try {
                detector = TensorFlowObjectDetectionAPIModel.create(TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
                cropSize = TF_OD_API_INPUT_SIZE;
                minConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            } catch (final Exception e) {
                e.printStackTrace();
            }
        }
    }

    public List<Recognition> recognizeImage(String imagePath) throws IOException {
        BufferedImage bImg = ImageIO.read(getClass().getResource(imagePath));
        List<Recognition> recognitions = recognizeImage(bImg);

//        System.out.println(recognitions.size());


//        String fileName = imagePath.substring(imagePath.lastIndexOf("/") + 1, imagePath.length());
//        ImageUtil.labelAndSaveImage(image, recognitions, fileName, YOLO_INPUT_SIZE);
//        ImageUtil.displayImage(bImg);

/*
        for (Recognition recognition : recognitions) {
            recognition.loadPiexels(bimg, YOLO_INPUT_SIZE);
        }
*/
        return recognitions;
    }
    public List<Recognition> recognizeImage(BufferedImage image) {

        List<Recognition> recognitions = detector.recognizeImage(image);
        recognitions = recognitions.
                stream().
                filter(r -> r.getConfidence() >= minConfidence).
                map(r -> {
                    r.loadPiexels(ImageUtil.BufferedImage2Mat(image), cropSize);
//                    ImageUtil.displayImage(r.getPixels());
                    return r;
                }).collect(Collectors.toList());

        return recognitions;
    }

}
