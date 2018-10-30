package edu.umb.cs.imageprocessinglib;

//import android.content.res.AssetManager;
//import android.graphics.Bitmap;
//import android.graphics.Canvas;
//import android.graphics.Matrix;
//import android.graphics.RectF;
//
//import com.example.imageprocessinglib.ImageFeatures.FeatureDetector;
//import com.example.imageprocessinglib.ImageFeatures.FeatureMatcher;
//import com.example.imageprocessinglib.ImageProcessorConfig.*;
//import com.example.imageprocessinglib.tensorflow.Classifier;
//import com.example.imageprocessinglib.tensorflow.TensorFlowMultiBoxDetector;
//import com.example.imageprocessinglib.tensorflow.TensorFlowObjectDetectionAPIModel;
//import com.example.imageprocessinglib.tensorflow.TensorFlowYoloDetector;
//import com.example.imageprocessinglib.utils.ImageUtils;
//
//import org.opencv.android.Utils;
//import org.opencv.core.Mat;
//import org.opencv.core.MatOfDMatch;
//import org.opencv.core.MatOfKeyPoint;
//

import edu.umb.cs.imageprocessinglib.feature.FeatureDetector;
import edu.umb.cs.imageprocessinglib.feature.FeatureMatcher;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.tensorflow.Classifier;
import edu.umb.cs.imageprocessinglib.tensorflow.TensorFlowObjectDetectionAPIModel;
import edu.umb.cs.imageprocessinglib.tensorflow.TensorFlowYoloDetector;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class ImageProcessor {

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


    public void initObjectDetector() {
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

        System.out.println(recognitions.size());


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

//        for (Recognition recognition : recognitions) {
//            recognition.loadPiexels(image, cropSize);
//        }

//        filter out low confidence recognition
//        recognitions.removeIf(r -> r.getConfidence() < minConfidence);

        recognitions = recognitions.stream().filter(r -> r.getConfidence() >= minConfidence).map(r -> {
            r.loadPiexels(image, cropSize);
//            ImageUtil.displayImage(r.getPixels());
            return r;
        }).collect(Collectors.toList());

        return recognitions;
    }

    /*
    Extract image feature points
     */
    static public ImageFeature extractDistinctFeatures(Mat img) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractDistinctFeatures(img, kps, des);
        return new ImageFeature(kps, des);
    }

    /*
    Extract image feature points
     */
    static public ImageFeature extractFeatures(Mat img) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractORBFeatures(img, kps, des);
        return new ImageFeature(kps, des);
    }

//    static public ImageFeature extractFeatures(Bitmap bitmap) {
//        Mat img = new Mat();
//        Utils.bitmapToMat(bitmap, img);
//        return extractFeatures(img);
//    }

    /*
    Match two images
     */
    static public MatOfDMatch matcheImages(ImageFeature qIF, ImageFeature tIF) {
        return FeatureMatcher.getInstance().matchFeature(qIF.getDescriptors(), tIF.getDescriptors(), qIF.getObjectKeypoints(), tIF.getObjectKeypoints());
//        return FeatureMatcher.getInstance().BFMatchFeature(qIF.getDescriptors(), tIF.getDescriptors());
    }

    static public MatOfDMatch matcheImages(Mat queryImg, Mat temImg) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        ImageFeature qIF = extractFeatures(queryImg);
        ImageFeature tIF = extractFeatures(temImg);
        return matcheImages(qIF, tIF);
    }

//    static public MatOfDMatch matcheImages(Bitmap queryImg, Bitmap temImg) {
//        feature qIF = extractFeatures(queryImg);
//        feature tIF = extractFeatures(temImg);
//        return matcheImages(qIF, tIF);
//    }

    static public Mat loadImage(String file){
        Imgcodecs imageCodecs = new Imgcodecs();
        return imageCodecs.imread(file);
    }
}

