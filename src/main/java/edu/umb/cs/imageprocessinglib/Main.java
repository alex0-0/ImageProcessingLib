package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.feature.FeatureStorage;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.opencv.core.*;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        testOpenCV();
//        testTensorFlow();
    }

    private static void testTensorFlow() throws IOException {
//      String IMAGE = "/image/cow-and-bird.jpg";
        String IMAGE = "/image/eagle.jpg";
        ImageProcessor imageProcessor = new ImageProcessor();
        imageProcessor.initObjectDetector();
        List<Recognition> recognitions = imageProcessor.recognizeImage(IMAGE);
        for (Recognition recognition : recognitions) {
            System.out.printf("Object: %s - confidence: %f", recognition.getTitle(), recognition.getConfidence());
        }
    }

    private static void testOpenCV() {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
        System.out.println("mat = " + mat.dump());
        //Reading the Image from the file
        System.out.println("Working Directory = " +
                System.getProperty("user.dir"));
        Imgcodecs imageCodecs = new Imgcodecs();
//        String file ="/Users/alexli/IdeaProjects/ImageProcessingLib/src/main/resources/image/eagle.jpg";
        String file ="src/main/resources/image/eagle.jpg";
        Mat img = imageCodecs.imread(file);
//        ImageFeature imageFeature = ImageProcessor.extractDistinctFeatures(img);
        ImageFeature imageFeature = ImageProcessor.extractFeatures(img);

//        Features2d.drawKeypoints(img, imageFeature.getObjectKeypoints(), img, Scalar.all(-1), Features2d.DRAW_RICH_KEYPOINTS);
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(img));

        FeatureStorage storage = new FeatureStorage();
        storage.open("des.xml", FeatureStorage.FeatureStorageFlag.WRITE);
        storage.writeMat("des", imageFeature.getDescriptors());
//        storage.writeMat("kp", imageFeature.getObjectKeypoints());
        storage.release();

        storage.open("des.xml");
        Mat des = storage.readMat("des");
        MatOfDMatch matches = ImageProcessor.matcheImages(imageFeature, new ImageFeature(imageFeature.getObjectKeypoints(), des));
        System.out.printf("Precision: %f", (float)matches.total()/ imageFeature.getDescriptors().rows());


    }
}
