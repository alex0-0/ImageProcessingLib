package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.feature.FeatureStorage;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.opencv.core.*;
import org.opencv.features2d.Features2d;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        testOpenCV();
        testTensorFlow();
    }

    private static void testTensorFlow() throws IOException {
//      String IMAGE = "/image/cow-and-bird.jpg";
        //String IMAGE = "/image/eagle.jpg";
        String IMAGE = "/image/test_2.jpg";
        ObjectDetector objectDetector = new ObjectDetector();
        objectDetector.init();
        List<Recognition> recognitions = objectDetector.recognizeImage(IMAGE);
        for (Recognition recognition : recognitions) {
            System.out.printf("Object: %s - confidence: %f box: %s\n",
                    recognition.getTitle(), recognition.getConfidence(), recognition.getLocation());
            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(recognition.getPixels()));
        }
    }

    private static void testOpenCV() throws IOException {
        //Reading the Image from the file
        System.out.println("Working Directory = " + System.getProperty("user.dir"));
        String file ="src/main/resources/image/eagle.jpg";
        BufferedImage image = ImageIO.read(new File(file));
        Mat img = ImageUtil.BufferedImage2Mat(image);
//        Mat img = ImageProcessor.loadImage(file); //imageCodecs.imread(file);
//        ImageFeature imageFeature = ImageProcessor.extractDistinctFeatures(img);
        ImageFeature imageFeature = ImageProcessor.extractFeatures(img);

//        Features2d.drawKeypoints(img, imageFeature.getObjectKeypoints(), img, Scalar.all(-1), Features2d.DRAW_RICH_KEYPOINTS);
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(img));

        FeatureStorage storage = new FeatureStorage();

        storage.saveFPtoFile("des.xml", imageFeature);
/*
        storage.open("des.xml", FeatureStorage.FeatureStorageFlag.WRITE);
        storage.writeMat("des", imageFeature.getDescriptors());
        storage.writeKeyPoints("keypoint", imageFeature.getObjectKeypoints());
        storage.release();
*/
        //test if the KPs retrieved from xml file is identical to original KPs

        ImageFeature imageFeature1=storage.loadFPfromFile("des.xml");

        /*
        storage.open("des.xml");
        Mat des = storage.readMat("des");
        MatOfKeyPoint kps = storage.readKeyPoints("keypoint");
        */

        Mat des=imageFeature1.getDescriptors();
        MatOfKeyPoint kps=imageFeature1.getObjectKeypoints();
        //MatOfDMatch matches = ImageProcessor.matcheImages(imageFeature, new ImageFeature(kps, des));
        MatOfDMatch matches = ImageProcessor.matcheImages(imageFeature, imageFeature1);
        System.out.printf("Precision: %f", (float)matches.total()/ imageFeature.getDescriptors().rows());

        MatOfDMatch m = new MatOfDMatch();
        m.fromList(matches.toList().subList(0,50));
        //display matches
        Mat display = new Mat();
        Features2d.drawMatches(img, kps, img, kps, m, display);
        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
    }
}
