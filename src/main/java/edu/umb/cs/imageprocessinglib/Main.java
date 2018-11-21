package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.feature.FeatureStorage;
import edu.umb.cs.imageprocessinglib.model.DescriptorType;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import edu.umb.cs.imageprocessinglib.util.StorageUtil;
import org.opencv.core.*;
import org.opencv.features2d.Features2d;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        testOpenCV();
//        testTensorFlow();
        testRobustFeature();
//        testTFRobustFeature();
    }

    private static void testRobustFeature() throws IOException {
        String image_1 = "src/main/resources/image/Vegeta_1.png";
        String image_2 = "src/main/resources/image/Vegeta_2.png";
        Mat img = ImageUtil.loadMatImage(image_1);
        Mat testImg = ImageUtil.loadMatImage(image_2);
        testImg = ImageUtil.scaleImage(testImg, 0.5f);
        ImageFeature templateF = ImageProcessor.extractRobustFeatures(img, 100, DescriptorType.ORB);
//        ImageFeature templateF = ImageProcessor.extractORBFeatures(img, 100);
        ImageFeature testF = ImageProcessor.extractORBFeatures(testImg);
        System.out.printf("Comparing %d vs %d FPs ", templateF.getSize(), testF.getSize());
        MatOfDMatch matches = ImageProcessor.matchImages(testF, templateF);
//        MatOfDMatch matches = ImageProcessor.matchImages(templateF, testF);


//        SimpleRegression rx=new SimpleRegression();
//        SimpleRegression ry=new SimpleRegression();
//
//        DMatch[] dMatches=matches.toArray();
//        for(int i=0;i<dMatches.length;i++){
//            DMatch tmpd=dMatches[i];
//            KeyPoint kp1=ImageProcessor.findKeyPoint(templateF, tmpd.queryIdx);
//            KeyPoint kp2=ImageProcessor.findKeyPoint(testF, tmpd.trainIdx);
//
//            System.out.printf("x:%.02f, y:%.02f \t x:%.02f, y:%.02f \t dist:%.02f\n",kp1.pt.x, kp1.pt.y, kp2.pt.x, kp2.pt.y, tmpd.distance);
//            rx.addData(kp1.pt.x, kp2.pt.x);
//            ry.addData(kp1.pt.y, kp2.pt.y);
//        }
//        System.out.println();
//
//        MatOfDMatch mymatches = ImageProcessor.myMatchImages(templateF, testF, rx, ry);
        MatOfDMatch mymatches = ImageProcessor.matchWithRegression(testF, templateF);



//        Features2d.drawKeypoints(img, templateF.getObjectKeypoints(), img, Scalar.all(-1), Features2d.DRAW_RICH_KEYPOINTS);
//        Features2d.drawKeypoints(testImg, testF.getObjectKeypoints(), testImg, Scalar.all(-1), Features2d.DRAW_RICH_KEYPOINTS);
        System.out.printf("Match number: %d, Precision: %f\n", matches.total(), (float)matches.total()/ templateF.getSize());
        //display matches
        Mat display = new Mat();
        Features2d.drawMatches(testImg, testF.getObjectKeypoints(), img, templateF.getObjectKeypoints(), matches, display);
        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(img));
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(testImg));


        System.out.printf("Match number: %d, Precision: %f\n", mymatches.total(), (float)mymatches.total()/ templateF.getSize());
        //display matches
        Mat display1 = new Mat();
        Features2d.drawMatches(testImg, testF.getObjectKeypoints(),img, templateF.getObjectKeypoints(),  mymatches, display1);
        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display1));
    }



    private static void testTFRobustFeature() throws IOException {
        //match TensorFlow cropped image
        String image_2 = "src/main/resources/image/test.jpg";
        String image_1 = "src/main/resources/image/test_10.jpg";
        ObjectDetector objectDetector = new ObjectDetector();
        objectDetector.init();
        List<Recognition> r1 = objectDetector.recognizeImage(image_1);
        List<Recognition> r2 = objectDetector.recognizeImage(image_2);
        for (Recognition r : r1) {
            for (Recognition rt : r2) {
//                if (r.getTitle().equals(rt.getTitle())) {
                    Mat img = r.loadPixels();
                    Mat testImg = rt.loadPixels();
                    ImageFeature templateF = ImageProcessor.extractORBFeatures(img, 100);
//                    ImageFeature templateF = ImageProcessor.extractRobustFeatures(img, 100, DescriptorType.ORB);
                    ImageFeature testF = ImageProcessor.extractORBFeatures(testImg);
                    System.out.printf("Comparing %d vs %d FPs ", testF.getSize(), templateF.getSize());
//                    MatOfDMatch matches = ImageProcessor.BFMatchImages(templateF, testF);
                    MatOfDMatch matches = ImageProcessor.matchWithRegression(testF, templateF);

//                    Features2d.drawKeypoints(img, templateF.getObjectKeypoints(), img, Scalar.all(-1), Features2d.DRAW_RICH_KEYPOINTS);
//        Features2d.drawKeypoints(testImg, testF.getObjectKeypoints(), testImg, Scalar.all(-1), Features2d.DRAW_RICH_KEYPOINTS);
                    System.out.printf("Match number: %d, Precision: %f\n", matches.total(), (float)matches.total()/ templateF.getSize());
                    //display matches
                    Mat display = new Mat();
                    Features2d.drawMatches(testImg, testF.getObjectKeypoints(), img, templateF.getObjectKeypoints(), matches, display);
                    ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
//                }
            }
        }
    }


    private static void testTensorFlow() throws IOException {
//      String IMAGE = "/image/cow-and-bird.jpg";
        //String IMAGE = "/image/eagle.jpg";
        String imgPath = "src/main/resources/image/test_10.jpg";
        BufferedImage img = ImageUtil.loadImage(imgPath);
        ObjectDetector objectDetector = new ObjectDetector();
        objectDetector.init();
        List<Recognition> recognitions = objectDetector.recognizeImage(img);
        int i = 0;
        for (Recognition recognition : recognitions) {
            System.out.printf("Object: %s - confidence: %f box: %s\n",
                    recognition.getTitle(), recognition.getConfidence(), recognition.getLocation());

            StorageUtil.saveRecognitionToFile(recognition,"test" + (++i));
//            Recognition temp=StorageUtil.readRecognitionFromFile("test");
//            ImageUtil.displayImage(temp.loadPixels());
//            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(temp.getPixels()));
            //ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(recognition.getPixels()));
            Recognition temp = StorageUtil.readRecognitionFromFile("test" + i);
            Mat croppedImg = temp.loadPixels();
            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(croppedImg));
            ImageFeature imageFeature = temp.loadFeature();
            MatOfDMatch m = ImageProcessor.matchImages(imageFeature, imageFeature);
            Mat display = new Mat();
            Features2d.drawMatches(croppedImg, imageFeature.getObjectKeypoints(), croppedImg, imageFeature.getObjectKeypoints(), m, display);
            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
        }
//        Recognition temp=StorageUtil.readRecognitionFromFile("test");
//        Mat croppedImg = temp.loadPixels();
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(croppedImg));
//        ImageFeature imageFeature = temp.loadFeature();
//        MatOfDMatch m = ImageProcessor.matchImages(imageFeature, imageFeature);
//        Mat display = new Mat();
//        Features2d.drawMatches(croppedImg, imageFeature.getObjectKeypoints(), croppedImg, imageFeature.getObjectKeypoints(), m, display);
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
    }

    private static void testOpenCV() throws IOException {
        //Reading the Image from the file
        // System.out.println("Working Directory = " + System.getProperty("user.dir"));
        //String file ="src/main/resources/image/eagle.jpg";
        //BufferedImage image = ImageIO.read(new File(file));
        //Mat img = ImageUtil.BufferedImage2Mat(image);

        File folder = new File(".");
        File[] listOfFiles = folder.listFiles();

        String prefix = StorageUtil.RECOGNITION_TAG;
        String suffix = ".png";

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                String filename = listOfFiles[i].getName();
//                if (filename.contains(prefix)) {
                if (filename.contains(suffix)) {
                    System.out.println(filename);
                    testFP(filename);
                    break;
                }
            }
        }

    }

    private static void testFP(String filename){
        Mat img = StorageUtil.readMatFromFile(filename);

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
        //MatOfDMatch matches = ImageProcessor.matchImages(imageFeature, new ImageFeature(kps, des));

        System.out.printf("Comparing %d vs %d FPs ", imageFeature.getSize(),imageFeature.getSize());
        long t_before=System.currentTimeMillis();
        MatOfDMatch matches = ImageProcessor.matchImages(imageFeature, imageFeature1);
        long t_after=System.currentTimeMillis();
        System.out.printf("takes %.02f s\n", ((float)(t_after-t_before)/1000));

        System.out.printf("Precision: %f\n", (float)matches.total()/ imageFeature.getDescriptors().rows());

        MatOfDMatch m = new MatOfDMatch();
        m.fromList(matches.toList().subList(0,50));
        //display matches
        Mat display = new Mat();
        Features2d.drawMatches(img, kps, img, kps, m, display);
        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));

        //test the influence of feature points number on matching time
//        for (int i = 100; i <= 500; i += 50) {
//            ImageFeature feature = ImageProcessor.extractORBFeatures(img, i);
//            System.out.printf("Comparing %d vs %d FPs ", feature.getSize(),feature.getSize());
//            long before=System.currentTimeMillis();
//            MatOfDMatch ms = ImageProcessor.matchImages(feature, feature);
//            long after=System.currentTimeMillis();
//            System.out.printf("takes %.03f s\n", ((float)(after-before)/1000));
//        }
        //display lighted image
        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(ImageUtil.lightImage(img, 1.8f, 20)));
    }
}
