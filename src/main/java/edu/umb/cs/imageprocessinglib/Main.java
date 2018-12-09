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
import java.util.Comparator;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        testOpenCV();
//        testTensorFlow();
//        testRobustFeature();
//        testTFRobustFeature();
        testDistortion();
    }

    private static void testDistortion() throws IOException {
        String image_1 = "src/main/resources/image/Vegeta_1.png";
        Mat img = ImageUtil.loadMatImage(image_1);
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(img));
//        List<Mat> distortedImg = ImageProcessor.rotatedImage(img, 5f, 5);
//        List<Mat> distortedImg = ImageProcessor.scaleImage(img, -0.1f, 5);
//        List<Mat> distortedImg = ImageProcessor.lightImage(img, -0.1f, 5);
        List<Mat> distortedImg = ImageProcessor.changeToLeftPerspective(img, 10f, 5);
//        List<Mat> distortedImg = ImageProcessor.changeToRightPerspective(img, 10f, 5);
//        List<Mat> distortedImg = ImageProcessor.changeToBottomPerspective(img, 10f, 5);
//        List<Mat> distortedImg = ImageProcessor.changeToTopPerspective(img, 10f, 5);
        for (Mat i : distortedImg) {
//            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(i));
        }

        //test robust feature
//        ImageFeature tIF = ImageProcessor.extractRobustFeatures(img, distortedImg, 100, DescriptorType.SURF);
        ImageFeature tIF = ImageProcessor.extractRobustFeatures(img, distortedImg, 100, DescriptorType.ORB);
//        String image_2 = "src/main/resources/image/test.jpg";
        String image_2 = "src/main/resources/image/Vegeta_2.png";
        Mat testImg = ImageUtil.loadMatImage(image_2);
        ImageFeature testF = ImageProcessor.extractORBFeatures(testImg);
//        ImageFeature testF = ImageProcessor.extractSURFFeatures(testImg);
        System.out.printf("Comparing %d vs %d FPs\n", tIF.getSize(), testF.getSize());
//        MatOfDMatch mymatches = ImageProcessor.matchImages(testF, tIF);
        MatOfDMatch mymatches = ImageProcessor.matchWithRegression(testF, tIF);
//        MatOfDMatch mymatches = ImageProcessor.matchWithRegression(testF, tIF, 5, 500, 15);
        Mat display1 = new Mat();
        Features2d.drawMatches(testImg, testF.getObjectKeypoints(),img, tIF.getObjectKeypoints(),  mymatches, display1);
        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display1));
        System.out.printf("Distortion match number: %d, Precision: %f\n", mymatches.total(), (float)mymatches.total()/ tIF.getSize());


        ImageFeature tIF2 = ImageProcessor.extractORBFeatures(img, 100);
//        ImageFeature tIF2 = ImageProcessor.extractSURFFeatures(img);
        Mat display2 = new Mat();
//        MatOfDMatch mymatches2 = ImageProcessor.matchImages(testF, tIF2);
        MatOfDMatch mymatches2 = ImageProcessor.matchWithRegression(testF, tIF2);
//        MatOfDMatch mymatches2 = ImageProcessor.matchWithRegression(testF, tIF2, 5, 500, 15);
        Features2d.drawMatches(testImg, testF.getObjectKeypoints(),img, tIF2.getObjectKeypoints(),  mymatches2, display2);
        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display2));
        System.out.printf("Regular match number: %d, Precision: %f\n", mymatches2.total(), (float)mymatches2.total()/ tIF2.getSize());

        List<KeyPoint> tKP = tIF2.getObjectKeypoints().toList();
        List<KeyPoint> qKP = testF.getObjectKeypoints().toList();
        List<KeyPoint> dtKP = tIF.getObjectKeypoints().toList();
        Comparator<DMatch> comparator = new Comparator<DMatch>() {
            @Override
            public int compare(DMatch o1, DMatch o2) {
                if ( tKP.get(o1.trainIdx).pt.x > tKP.get(o2.trainIdx).pt.x)
                    return 1;
                if ( tKP.get(o1.trainIdx).pt.x < tKP.get(o2.trainIdx).pt.x)
                    return -1;
                return 0;
            }
        };
        List<DMatch> matches1 = mymatches.toList();
        matches1.sort((o1, o2) -> {
            return (int)(dtKP.get(o1.trainIdx).pt.x - dtKP.get(o2.trainIdx).pt.x);
        });
        List<DMatch> matches2 = mymatches2.toList();
        matches2.sort((o1, o2) -> {
            return (int)(tKP.get(o1.trainIdx).pt.x - tKP.get(o2.trainIdx).pt.x);
        });
        for (int i = 0; i < Math.max(matches1.size(), matches2.size()); i++) {
            if (matches2.size() > i) {
                DMatch match = matches2.get(i);
                System.out.printf("t: (%.2f, %.2f), q: (%.2f, %.2f)",
                        tKP.get(match.trainIdx).pt.x, tKP.get(match.trainIdx).pt.y,
                        qKP.get(match.queryIdx).pt.x, qKP.get(match.queryIdx).pt.y
                );
            }
            if (matches1.size() > i) {
                DMatch match = matches1.get(i);
                System.out.printf("\t|\tt: (%.2f, %.2f), q: (%.2f, %.2f)",
                        dtKP.get(match.trainIdx).pt.x, dtKP.get(match.trainIdx).pt.y,
                        qKP.get(match.queryIdx).pt.x, qKP.get(match.queryIdx).pt.y
                );
            }
            System.out.printf("\n");
        }
//        for (DMatch match : matches1) {
//            System.out.printf("template pos: (%.2f, %.2f), query pos: (%.2f, %.2f)\n",
//                    tKP.get(match.trainIdx).pt.x, dtKP.get(match.trainIdx).pt.y,
//                    qKP.get(match.queryIdx).pt.x, qKP.get(match.queryIdx).pt.y
//            );
//        }
        dtKP.sort((kp1, kp2) -> {
            return (int)(kp1.pt.x - kp2.pt.x);
        });
        for (KeyPoint kp : dtKP) {
            System.out.printf("keypoint pos: (%.2f, %.2f)\n", kp.pt.x, kp.pt.y);
        }
        System.out.printf("\n*************************\n");
//        List<DMatch> matches2 = mymatches2.toList();
//        matches2.sort(comparator);
//        for (DMatch match : matches2) {
//            System.out.printf("template pos: (%.2f, %.2f), query pos: (%.2f, %.2f)\n",
//                    tKP.get(match.trainIdx).pt.x, tKP.get(match.trainIdx).pt.y,
//                    qKP.get(match.queryIdx).pt.x, qKP.get(match.queryIdx).pt.y
//                    );
//        }

        System.out.printf("dis_thd\trobust\tregular\n");
        for (float i=200f; i<800; i+=50) {
//        for (float i=0.5f; i<4; i+=0.2) {
            MatOfDMatch robustMatch = ImageProcessor.matchWithRegression(testF, tIF, 5, i, 100);
            MatOfDMatch regularMatch = ImageProcessor.matchWithRegression(testF, tIF2, 5, i, 100);
//            MatOfDMatch robustMatch = ImageProcessor.matchImages(testF, tIF);
//            MatOfDMatch regularMatch = ImageProcessor.matchImages(testF, tIF2);
//            System.out.printf("%.2f \t %.2f(%d) \t %.2f(%d)\n",
//                    i,
//                    (float)robustMatch.total()/ tIF.getSize(),
//                    robustMatch.total(),
//                    (float)regularMatch.total()/tIF2.getSize(),
//                    regularMatch.total());
        }

//        Mat tMat = new Mat();//img.clone();
//        Features2d.drawKeypoints(img, tIF.getObjectKeypoints(), tMat, Scalar.all(-1), Features2d.DRAW_RICH_KEYPOINTS);
//        ImageFeature imageFeature = ImageProcessor.extractORBFeatures(img, 100);
//        Mat tMat_1 = new Mat();//img.clone();
//        Features2d.drawKeypoints(img, imageFeature.getObjectKeypoints(), tMat_1, Scalar.all(-1), Features2d.DRAW_RICH_KEYPOINTS);
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(tMat));
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(tMat_1));
//        MatOfDMatch mymatches3 = ImageProcessor.matchImages(tIF, imageFeature);
//        Mat display3 = new Mat();
//        Features2d.drawMatches(tMat, tIF.getObjectKeypoints(),tMat_1, imageFeature.getObjectKeypoints(),  mymatches3, display3);
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display3));
//        System.out.printf("Robust-Regular match number: %d, Precision: %f\n", mymatches3.total(), (float)mymatches3.total()/ tIF.getSize());
    }

    private static void testRobustFeature() throws IOException {
        String image_1 = "src/main/resources/image/Vegeta_1.png";
        String image_2 = "src/main/resources/image/Vegeta_2.png";
        Mat img = ImageUtil.loadMatImage(image_1);
        Mat testImg = ImageUtil.loadMatImage(image_2);
//        testImg = ImageUtil.scaleImage(testImg, 0.8f);
//        testImg = ImageUtil.scaleImage(testImg, 1.0f);
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
//                    ImageFeature templateF = ImageProcessor.extractORBFeatures(img, 100);
                    ImageFeature templateF = ImageProcessor.extractRobustFeatures(img, 100, DescriptorType.ORB);
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
//        ImageFeature imageFeature = ImageProcessor.extractRobustFeatures(img);
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
