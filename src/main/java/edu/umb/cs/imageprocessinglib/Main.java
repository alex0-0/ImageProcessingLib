package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.feature.FeatureStorage;
import edu.umb.cs.imageprocessinglib.model.DescriptorType;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import edu.umb.cs.imageprocessinglib.util.StorageUtil;
import org.opencv.core.*;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.FlannBasedMatcher;
import org.opencv.features2d.ORB;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.opencv.core.CvType.CV_32F;

public class Main {

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        testOpenCV();
//        testTensorFlow();
//        testRobustFeature("src/main/resources/image/motorcycle1/", "005.JPG");
//        testRobustFeature("src/main/resources/image/toy_car/", "000.png");
//        testRobustFeature("src/main/resources/image/horse1/", "000.JPG");
//        testRobustFeature("src/main/resources/image/furry_dog/", "0.png");
//        testRobustFeature("src/main/resources/image/horse2/", "-05.JPG");
//        testRobustFeature("src/main/resources/image/girl_statue/", "5.png");
//        testRobustFeature("src/main/resources/image/van_gogh/", "5.png");
//        testRobustFeature("src/main/resources/image/toy_bear/", "0.png");
//        testTFRobustFeature();
//        testDistortion();

        String[] dirNames = {"lego_man", "shoe", "furry_elephant", "toy_bear", "van_gogh", "furry_bear", "duck_cup"};
        for (String dir : dirNames) {
            for (int i=0; i <= 60; i+=10) {
                testRobustFeature("src/main/resources/image/"+dir+"/", i, dir+"_left_robust_fp_test", false, DistortionType.LeftPers,
                        5, 10, 100, 300, 500, 500, 20, 8);
                int k = 350-i;
                testRobustFeature("src/main/resources/image/"+dir+"/", k, dir+"_right_robust_fp_test", false, DistortionType.RightPers,
                        5, 10, 100, 300, 500, 500, 20, 8);
            }
        }
//        testRobustFeature("src/main/resources/image/van_gogh/", 0, null, true,  DistortionType.LeftPers,
//                5, 10, 100, 300, 500, 500, 20, 8);
//        testRobustFeature("src/main/resources/image/furry_bear/", 50, null, true, DistortionType.RightPers,
//                5, 10, 100, 300, 500, 300, 20, 8);
    }

    private static void testDistortion() throws IOException {
        String image_1 = "src/main/resources/image/Vegeta_10.png";
        Mat img = ImageUtil.loadMatImage(image_1);
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(img));
//        List<Mat> distortedImg = ImageProcessor.rotatedImage(img, 5f, 5);
//        List<Mat> distortedImg = ImageProcessor.scaleImage(img, -0.1f, 5);
//        List<Mat> distortedImg = ImageProcessor.lightImage(img, -0.1f, 5);
        List<Mat> distortedImg = ImageProcessor.changeToLeftPerspective(img, 10f, 10);
//        List<Mat> distortedImg = ImageProcessor.changeToRightPerspective(img, 10f, 5);
//        List<Mat> distortedImg = ImageProcessor.changeToBottomPerspective(img, 10f, 5);
//        List<Mat> distortedImg = ImageProcessor.changeToTopPerspective(img, 10f, 5);
        for (Mat i : distortedImg) {
//            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(i));
            //match distorted images with original image
            ImageFeature qIF = ImageProcessor.extractFeatures(i);
            ImageFeature tIF = ImageProcessor.extractFeatures(img);
            Mat descriptors_1 = qIF.getDescriptors();
            Mat descriptors_2 = tIF.getDescriptors();
//            if(descriptors_1.type()!=CV_32F) {
//                descriptors_1.convertTo(descriptors_1, CV_32F);
//            }
//
//            if(descriptors_2.type()!=CV_32F) {
//                descriptors_2.convertTo(descriptors_2, CV_32F);
//            }
//            FlannBasedMatcher matcher = FlannBasedMatcher.create();
//            MatOfDMatch mymatches = new MatOfDMatch();
//            matcher.match(qIF.getDescriptors(), tIF.getDescriptors(), mymatches);
            MatOfDMatch mymatches = ImageProcessor.BFMatchImages(qIF, tIF);
            List<DMatch> m = new ArrayList<>();

            for (DMatch match : mymatches.toList()) {
                if (match.distance < 200)
                    m.add(match);
            }
            mymatches = new MatOfDMatch();
            mymatches.fromList(m);
            Mat display1 = new Mat();
            Features2d.drawMatches(i, qIF.getObjectKeypoints(),img, tIF.getObjectKeypoints(),  mymatches, display1);
            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display1));
            System.out.printf("Distortion match number: %d, Precision: %f\n", mymatches.total(), (float)mymatches.total()/ tIF.getSize());
        }

        //test robust feature
//        ImageFeature tIF = ImageProcessor.extractRobustFeatures(img, distortedImg, 100, DescriptorType.SURF);
        ImageFeature tIF = ImageProcessor.extractRobustFeatures(img, distortedImg, 100, 200, DescriptorType.ORB);
//        String image_2 = "src/main/resources/image/test.jpg";
        String image_2 = "src/main/resources/image/Vegeta_00.png";
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
                System.out.printf("t: (%.2f, %.2f), q: (%.2f, %.2f), dis: %.2f",
                        tKP.get(match.trainIdx).pt.x, tKP.get(match.trainIdx).pt.y,
                        qKP.get(match.queryIdx).pt.x, qKP.get(match.queryIdx).pt.y,
                        match.distance
                );
            }
            if (matches1.size() > i) {
                DMatch match = matches1.get(i);
                System.out.printf("\t|\tt: (%.2f, %.2f), q: (%.2f, %.2f), dis: %.2f",
                        dtKP.get(match.trainIdx).pt.x, dtKP.get(match.trainIdx).pt.y,
                        qKP.get(match.queryIdx).pt.x, qKP.get(match.queryIdx).pt.y,
                        match.distance
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
//            MatOfDMatch robustMatch = ImageProcessor.matchWithRegression(testF, tIF, 5, i, 100);
//            MatOfDMatch regularMatch = ImageProcessor.matchWithRegression(testF, tIF2, 5, i, 100);
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

    enum DistortionType {
        LeftPers,
        RightPers,
        BottomPers,
        TopPers,
        Scale,
        Light,
        Rotation
    }

    /**
     *
     * @param filePath          the path of directory in where the images are
     * @param templateValue     the value used as template
     * @param logName           file used to log. If logName is not given, directory_name+parameters will be used
     * @param rewriteHP           whether the method should write down hyperparameters again if the file already exist
     * @param dType             what kind of distortion is used to generate distorted images
     * @param dStep             step value for distortion
     * @param dNum              how many distorted images should be generated
     * @param tFPNum            the number of template feature points
     * @param robustDisThd      matching distance threshold used in extracting robust feature point
     * @param qFPNum            the number of feature points in query images
     * @param matchDisThd       matching distance threshold used in matching query images and template image
     * @param matchPosThd       position distance threshold used in matchWithRegression
     * @param testNum           how many tests should be done
     * @throws IOException
     * TODO: need to think about what if templateAngle is required to be a float value
     */
    static void testRobustFeature(String filePath,
                                  int templateValue,
                                  String logName,
                                  boolean rewriteHP,
                                  DistortionType dType,
                                  float dStep,
                                  int dNum,
                                  int tFPNum,
                                  int robustDisThd,
                                  int qFPNum,
                                  int matchDisThd,
                                  int matchPosThd,
                                  int testNum
                                  ) throws IOException {
        String hyperParams = "*********hyperparameters**********\n" +
                "ratioTest: knnRatioThreshold 0.7; " +
                "ransac: reproj_thd 15, max_itd 2000, conf 0.995; " +
                "matchWithRegression: posThd min(posThd,avg_pos_dif*1.5f)), ransac_condition symMatches.length>20, intern-quartile 25%~75%, " +
                "intern-quartile init min dis thd 5, init diff ratio 2";
        String tFName = filePath + templateValue + ".png";
        Mat tImg = ImageUtil.loadMatImage(tFName);
        List<Mat> distortedImg = null;
        float testStep = 0;
        String distortionStr = "";  //used in log file name
        switch (dType) {
            case LeftPers:
                distortedImg = ImageProcessor.changeToLeftPerspective(tImg, dStep, dNum);
                testStep = 5;
                distortionStr = "lp";
                break;
            case RightPers:
                distortedImg = ImageProcessor.changeToRightPerspective(tImg, dStep, dNum);
                testStep = -5;
                distortionStr = "rp";
                break;
            case TopPers:
                distortedImg = ImageProcessor.changeToTopPerspective(tImg, dStep, dNum);
                distortionStr = "tp";
                break;
            case BottomPers:
                distortedImg = ImageProcessor.changeToBottomPerspective(tImg, dStep, dNum);
                distortionStr = "bp";
                break;
            case Scale:
                distortedImg = ImageProcessor.scaleImage(tImg, dStep, dNum);
                distortionStr = "s";
                break;
            case Light:
                distortedImg = ImageProcessor.lightImage(tImg, dStep, dNum);
                distortionStr = "l";
                break;
            case Rotation:
                distortedImg = ImageProcessor.rotatedImage(tImg, dStep, dNum);
                distortionStr = "r";
                break;
            default:
                break;
        }

        File dir = new File(filePath);
        //prepare for writing to log file
        if (logName == null) {
            logName = dir.getName() + "_" + distortionStr + templateValue + "_ds" + dStep + "_dn" + dNum + "_tfpn" + tFPNum + "_rdt" + robustDisThd
            + "_qfpn" + qFPNum + "_mdt" + matchDisThd + "_mpt" + matchPosThd + "_tn" + testNum;
        }
        File logFile = new File(logName);
        boolean append = logFile.exists();
        logFile.createNewFile();
        PrintWriter pw = new PrintWriter(new FileOutputStream(logFile, true));
        List<String> fNames = new ArrayList<>();
        List<Float> pres = new ArrayList<>();

        ImageFeature tIF = ImageProcessor.extractRobustFeatures(tImg, distortedImg, tFPNum, robustDisThd, DescriptorType.ORB, null);

        System.out.printf("-----%s-------\n", dir.getName());
        for (int k=1; k <= testNum; k++) {
            int i = templateValue + (int)testStep * k;
            Mat qImg = ImageUtil.loadMatImage(filePath+i+".png");
            //assume we use ORB feature points in default
            ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, qFPNum);
            MatOfDMatch matches = ImageProcessor.matchWithRegression(qIF, tIF, 5, matchDisThd, matchPosThd);
            float p = (float)matches.total() / tIF.getSize();
            System.out.printf("%d: %f\n", i, p);
//            pw.printf("%d: %f\n", i, (float)matches.total() / tIF.getSize());
            fNames.add(""+(int)testStep*k);
            pres.add(p);
            //display matches
//            Mat display = new Mat();
//            Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(),  matches, display);
//            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
        }

        //assume all false image are named as "f"+number+".png"
        for (int i=1; i<=Integer.MAX_VALUE ;i++) {
            File fImg = new File(filePath + "f" + i + ".png");
            if (!fImg.exists()) break;
            Mat qImg = ImageUtil.loadMatImage(fImg.getAbsolutePath());
            //assume we use ORB feature points in default
            ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, qFPNum);
            MatOfDMatch matches = ImageProcessor.matchWithRegression(qIF, tIF, 5, matchDisThd, matchPosThd);
            float p = (float)matches.total() / tIF.getSize();
            System.out.printf("f%d: %f\n", i, p);
//            pw.printf("f%d: %f\n", i, (float)matches.total() / tIF.getSize());
            fNames.add("f"+i);
            pres.add(p);
            //display matches
//            Mat display = new Mat();
//            Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(),  matches, display);
//            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
        }
        if (!append || rewriteHP) {
            pw.println(hyperParams);
            pw.printf("dType: %s dStep: %f, dNum: %d, tFPNum: %d, robustDisThd: %d, qFPNum: %d, matchDisThd: %d, matchPosThd: %d\n",
                    distortionStr, dStep, dNum, tFPNum, robustDisThd, qFPNum, matchDisThd, matchPosThd);

            pw.printf("-----%s-------\n", dir.getName());
            pw.print("init/change\t");
            for (String f : fNames)
                pw.print(f + "\t");
            pw.println();   //next line
        }
        pw.print(templateValue + "\t");
        for (float p : pres)
            pw.printf("%.2f\t",p);
        pw.println();   //next line
        pw.close();
    }

    private static void testRobustFeature(String filePath, String templateImg) throws IOException {
        Mat tImg = ImageUtil.loadMatImage(filePath+templateImg);
//        List<Mat> distortedImg = ImageProcessor.rotatedImage(img, 5f, 5);
//        List<Mat> distortedImg = ImageProcessor.scaleImage(img, -0.1f, 5);
//        List<Mat> distortedImg = ImageProcessor.lightImage(img, -0.1f, 5);
        List<Mat> distortedImg = ImageProcessor.changeToLeftPerspective(tImg, 5f, 10);
//        List<Mat> distortedImg = ImageProcessor.changeToRightPerspective(tImg, 5f, 10);
//        List<Mat> distortedImg = ImageProcessor.changeToBottomPerspective(img, 10f, 5);
//        List<Mat> distortedImg = ImageProcessor.changeToTopPerspective(img, 10f, 5);
        for (Mat i : distortedImg) {
//            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(i));
        }
        List<Integer> minTracker = new ArrayList<>();
        ImageFeature tIF = ImageProcessor.extractRobustFeatures(tImg, distortedImg, 100, 300, DescriptorType.ORB, minTracker);
        System.out.printf("number of template robust FP: %d\n", tIF.getSize());
        //calculate min precision
        List<Float> minRatioTracker = IntStream.range(0, minTracker.size()).mapToObj(i->{
            return (float)minTracker.get(i)/(i+1);
        }).collect(Collectors.toList());
        System.out.printf("min num:\t%s\nmin precision:\t%s\n", minTracker, minRatioTracker);

        //test on real images
        File dir = new File(filePath);
        File[] directoryListing = dir.listFiles();
        if (directoryListing != null) {
            List<File> files = new ArrayList<>(Arrays.asList(directoryListing));
            files.sort(Comparator.comparing(File::getName));
            for (File f : files) {
                if (f.getName().equals(templateImg))
                    continue;
                Mat qImg = ImageUtil.loadMatImage(f.getAbsolutePath());
                ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, 500);
                MatOfDMatch matches = ImageProcessor.matchWithRegression(qIF, tIF, 5, 500, 20);
                System.out.printf("%s: %f\n", f.getName(), (float)matches.total() / tIF.getSize());
                //display matches
//                Mat display = new Mat();
//                Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(),  matches, display);
//                ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
            }
        }
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
        String imgPath = "src/main/resources/image/test_20.jpg";
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
//            Recognition temp = StorageUtil.readRecognitionFromFile("test" + i);
//            Mat croppedImg = temp.loadPixels();
//            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(croppedImg));
//            ImageFeature imageFeature = temp.loadFeature();
//            MatOfDMatch m = ImageProcessor.matchImages(imageFeature, imageFeature);
//            Mat display = new Mat();
//            Features2d.drawMatches(croppedImg, imageFeature.getObjectKeypoints(), croppedImg, imageFeature.getObjectKeypoints(), m, display);
//            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
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
