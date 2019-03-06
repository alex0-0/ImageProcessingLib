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
import java.io.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Main {
    static int DEBUG = 1;

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        testOpenCV();
//        testTensorFlow();
//        testRobustFeature("src/main/resources/image/single_distortion/motorcycle1/", "005.JPG");
//        testRobustFeature("src/main/resources/image/single_distortion/toy_car/", "000.png");
//        testRobustFeature("src/main/resources/image/single_distortion/horse1/", "000.JPG");
//        testRobustFeature("src/main/resources/image/single_distortion/furry_dog/", "0.png");
//        testRobustFeature("src/main/resources/image/single_distortion/horse2/", "-05.JPG");
//        testRobustFeature("src/main/resources/image/single_distortion/girl_statue/", "5.png");
//        testRobustFeature("src/main/resources/image/single_distortion/van_gogh/", "5.png");
//        testRobustFeature("src/main/resources/image/single_distortion/toy_bear/", "0.png");
//        testRobustFeature("src/main/resources/image/single_distortion/horse_scale/", "s1.00.JPG");
//        testTFRobustFeature();
//        testDistortion();

//        String[] dirNames = {"lego_man", "shoe", "furry_elephant", "toy_bear", "van_gogh", "furry_bear", "duck_cup"};
//        for (String dir : dirNames) {
//            for (int i=0; i <= 60; i+=10) {
//                testRobustFeature("src/main/resources/image/single_distortion/"+dir+"/", i, dir+"_left_robust_fp_test", false, new Distortion(DistortionType.LeftPers,
//                        5, 10, 100, 300), 500, 300, 20, 8);
//                int k = 350-i;
//                testRobustFeature("src/main/resources/image/single_distortion/"+dir+"/", k, dir+"_right_robust_fp_test", false, new Distortion(DistortionType.RightPers,
//                        5, 10, 100, 300), 500, 300, 20, 8);
//            }
//        }
//        testRobustFeature("src/main/resources/image/single_distortion/furry_elephant/", 350, null, true, new Distortion(DistortionType.RightPers,
//                5, 10, 100, 300), 500, 300, 20, 6);
//        testRobustFeature("src/main/resources/image/single_distortion/furry_bear/", 50, "ttt_log", true, new Distortion(DistortionType.LeftPers,
//                5, 10, 100, 300), 500, 300, 20, 6);
//        testRobustFeature("src/main/resources/image/single_distortion/shoe_scale/", 60, "ttt_log", true, new Distortion(DistortionType.ScaleUp,
//                0.05f, 10, 100, 300), 500, 300, 20, 8);
//        testRobustFeature("src/main/resources/image/single_distortion/shoe_scale/", 60, "ttt_log", true, new Distortion(DistortionType.ScaleUp,
//                0.05f, 10, 100, 300), 500, 300, 20, 8);
        testCombinedDistortion("src/main/resources/image/multi_distortion/detergent/", "NP2_0.jpg", null, true,
                new Distortion[]{
                        new Distortion(DistortionType.RightPers, 5f, 10, 60, 300),
                        new Distortion(DistortionType.TopPers, 5f, 10, 60, 300)
                },
                500, 300, 20, 8);
//        testRobustFeature("src/main/resources/image/single_distortion/detergent/", 1, "ttt_log", true,
//                new Distortion(DistortionType.TopPers, 5f, 10, 100, 300), 500, 300, 20, 3);
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
        ScaleDown,
        ScaleUp,
        Light,
        Rotation
    }

    static class Distortion {
        DistortionType dType;
        float dStep;
        int dNum;
        int tFPNum;
        int robustDisThd;

        /**
         *
         * @param dType             what kind of distortion is used to generate distorted images
         * @param dStep             step value for distortion
         * @param dNum              how many distorted images should be generated
         * @param tFPNum            the number of template feature points
         * @param robustDisThd      matching distance threshold used in extracting robust feature point
         */
        public Distortion(DistortionType dType, float dStep, int dNum, int tFPNum, int robustDisThd) {
            this.dType = dType;
            this.dStep = dStep;
            this.dNum = dNum;
            this.tFPNum = tFPNum;
            this.robustDisThd = robustDisThd;
        }
    }

    /**
     *
     * @param filePath          the path of directory in where the images are
     * @param templateValue     the value used as template
     * @param logName           file used to log. If logName is not given, directory_name+parameters will be used
     * @param rewriteHP         whether the method should write down hyperparameters again if the file already exist
     * @param distortion        distortion types and parameters
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
                                  Distortion distortion,
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
        int testStep = 0;
        String distortionStr = "";  //used in log file name
        float dStep = distortion.dStep;
        int dNum = distortion.dNum;
        int tFPNum = distortion.tFPNum;
        int robustDisThd = distortion.robustDisThd;
        switch (distortion.dType) {
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
                testStep = 1;
                break;
            case BottomPers:
                distortedImg = ImageProcessor.changeToBottomPerspective(tImg, dStep, dNum);
                distortionStr = "bp";
                break;
            case ScaleDown:
                distortedImg = ImageProcessor.scaleImage(tImg, dStep, dNum);
                testStep = -5;
                distortionStr = "sd";
                break;
            case ScaleUp:
                distortedImg = ImageProcessor.scaleImage(tImg, dStep, dNum);
                testStep = 5;
                distortionStr = "su";
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

        for (Mat img : distortedImg) {
            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(img));
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
            int i = templateValue + testStep * k;
            Mat qImg = ImageUtil.loadMatImage(filePath+i+".png");
            //assume we use ORB feature points in default
            ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, qFPNum);
            MatOfDMatch matches = ImageProcessor.matchWithRegression(qIF, tIF, 5, matchDisThd, matchPosThd);
            float p = (float)matches.total() / tIF.getSize();
            System.out.printf("%d: %f\n", i, p);
//            pw.printf("%d: %f\n", i, (float)matches.total() / tIF.getSize());
            fNames.add(""+testStep*k);
            pres.add(p);
            //display matches
            Mat display = new Mat();
            Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(),  matches, display);
            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
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
            Mat display = new Mat();
            Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(),  matches, display);
            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
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

    static int kVStep = 18;
    static int kHStep = 3;
    static void testCombinedDistortion(String filePath,
                                       String tFile,
                                       String logName,
                                       boolean rewriteHP,
                                       Distortion[] ds,
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
        String tFName = filePath + tFile;
        Mat tImg = ImageUtil.loadMatImage(tFName);
        List<ImageFeature> tIFs = new ArrayList<>();
        String distortionStr = tFile.split("\\.")[0];  //used in log file name
        int testStep = 0;
        for (Distortion d : ds) {
            List<Mat> distortedImg = null;
            float dStep = d.dStep;
            int dNum = d.dNum;
            int tFPNum = d.tFPNum;
            int robustDisThd = d.robustDisThd;
            distortionStr += "_";
            switch (d.dType) {
                case LeftPers:
                    distortedImg = ImageProcessor.changeToLeftPerspective(tImg, dStep, dNum);
                    testStep = -3;
                    distortionStr += "lp";
                    break;
                case RightPers:
                    distortedImg = ImageProcessor.changeToRightPerspective(tImg, dStep, dNum);
                    testStep = 3;
                    distortionStr += "rp";
                    break;
                case TopPers:
                    distortedImg = ImageProcessor.changeToTopPerspective(tImg, dStep, dNum);
                    distortionStr += "tp";
                    break;
                case BottomPers:
                    distortedImg = ImageProcessor.changeToBottomPerspective(tImg, dStep, dNum);
                    distortionStr += "bp";
                    break;
                case ScaleDown:
                    distortedImg = ImageProcessor.scaleImage(tImg, dStep, dNum);
                    testStep = -5;
                    distortionStr += "sd";
                    break;
                case ScaleUp:
                    distortedImg = ImageProcessor.scaleImage(tImg, dStep, dNum);
                    testStep = 5;
                    distortionStr += "su";
                    break;
                case Light:
                    distortedImg = ImageProcessor.lightImage(tImg, dStep, dNum);
                    distortionStr += "l";
                    break;
                case Rotation:
                    distortedImg = ImageProcessor.rotatedImage(tImg, dStep, dNum);
                    distortionStr += "r";
                    break;
                default:
                    break;
            }
            distortionStr += "_ds" + dStep + "_dn" + dNum + "_tfpn" + tFPNum + "_rdt" + robustDisThd;
            tIFs.add(ImageProcessor.extractRobustFeatures(tImg, distortedImg, tFPNum, robustDisThd, DescriptorType.ORB, null));
        }

        File dir = new File(filePath);
        //prepare for writing to log file
        if (logName == null) {
            logName = dir.getName() + "_" + distortionStr + "_qfpn" + qFPNum + "_mdt" + matchDisThd + "_mpt" + matchPosThd + "_tn" + testNum;
        }
        File logFile = new File(logName);
        boolean append = logFile.exists();
        logFile.createNewFile();
        PrintWriter pw = new PrintWriter(new FileOutputStream(logFile, true));
        List<String> fNames = new ArrayList<>();
        List<Float> pres = new ArrayList<>();

        int templateValue = new Integer(tFile.split("\\.")[0].split("_")[1]);
        int vValue = new Integer(tFile.split("\\.")[0].split("_")[0].replace("NP",""));
        System.out.printf("-----%s-------\n", dir.getName());
        for (int d=1; d <= 3; d++) {
            System.out.printf("******%d*******\n",d);
            for (int k = 1; k <= testNum; k++) {
                ImageFeature tIF = constructTemplateFP(tIFs, k*kHStep, (d-vValue)* kVStep, 80);
                int i = templateValue + testStep * k;
                Mat qImg = ImageUtil.loadMatImage(filePath + "NP" + d + "_" + i + ".jpg");
                //assume we use ORB feature points in default
                ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, qFPNum);
                MatOfDMatch matches = ImageProcessor.matchWithRegression(qIF, tIF, 5, matchDisThd, matchPosThd);
                float p = (float) matches.total() / tIF.getSize();
                System.out.printf("%d: %f\n", i, p);
//            pw.printf("%d: %f\n", i, (float)matches.total() / tIF.getSize());
                fNames.add("" + d + "_" + testStep * k);
                pres.add(p);
//                    //display matches
//                    Mat display = new Mat();
//                    Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(), matches, display);
//                    ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
            }
        }
        System.out.printf("******false*******\n");
        //assume all false image are named as "f"+number+".png"
        for (int i = 1; i <= Integer.MAX_VALUE; i++) {
            File fImg = new File(filePath + "f" + i + ".png");
            if (!fImg.exists()) break;
            ImageFeature tIF = tIFs.get(0);
            Mat qImg = ImageUtil.loadMatImage(fImg.getAbsolutePath());
            //assume we use ORB feature points in default
            ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, qFPNum);
            MatOfDMatch matches = ImageProcessor.matchWithRegression(qIF, tIF, 5, matchDisThd, matchPosThd);
            float p = (float) matches.total() / tIF.getSize();
            System.out.printf("f%d: %f\n", i, p);
//            pw.printf("f%d: %f\n", i, (float)matches.total() / tIF.getSize());
            fNames.add("f" + i);
            pres.add(p);
//                //display matches
//                Mat display = new Mat();
//                Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(), matches, display);
//                ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
        }
        if (!append || rewriteHP) {
            pw.println(hyperParams);
            pw.printf("distortions: %s, qFPNum: %d, matchDisThd: %d, matchPosThd: %d\n",
                    distortionStr, qFPNum, matchDisThd, matchPosThd);

            pw.printf("-----%s, %s-------\n", dir.getName(), tFile);
            pw.print("tIF/change\t");
            for (String f : fNames)
                pw.print(f + "\t");
            pw.println();   //next line
        }
        pw.print(templateValue + "\t");
        for (int i=0; i < pres.size(); i++) {
            float p = pres.get(i);
            pw.printf("%.2f\t", p);
            if ((i+1) % fNames.size() == 0 && i < pres.size()-1) {
                pw.println();   //next line
                pw.print(templateValue + "\t");
            }
        }
        pw.println();   //next line
        pw.close();
    }

    //assume tIFs contains only horizontal robust and vertical robust ImageFeature, at total 2.
    static ImageFeature constructTemplateFP(List<ImageFeature> tIFs, float hd, float vd, int tNum) {
        //calculate ratios
        float hr = Math.abs(hd)/(Math.abs(hd) + Math.abs(vd));
        float vr = Math.abs(vd)/(Math.abs(hd) + Math.abs(vd));
        ImageFeature IF1;
        ImageFeature IF2;
        //guarantee the feature point robust on more-changed orientation is returned at first
        if (hr >= vr) {
            int num = (int)(hr * tNum);
            IF1 = (num > tIFs.get(0).getSize())?tIFs.get(0) : tIFs.get(0).subImageFeature(0, num);
            IF2 = tIFs.get(1);
        } else {
            int num = (int)(vr * tNum);
            IF1 = (num > tIFs.get(1).getSize())?tIFs.get(1) : tIFs.get(1).subImageFeature(0, num);
            IF2 = tIFs.get(0);
        }
        if (IF1.getSize() >= tNum) return IF1;

        List<KeyPoint> kp = new ArrayList<>(IF1.getObjectKeypoints().toList());
        Mat des = new Mat();//new Size(IF1.getDescriptors().cols(),tNum), IF1.getDescriptors().type());
        des.push_back(IF1.getDescriptors());

        List<KeyPoint> kp1 = IF1.getObjectKeypoints().toList();
        List<KeyPoint> kp2 = IF2.getObjectKeypoints().toList();
        for (int i=0; i < kp2.size(); i++) {
            KeyPoint k = kp2.get(i);
            boolean newFP = true;
            for (KeyPoint k1 : kp1) {
                if (k1.pt.x != k.pt.x || k1.pt.y != k.pt.y)
                    continue;
                else {
                    newFP = false;
                    break;
                }
            }
            if (newFP) {
                kp.add(k);
                Mat tMat = IF2.getDescriptors().row(i);
                des.push_back(tMat);
            }
            if (kp.size() >= tNum)
                break;
        }
        MatOfKeyPoint tKP = new MatOfKeyPoint();
        tKP.fromList(kp);
//        System.out.printf("construct FP size: %d", kp.size());
        return new ImageFeature(tKP, des, IF1.getDescriptorType());
    }

    private static void testRobustFeature(String filePath, String templateImg) throws IOException {
        Mat tImg = ImageUtil.loadMatImage(filePath+templateImg);
//        List<Mat> distortedImg = ImageProcessor.rotatedImage(img, 5f, 5);
        List<Mat> distortedImg = ImageProcessor.scaleImage(tImg, -0.1f, 5);
//        List<Mat> distortedImg = ImageProcessor.lightImage(img, -0.1f, 5);
//        List<Mat> distortedImg = ImageProcessor.changeToLeftPerspective(tImg, 5f, 10);
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
                Mat display = new Mat();
                Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(),  matches, display);
                ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
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
        String imgPath = "src/main/resources/image/NP1_0.jpg";
        BufferedImage img = ImageUtil.loadImage(imgPath);
        ObjectDetector objectDetector = new ObjectDetector();
        objectDetector.init();
        List<Recognition> recognitions = objectDetector.recognizeImage(img);
        int i = 0;
        for (Recognition recognition : recognitions) {
            System.out.printf("Object: %s - confidence: %f box: %s\n",
                    recognition.getTitle(), recognition.getConfidence(), recognition.getLocation());

//            StorageUtil.saveRecognitionToFile(recognition,"test" + (++i));
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
