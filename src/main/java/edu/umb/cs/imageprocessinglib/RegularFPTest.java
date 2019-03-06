package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.DescriptorType;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import javafx.beans.binding.IntegerBinding;
import javafx.util.Pair;
import org.opencv.core.*;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RegularFPTest {

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        orb = ORB.create(500, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 20);
        positiveTest();
//        negativeTest();
//        compareTFWithRegular("src/main/resources/image/street_car/");
//        compareTFWithRegular("src/main/resources/image/indoor/");
//        compareTFWithRegular("src/main/resources/image/frame/");
//        compareTFWithRegular("src/main/resources/image/standing/");
//        tmp();
//        testFP();
//        mergeImagesInDir();
//        extractObjectsInDir("src/main/resources/image/multi_distortion/coffee_mate/");
//        testRegularFP("src/main/resources/image/horse1/", "Motorcycle_s1.00.JPG");
//        testRegularFP("src/main/resources/image/motorcycle1/", "000.JPG");
//        testRegularFP("src/main/resources/image/single_distortion/horse1/", "000.JPG");
//        testMaxMin("src/main/resources/image/single_distortion/motorcycle1/", "000.JPG");
//        testMaxMin("src/main/resources/image/toy_car/", "000.png");
//        testMaxMin("src/main/resources/image/horse1/", "000.JPG");
//        scaleDownImage("src/main/resources/image/horse1/000.JPG");
//        String[] dirNames = {"lego_man", "shoe"};//, "furry_elephant", "toy_bear", "van_gogh", "furry_bear", "duck_cup"};
//        for (String dir : dirNames) {
//            scaleDownImage("src/main/resources/image/"+dir+"/0.png");
//        }
    }

    static void tmp() throws IOException {
        String path = "src/main/resources/image/";
        Mat qImg = ImageUtil.loadMatImage(path+"box.png");
        Mat tImg = ImageUtil.loadMatImage(path+"box_in_scene.png");
        ImageFeature tIF = ImageProcessor.extractORBFeatures(tImg, 100);
        ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, 100);
        MatOfDMatch match = ImageProcessor.matchImages(tIF, qIF);
        List<DMatch> m = match.toList();
        m.sort((o1, o2) -> {
            return (int) (o1.distance - o2.distance);
        });
        m = m.subList(0,10);
        match.fromList(m);

        Mat display = new Mat();
        Features2d.drawMatches(qImg, qIF.getObjectKeypoints(),tImg, tIF.getObjectKeypoints(),  match, display);
        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
        System.out.printf("matching ratio: %.2f", (float)match.total()/tIF.getSize());
    }

    static void positiveTest() throws IOException {
        int fpNum = 100;
        int diff = 30;
        int dis_thd = 500;
        String path = "src/main/resources/image/single_distortion/";
        String[] dirNames = {"lego_man", "shoe", "furry_elephant", "furry_bear", "girl_statue"};
        System.out.printf("image\t");
        List<Integer> ns = new ArrayList<>();
        for (int i=0; i<360; i+=5) {
            System.out.printf("%d\t",i);
            ns.add(i);
        }
        System.out.println();
        for (String dir : dirNames) {
            //load all images and extract features first
            List<Mat> imgs = ns.stream().map(n -> {return ImageUtil.loadMatImage(new File(path+dir+"/"+n+".png").getAbsolutePath());}).collect(Collectors.toList());
            List<ImageFeature> ifs = imgs.stream().map(i->{return ImageProcessor.extractORBFeatures(i, fpNum);}).collect(Collectors.toList());

            //test angle view difference from 5 to 15 degrees
            for (int i=5; i<=diff; i+=5) {
                System.out.printf("%s_%d\t",dir,i);

                //use images of different view angles as template image
                for (int d=0; d<360; d+=5) {
                    //assume we use ORB feature points in default
                    ImageFeature tIF = ifs.get(d/5);
                    ImageFeature qIF = ifs.get((d+i)%360/5);
//                   MatOfDMatch matches = ImageProcessor.matchImages(qIF, tIF);
//                   MatOfDMatch matches = ImageProcessor.BFMatchImages(qIF, tIF);
                    MatOfDMatch matches = ImageProcessor.BFMatchWithCrossCheck(qIF, tIF);
//                    MatOfDMatch matches = ImageProcessor.matchWithDistanceThreshold(qIF, tIF, dis_thd, true);
                    float p = (float)matches.total() / tIF.getSize();
                    System.out.printf("%.2f\t", p);
                }
                System.out.println();
            }

            //test angle view difference from -5 to -15 degrees
            for (int i=-5; i>=-diff; i-=5) {
                System.out.printf("%s_%d\t",dir,i);

                for (int d=0; d<360; d+=5) {
                    //assume we use ORB feature points in default
                    ImageFeature tIF = ifs.get(d/5);
                    ImageFeature qIF = ifs.get((d+i+360)%360/5);
//                   MatOfDMatch matches = ImageProcessor.matchImages(qIF, tIF);
//                   MatOfDMatch matches = ImageProcessor.BFMatchImages(qIF, tIF);
                    MatOfDMatch matches = ImageProcessor.BFMatchWithCrossCheck(qIF, tIF);
//                    MatOfDMatch matches = ImageProcessor.matchWithDistanceThreshold(qIF, tIF, dis_thd, true);
                    float p = (float)matches.total() / tIF.getSize();
                    System.out.printf("%.2f\t", p);
                }
                System.out.println();
            }
        }
    }

    //compare image to false images downloaded from google
    static void negativeTest() throws IOException {
        int fpNum = 100;
        String path = "src/main/resources/image/single_distortion/";
        String[] dirNames = {"lego_man", "shoe", "furry_elephant", "furry_bear", "girl_statue"};
        System.out.printf("image\t");
        for (int i=1; i<=100; i++) {
            System.out.printf("f%d\t",i);
        }
        System.out.println();
        for (String dir : dirNames) {
            File fDir = new File(path+dir+"/false/");
            File[] files = fDir.listFiles((d, name) -> !name.equals(".DS_Store"));  //exclude mac hidden system file
            List<ImageFeature> qIFs = new ArrayList<>(Arrays.asList(files)).stream().map(
                    file -> {return ImageProcessor.extractORBFeatures(ImageUtil.loadMatImage(file.getAbsolutePath()), fpNum);}).collect(Collectors.toList());
            for (int i=0; i<360; i+=35) {
                System.out.printf("%s_%d\t",dir,i);
                Mat img = ImageUtil.loadMatImage(path+dir+"/"+i+".png");
                ImageFeature tIF = ImageProcessor.extractORBFeatures(img, fpNum);

                for (int d=0; d < qIFs.size(); d++) {
//                    Mat qImg = ImageUtil.loadMatImage(f.getAbsolutePath());
                    //assume we use ORB feature points in default
                    ImageFeature qIF = qIFs.get(d);
//                   MatOfDMatch matches = ImageProcessor.matchImages(qIF, tIF);
//                   MatOfDMatch matches = ImageProcessor.BFMatchImages(qIF, tIF);
                    MatOfDMatch matches = ImageProcessor.BFMatchWithCrossCheck(qIF, tIF);
//                    MatOfDMatch matches = ImageProcessor.matchWithDistanceThreshold(qIF, tIF, 350, true);
                    float p = (float)matches.total() / tIF.getSize();
                    System.out.printf("%.2f\t", p);
                }
                System.out.println();
            }
        }
    }

    static void compareTFWithRegular(String dirPath) throws IOException {
        ObjectDetector objectDetector = new ObjectDetector();
        objectDetector.init();
        int diff = 5;  //the angle difference threshold
        List<Float> lRatio = new ArrayList<>(); //ratios of left change
        List<Float> rRatio = new ArrayList<>(); //ratios of right change
        List<Float> olRatio = new ArrayList<>(); //ratios of original
        List<Float> orRatio = new ArrayList<>(); //ratios of original
        int num = 58;   //the number of images inside the directory
//        int num = 6;

        List<Integer> ns = new ArrayList<>();
        for (int i=1; i<=num; i+=1) {
            ns.add(i);
        }

        //load all images and do their recognition, calculate image feature points
        List<Mat> mats = ns.stream().map(n -> {return ImageUtil.loadMatImage(new File(dirPath+n+".png").getAbsolutePath());}).collect(Collectors.toList());
        List<BufferedImage> imgs = ns.stream().map(n -> {return ImageUtil.loadImage(new File(dirPath+n+".png").getAbsolutePath());}).collect(Collectors.toList());
        List<List<Recognition>> rgs = imgs.stream().map(i->{return objectDetector.recognizeImage(i);}).collect(Collectors.toList());
        List<List<ImageFeature>> ifsList = new ArrayList<>();
        for (int i=0; i < rgs.size(); i++) {
            List<Recognition> rg = rgs.get(i);
            Mat img = mats.get(i);
            List<ImageFeature>  ifs = rg.stream().map(r->{return ImageProcessor.extractORBFeatures(r.cropPixels(img,r.getModelSize()),100);}).collect(Collectors.toList());
            ifsList.add(ifs);
        }

        for (int i=1; i <= num-diff; i+=1) {
//        for (int i=0; i <= 360-diff; i+=5) {
//        for (int i=0; i <= 260-diff; i+=5) {
            List<Recognition> lr = rgs.get(i-1);
            //only consider the case of multiple recognized objects
            if (lr.size() >= 1) {
                //get template image and corresponding recognized part's feature point list, one ImageFeature for one recognized object
                Mat img = mats.get(i-1);
                List<ImageFeature> ifs = ifsList.get(i-1);
//                for (int d=5; d<=diff; d+=5) {
                for (int d=1; d<=diff; d+=1) {
//                    String tName = String.format("%03d.jpg",i+d);
                    //get query images' TF recognition
                    List<Recognition> rs = rgs.get(i+d-1);
                    if (rs.size() >= 1) {
                        Mat t = mats.get(i+d-1);
                        int count = 0;  //count how many object matching happen
                        float sumRatio = 0;
                        for (int index=0; index < rs.size(); index++) {
                            Recognition r = rs.get(index);
                            //get query image recognized part's feature point
                            ImageFeature tmpIF = ifsList.get(i+d-1).get(index);
                            float max = 0;
                            for (int k=0; k < lr.size(); k++) {
                                //if there are multiple object recognized as same title, take the one with highest matching ratio
                                if (r.getTitle().equals(lr.get(k).getTitle())) {
                                    if (ifs.get(k).getSize()<=0)
                                        continue;
//                                    MatOfDMatch m = ImageProcessor.matchWithDistanceThreshold(tmpIF,ifs.get(k),300);
                                    MatOfDMatch m = ImageProcessor.BFMatchWithCrossCheck(tmpIF,ifs.get(k));
//                                    MatOfDMatch m = ImageProcessor.matchImages(tmpIF,ifs.get(k));
//                                    MatOfDMatch m = ImageProcessor.matchWithRegression(tmpIF,ifs.get(k));
//                                    MatOfDMatch m = ImageProcessor.matchWithDistanceThreshold(tmpIF,ifs.get(k),300, true);
                                    float ratio = (float)m.total()/ifs.get(k).getSize();
                                    if (ratio > max) max = ratio;
                                }
                            }
                            if (max > 0) {
                                count++;
                                sumRatio += max;
                            }
                        }
                        if (count > 0) {
                            //calculate matching ratio of directly matching images
                            ImageFeature oIF = ImageProcessor.extractORBFeatures(img, 100*count);
//                            olRatio.add((float)(ImageProcessor.matchWithDistanceThreshold(ImageProcessor.extractORBFeatures(t,100*count), oIF,300, true).total())/oIF.getSize());
//                            olRatio.add((float)(ImageProcessor.matchImages(ImageProcessor.extractORBFeatures(t,100*count), oIF).total())/oIF.getSize());
//                            olRatio.add((float)(ImageProcessor.matchWithRegression(ImageProcessor.extractORBFeatures(t,100*count), oIF).total())/oIF.getSize());
                            olRatio.add((float)(ImageProcessor.BFMatchWithCrossCheck(ImageProcessor.extractORBFeatures(t,100*count), oIF).total())/oIF.getSize());
//                            olRatio.add((float)(ImageProcessor.matchWithDistanceThreshold(ImageProcessor.extractORBFeatures(t,100*count), oIF, 300).total())/oIF.getSize());
                            lRatio.add(sumRatio/count);
                        }
                    }
                }
            }
            //loop in reverse order
//            String rName = dirPath+String.format("%03d.jpg",360-i);
//            String rName = dirPath+String.format("%03d.jpg",260-i);
            //get template image's recognition
            List<Recognition> rr = rgs.get(num-i);
            //only consider the case of multiple recognized objects
            if (rr.size() >= 1) {
                //get template image and corresponding recognized part's feature point list, one ImageFeature for one recognized object
                Mat img = mats.get(num-i);
                List<ImageFeature> ifs = ifsList.get(num-i);
//                for (int d=5; d<=diff; d+=5) {
                for (int d=1; d<=diff; d+=1) {
//                    String tName = String.format("%03d.jpg", 360 - (i + d));
//                    String tName = String.format("%d.png", num + 1 - (i + d));
//                    String tName = String.format("%03d.jpg", 260 - (i + d));
                    //get query images' TF recognition
                    List<Recognition> rs = rgs.get(num-(i+d));
                    if (rs.size() >= 1) {
                        Mat t = mats.get(num-(i+d));
                        int count = 0;  //count how many object matching happen
                        float sumRatio = 0;
                        for (int index=0; index<rs.size(); index++) {
                            Recognition r = rs.get(index);
                            ImageFeature tmpIF = ifsList.get(num-(i+d)).get(index);
                            float max = 0;
                            for (int k = 0; k < rr.size(); k++) {
                                //if there are multiple object recognized as same title, take the one with highest matching ratio
                                if (r.getTitle().equals(rr.get(k).getTitle())) {
                                    if (ifs.get(k).getSize()<=0)
                                        continue;
//                                    MatOfDMatch m = ImageProcessor.matchWithDistanceThreshold(tmpIF, ifs.get(k), 300);
//                                    MatOfDMatch m = ImageProcessor.matchWithDistanceThreshold(tmpIF, ifs.get(k),300, true);
                                    MatOfDMatch m = ImageProcessor.BFMatchWithCrossCheck(tmpIF, ifs.get(k));
//                                    MatOfDMatch m = ImageProcessor.matchImages(tmpIF, ifs.get(k));
//                                    MatOfDMatch m = ImageProcessor.matchWithRegression(tmpIF, ifs.get(k));
//                                    displayMatches(rr.get(k).cropPixels(img), r.cropPixels(t), ifs.get(k), tmpIF, m);
                                    float ratio = (float)m.total() / ifs.get(k).getSize();
                                    if (ratio > max) max = ratio;
                                }
                            }
                            if (max > 0) {
                                count++;
                                sumRatio += max;
                            }
                        }
                        if (count > 0) {
                            //calculate matching ratio of directly matching images
                            ImageFeature oIF = ImageProcessor.extractORBFeatures(img, 100 * count);
//                            orRatio.add((float)(ImageProcessor.matchWithDistanceThreshold(ImageProcessor.extractORBFeatures(t,100*count), oIF, 300).total())/oIF.getSize());
//                            orRatio.add((float)(ImageProcessor.matchWithDistanceThreshold(ImageProcessor.extractORBFeatures(t,100*count), oIF,300, true).total())/oIF.getSize());
                            ImageFeature tIF = ImageProcessor.extractORBFeatures(t,100*count);
                            MatOfDMatch m = ImageProcessor.BFMatchWithCrossCheck(tIF, oIF);
//                            MatOfDMatch m = ImageProcessor.matchImages(tIF, oIF);
//                            MatOfDMatch m = ImageProcessor.matchWithRegression(tIF, oIF);
                            orRatio.add((float)(m.total())/oIF.getSize());
                            rRatio.add(sumRatio / count);
//                            displayMatches(t, img, tIF, oIF, m);
                        }
                    }
                }
            }
        }
        double a_l = lRatio.stream().mapToDouble(Float::doubleValue).sum()/lRatio.size();
        double a_r = rRatio.stream().mapToDouble(Float::doubleValue).sum()/rRatio.size();
        double l = olRatio.stream().mapToDouble(Float::doubleValue).sum()/olRatio.size();
        double r = orRatio.stream().mapToDouble(Float::doubleValue).sum()/orRatio.size();
        System.out.printf("*******Average*******\nleft tf: %.2f\nright tf: %.2f\nleft ori: %.2f\nright ori: %.2f\n", a_l, a_r, l, r);
        System.out.printf("tf left:\n%s\n", lRatio);
        System.out.printf("tf right:\n%s\n", rRatio);
        System.out.printf("ori left:\n%s\n", olRatio);
        System.out.printf("ori right:\n%s\n", orRatio);
//        System.out.printf("Angle\t");
//        for (int i=5; i<=diff; i+=5) {
//            System.out.printf("%d\t",i);
//        }
//        System.out.println();
//        for (int i=0; i<=rRatio.size(); i++) {
////            System.out.printf();
//        }
    }

    static void displayMatches(Mat tImg, Mat qImg, ImageFeature tIF, ImageFeature qIF, MatOfDMatch m) {
        Mat d = new Mat();
        Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(), m, d);
        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(d));
    }

    static void testFP() throws IOException {
//        String[] dirNames = {"lego_man", "shoe", "furry_elephant", "toy_bear", "van_gogh", "furry_bear", "duck_cup", "furry_dog", "baby_cream", "girl_statue"};
        String[] dirNames = {"girl_statue"};
        int tValue = 0;
        int tNum = 8;
        List<Float> ratios = new ArrayList<>();
//        int index = 0;
        //when change tValue, remember to change matching ratio calculation below
//        for (int tValue=0; tValue <= 45; tValue+=5)
            for (String dir : dirNames) {
                String path = "src/main/resources/image/single_distortion/"+dir+"/";
                Mat tImg = ImageUtil.loadMatImage(path+tValue+".png");
                ImageFeature tIF = ImageProcessor.extractORBFeatures(tImg, 100);
//            List<Mat> dImg = ImageProcessor.changeToLeftPerspective(tImg, 5, 10);
//            ImageFeature tIF = ImageProcessor.extractRobustFeatures(tImg, dImg, 100, 300, DescriptorType.ORB);
//            ImageFeature tIF = ImageProcessor.extractORBFeatures(tImg, 500);
//            tIF = pickNRandomFP(tImg, tIF, 100);
//                for (int i=1; i <= tNum; i++) {
                List<Mat> mats = new ArrayList<>();
                for (int i=2; i <= tNum; i+=5) {
                    if (ratios.size()<i)
                        ratios.add(0f);
                    int value = tValue + i*5;
                    Mat qImg = ImageUtil.loadMatImage(path+value+".png");
                    ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, 500);
//                    MatOfDMatch m = ImageProcessor.matchWithDistanceThreshold(qIF, tIF, 300);
//                    MatOfDMatch m = ImageProcessor.matchImages(qIF, tIF);
//                    MatOfDMatch m = ImageProcessor.matchWithRegression(qIF, tIF, 5, 300, 20);
//                    ratios.set(i-1, ratios.get(i-1) + (float)m.total() / tIF.getSize());

//                Mat display = new Mat();
//                Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(), m, display);
//                ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
                    List<MatOfDMatch> tm = Arrays.asList(ImageProcessor.matchWithDistanceThreshold(qIF, tIF, 300),
                            ImageProcessor.matchImages(qIF, tIF),
                            ImageProcessor.matchWithRegression(qIF, tIF, 5, 300, 20));
                    List<Mat> ttt = tm.stream().map(m->{
                        Mat d = new Mat();
                        Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(), m, d);
                        return d;
                    }).collect(Collectors.toList());
                    Mat t = new Mat();
                    Core.vconcat(ttt, t);
                    mats.add(t);
                }

//                for (int i=1; i<=3; i++) {
                for (int i=1; i<=1; i++) {
                    if (ratios.size()<i+tNum)
                        ratios.add(0f);
                    Mat qImg = ImageUtil.loadMatImage(path+"f"+i+".png");
                    ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, 500);
//                    MatOfDMatch m = ImageProcessor.matchWithDistanceThreshold(qIF, tIF, 300);
//                    MatOfDMatch m = ImageProcessor.matchImages(qIF, tIF);
//                    MatOfDMatch m = ImageProcessor.matchWithRegression(qIF, tIF, 5, 100, 10);
//                    ratios.set(i+tNum-1, ratios.get(i+tNum-1) + (float)m.total() / tIF.getSize());
//                    Mat display = new Mat();
//                    Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(), m, display);
//                    ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
                    List<MatOfDMatch> tm = Arrays.asList(ImageProcessor.matchWithDistanceThreshold(qIF, tIF, 300),
                            ImageProcessor.matchImages(qIF, tIF),
                            ImageProcessor.matchWithRegression(qIF, tIF, 5, 300, 20));
                    List<Mat> ttt = tm.stream().map(m->{
                        Mat d = new Mat();
                        Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(), m, d);
                        return d;
                    }).collect(Collectors.toList());
                    Mat t = new Mat();
                    Core.vconcat(ttt, t);
                    mats.add(t);
                }
//                index++;
//                final int k = index;
//                ratios.stream().forEach(f->System.out.printf("%.2f\t", f/k));
//                System.out.println();
                Mat t = new Mat();
                Core.hconcat(mats, t);
                ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(t));
            }
        ratios.stream().forEach(f->System.out.printf("%.2f\t", f/dirNames.length));
        System.out.println();
    }

    static void testRegularFP(String filePath, String templateImg) throws IOException {
        //hyperparameter for picking random feature points
        int num = 100;

        File tImgFile = new File(filePath + templateImg);
        if (tImgFile == null || !tImgFile.isFile())
            return;
        Mat tImg = ImageUtil.BufferedImage2Mat(ImageIO.read(tImgFile));
        ImageFeature tIF = ImageProcessor.extractORBFeatures(tImg, 500);
        List<KeyPoint> tKP = tIF.getObjectKeypoints().toList();
//        System.out.printf("template key points number: %d\n", tIF.getSize());
        File dir = new File(filePath);
        File[] directoryListing = dir.listFiles();
        long totalTime = 0;
        System.out.printf("{| class=\"wikitable\"\n |+%dX500\n |- \n!file name \n!precision\n !matched template FP\n |- \n", num);
        if (directoryListing != null) {
            List<File> files = new ArrayList<>(Arrays.asList(directoryListing));
            files.sort(Comparator.comparing(File::getName));
            for (File f : files) {
                if (f.getName().equals(templateImg))
                    continue;
                System.out.printf("!%s", f.getName());
                Mat qImg = ImageUtil.BufferedImage2Mat(ImageIO.read(f));
//                ImageFeature qIF = ImageProcessor.extractFeatures(qImg);
                ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, 500);
                List<KeyPoint> qKP = qIF.getObjectKeypoints().toList();

                //randomly pick num feature points in template and query image
                ImageFeature rTIF = pickNRandomFP(tImg, tIF, num);
                ImageFeature rQIF = pickNRandomFP(qImg, qIF, 500);
//                ImageFeature rQIF = qIF;
//                List<DMatch> mL = matchImage(qIF, tIF);
                List<DMatch> mL = matchImage(rQIF, rTIF);
                MatOfDMatch m = new MatOfDMatch();
                m.fromList(mL);

//                System.out.printf("%s Match number: %d, Precision: %f\n\n", f.getName(), m.total(), (float)m.total()/ tIF.getSize());
//                System.out.printf("%s %dx%d, Match number: %d, Precision: %f\n", f.getName(), rTIF.getSize(), rQIF.getSize(), m.total(), (float)m.total()/ rTIF.getSize());
                System.out.printf("\n|%f\n", (float) m.total() / rTIF.getSize());
                //display matches
                Mat display = new Mat();
//                    Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(), m, display);
//                Features2d.drawMatches(qImg, rQIF.getObjectKeypoints(), tImg, rTIF.getObjectKeypoints(), m, display);
//                ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));

//                System.out.print("| ");
//                for (int i = 0; i < mL.size(); i++) {
//                    System.out.printf("%d ", mL.get(i).trainIdx);
//                    System.out.printf("%d\t", mL.get(i).trainIdx);
//                    if ((i+1)%20==0) System.out.println();
//                }
//                System.out.println("\n---------");
                System.out.println("\n|-");
            }
            System.out.println("|}");
        }
//        System.out.printf("average time:%f", (float)totalTime/(float)(directoryListing.length-1)/1000.0);
    }

    static void scaleDownImage(String filePath) throws IOException {
        Mat img = ImageUtil.loadMatImage(filePath);
        String dirPath = new File(filePath).getParent()+"_scale";
        new File(dirPath).mkdir();
        for (float s=0.95f; s >= 0.5f; s-=0.05f) {
            Mat sImg = ImageUtil.scaleImage(img, s);
//            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(sImg));
            ImageUtil.saveImage(ImageUtil.Mat2BufferedImage(sImg), dirPath + String.format("/%d.png",(int)Math.ceil(s * 100)));
        }
        ImageUtil.saveImage(ImageUtil.Mat2BufferedImage(img), dirPath + "/100.png");
    }

    static void mergeImagesInDir() throws IOException {
        String filePath = "src/main/resources/image/single_distortion/";
        String[] dirNames = {"lego_man", "shoe", "furry_elephant", "toy_bear", "van_gogh", "furry_bear", "duck_cup", "baby_cream"};
        List<Mat> images = new ArrayList<>();
        for (String dir : dirNames) {
            images.add(ImageUtil.BufferedImage2Mat(ImageUtil.loadImage(filePath+dir+"/0.png")));
        }
        Mat aImg = new Mat();
        Core.hconcat(images, aImg);
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(aImg));

        String dir = "shoe/";
        images = new ArrayList<>();
        for (int i=0; i < 40; i+=5) {
            images.add(ImageUtil.BufferedImage2Mat(ImageUtil.loadImage(filePath+dir+i+".png")));
        }
        Mat bImg = new Mat();
        Core.hconcat(images, bImg);
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(bImg));

        images = ImageProcessor.changeToLeftPerspective(ImageUtil.BufferedImage2Mat(ImageUtil.loadImage(filePath+dir+"0.png")),15, 8);
        Mat cImg = new Mat();
        Core.hconcat(images, cImg);
//        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(cImg));

        images = Arrays.asList(aImg, bImg, cImg);
        Mat dImg = new Mat();
        Core.vconcat(images, dImg);
        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(dImg));
        ImageUtil.saveImage(ImageUtil.Mat2BufferedImage(dImg), "display.png");
//        File dir = new File(filePath);
//        File[] directoryListing = dir.listFiles();
//        if (directoryListing != null) {
//            List<Mat> list = new ArrayList<>();
//            for (File f : directoryListing) {
//                if (f.isFile()) {
//                    list.add(ImageUtil.BufferedImage2Mat(ImageUtil.loadImage(f.getPath())));
//                }
//            }
//            Mat cImg = new Mat();
//            Core.vconcat(list, cImg);
//            ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(cImg));
//        }
    }
    static void extractObjectsInDir(String filePath) throws IOException {
        ObjectDetector objectDetector = new ObjectDetector();
        objectDetector.init();
        File dir = new File(filePath);
        File[] directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (File f : directoryListing) {
                if (f.isFile()) {
                    Mat img = ImageUtil.loadMatImage(f.getAbsolutePath());
//                    BufferedImage image = ImageIO.read(f);
                    Rect rect = new Rect(img.cols()/4, img.rows()/5, img.cols()/2, img.rows()/4*3);
                    Mat image = new Mat(img, rect);
                    List<Recognition> recognitions = objectDetector.recognizeImage(ImageUtil.Mat2BufferedImage(image));
                    int i = 0;
                    for (Recognition recognition : recognitions) {
                        System.out.printf("%s, Object: %s - confidence: %f box: %s\n", f.getName(),
                                recognition.getTitle(), recognition.getConfidence(), recognition.getLocation());
//                        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(recognition.cropPixels(ImageUtil.BufferedImage2Mat(image))));
//                        ImageUtil.saveImage(ImageUtil.Mat2BufferedImage(recognition.cropPixels(ImageUtil.BufferedImage2Mat(image))), f.getAbsolutePath());
                        //the recognition list is sorted by the confidence, so only extract the first recognition
//                        break;

//                        StorageUtil.saveRecognitionToFile(recognition, "test" + (++i));
                    }
                }
            }
        }
    }

    static ORB orb;

    public static ImageFeature pickNRandomFP(Mat image, ImageFeature imageFeature, int n) {
        if (imageFeature.getSize() < n)
            return imageFeature;
        List<KeyPoint> kps = imageFeature.getObjectKeypoints().toList();
        Mat des = new Mat();
        kps = pickNRandom(kps, n);
        MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
        matOfKeyPoint.fromList(kps);
        orb.compute(image, matOfKeyPoint, des);
        return new ImageFeature(matOfKeyPoint, des);
    }

    public static List<KeyPoint> pickNRandom(List<KeyPoint> lst, int n) {
        List<KeyPoint> copy = new LinkedList<KeyPoint>(lst);
        Collections.shuffle(copy);
        return copy.subList(0, n);
    }

    static void testMaxMin(String filePath, String templateImg) throws IOException {
        File tImgFile = new File(filePath + templateImg);
        //if template image can't be found, return
        if (tImgFile == null || !tImgFile.isFile())
            return;
        Mat tImg = ImageUtil.BufferedImage2Mat(ImageIO.read(tImgFile));
        ImageFeature tIF = ImageProcessor.extractORBFeatures(tImg, 500);

        List<Mat> testImages = new ArrayList<>();
        for (int i=5; i <= 35; i+=5) {
            String fileName = filePath + String.format("%03d.JPG", i);
//            String fileName = filePath + String.format("%03d.png", i);
            testImages.add(ImageUtil.BufferedImage2Mat(ImageIO.read(new File(fileName))));
        }
        List<Integer> minTracker = new ArrayList<>();
        ImageFeature tmpIF = ImageProcessor.extractRobustFeatures(tImg, testImages, 100, 350, DescriptorType.ORB, minTracker);

        System.out.printf("number of template robust FP: %d\n", tIF.getSize());
        //calculate min precision
        List<Float> minRatioTracker = IntStream.range(0, minTracker.size()).mapToObj(i->{
            return (float)minTracker.get(i)/(i+1);
        }).collect(Collectors.toList());
        System.out.printf("min num:\t%s\nmin precision:\t%s\n", minTracker, minRatioTracker);
//        List<List<Integer>> fpTrack = analyzeFPsInImages(tIF, testImages);
//        List<Integer> sizes = fpTrack.stream().map(o->o.size()).collect(Collectors.toList());
//        int num = 100;
//        for (int i : sizes) {
//           if (i < num)
//               num = i/10*10;
//        }
//        Pair<Integer, List<Integer>> candidates = maxMin(fpTrack, num);
//        System.out.printf("original template num: %d, min: %d, %d candidates:%s\n",tIF.getSize(), candidates.getKey(), candidates.getValue().size(), candidates.getValue());
//
//        List<KeyPoint> tKP = tIF.getObjectKeypoints().toList();
//        List<KeyPoint> kps = new ArrayList<>();
//        for (int i : candidates.getValue()) {
//            kps.add(tKP.get(i));
//        }
//        Mat des = new Mat();
//        MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
//        matOfKeyPoint.fromList(kps);
//        orb.compute(tImg, matOfKeyPoint, des);
//        ImageFeature imageFeature = new ImageFeature(matOfKeyPoint, des);
//
////        for (Mat qImg : testImages) {
//        File dir = new File(filePath); File[] directoryListing = dir.listFiles();
//        if (directoryListing != null) {
//            List<File> files = new ArrayList<>(Arrays.asList(directoryListing));
//            files.sort(Comparator.comparing(File::getName));
//            for (File f : files) {
//                if (f.getName().equals(templateImg))
//                    continue;
//                Mat qImg = ImageUtil.BufferedImage2Mat(ImageIO.read(f));
//                ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, 500);
//                List<DMatch> matches = matchImage(qIF, imageFeature);
//                MatOfDMatch m = new MatOfDMatch();
//                m.fromList(matches);
////            System.out.printf("%f\n", (float) matches.size() / imageFeature.getSize());
//                System.out.printf("%s: %f\n", f.getName(), (float) matches.size() / imageFeature.getSize());
////                Mat display = new Mat();
////                Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, imageFeature.getObjectKeypoints(), m, display);
////                ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
//            }
//        }
    }

    /**
     * Given a list of candidates, which is represent by a range from 0 to a specific number, this method finds out those
     * which can maximize the minimum counter value.
     * @param input for each target, this argument uses a list containing all qualified candidates
     * @param num   number of returned candidates
     * @return an integer indicating the minimum counter value and a list containing most promising candidates
     */
    static Pair<Integer, List<Integer>> maxMin(List<List<Integer>> input, int num) {
        List<Integer> ret = new ArrayList<>();
        List<Integer> counters = new ArrayList<>();
        Map<Integer, Set<Integer>> tracker = new HashMap<>();   //using set rather than list just in case the input is not properly preprocessed

        //record input into a hashmap, which use candidate as key and a list containing matched target as value
        for (int i=0; i < input.size(); i++) {
            for (int k=0; k < input.get(i).size(); k++) {
                int key = input.get(i).get(k);
                if (tracker.get(key) == null) {
                    tracker.put(key, new HashSet<>());
                }
                //for every candidates, use a list to record its matched target
                tracker.get(key).add(i);
            }
        }

        for (int i=0; i < input.size(); i++)
            counters.add(0);

        int min = 0;
        while (ret.size() < num) {
            List<Integer> mins = new ArrayList<>();
            //find out minimums
            for (int i=0; i < counters.size(); i++) {
                if (counters.get(i)<=min)
                    mins.add(i);
            }
            int max = 0;
            int maxKey = -1;
            for (int i : tracker.keySet()) {
                int c = 0;  //count how many mins can get increased if this candidate is selected
                Set<Integer> ts = tracker.get(i);
                for (int m : mins) {
                    if (ts.contains(m))
                        c++;
                }
                if (c > max) {
                    max = c;
                    maxKey = i;
                }
            }

            //no more optimization can be done, comment this condition if you wanna keep adding new candidate
            if (maxKey == -1)
                break;

            //update
            //all mins get a new matched candidate
            if (max >= mins.size())
                min++;
            if (maxKey != -1)
                ret.add(maxKey);
            for (int i : tracker.get(maxKey))
                counters.set(i, counters.get(i)+1);
            tracker.remove(maxKey);
        }

        return new Pair<Integer, List<Integer>>(min, ret);
    }

    //return a list of lists containing the index of matched feature points
    public static List<List<Integer>> analyzeFPsInImages(ImageFeature tIF, List<Mat> images) {
        List<List<Integer>> ret = new ArrayList<>();
        List<KeyPoint> tKP = tIF.getObjectKeypoints().toList();

        for (Mat img : images) {
            ImageFeature qIF = ImageProcessor.extractORBFeatures(img);
            List<DMatch> mL = matchImage(qIF, tIF);
            ret.add(mL.stream().map(o->o.trainIdx).collect(Collectors.toList()));
        }
        return ret;
    }

    static List<DMatch> matchImage(ImageFeature qIF, ImageFeature tIF) {
        List<DMatch> mL = ImageProcessor.matchWithDistanceThreshold(qIF, tIF, 300).toList();
        //display matches
        mL.sort((o1, o2) -> {
            return (int) (o1.trainIdx - o2.trainIdx);
        });
        return  mL;
    }

}
