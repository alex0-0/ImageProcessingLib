package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
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

public class RegularFPTest {

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        extractObjectsInDir("src/main/resources/image/horse1/");
//        testRegularFP("src/main/resources/image/horse1/", "Motorcycle_s1.00.JPG");
        orb = ORB.create(500, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 20);
//        testRegularFP("src/main/resources/image/motorcycle1/", "000.JPG");
//        testRegularFP("src/main/resources/image/horse1/", "000.JPG");
//        testMaxMin("src/main/resources/image/motorcycle1/", "000.JPG");
//        testMaxMin("src/main/resources/image/toy_car/", "000.png");
//        testMaxMin("src/main/resources/image/horse1/", "000.JPG");
//        scaleDownImage("src/main/resources/image/horse1/000.JPG");
        String[] dirNames = {"lego_man", "shoe"};//, "furry_elephant", "toy_bear", "van_gogh", "furry_bear", "duck_cup"};
        for (String dir : dirNames) {
            scaleDownImage("src/main/resources/image/"+dir+"/0.png");
        }
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
                Features2d.drawMatches(qImg, rQIF.getObjectKeypoints(), tImg, rTIF.getObjectKeypoints(), m, display);
                ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));

                System.out.print("| ");
                for (int i = 0; i < mL.size(); i++) {
                    System.out.printf("%d ", mL.get(i).trainIdx);
//                    System.out.printf("%d\t", mL.get(i).trainIdx);
//                    if ((i+1)%20==0) System.out.println();
                }
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

    static void extractObjectsInDir(String filePath) throws IOException {
        ObjectDetector objectDetector = new ObjectDetector();
        objectDetector.init();
        File dir = new File(filePath);
        File[] directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (File f : directoryListing) {
                if (f.isFile()) {
                    BufferedImage image = ImageIO.read(f);
                    List<Recognition> recognitions = objectDetector.recognizeImage(image);
                    int i = 0;
                    for (Recognition recognition : recognitions) {
                        System.out.printf("%s, Object: %s - confidence: %f box: %s\n", f.getName(),
                                recognition.getTitle(), recognition.getConfidence(), recognition.getLocation());
//                        ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(recognition.cropPixels(ImageUtil.BufferedImage2Mat(image))));
                        ImageUtil.saveImage(ImageUtil.Mat2BufferedImage(recognition.cropPixels(ImageUtil.BufferedImage2Mat(image))), f.getAbsolutePath());
                        //the recognition list is sorted by the confidence, so only extract the first recognition
                        break;

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
        List<List<Integer>> fpTrack = analyzeFPsInImages(tIF, testImages);
        List<Integer> sizes = fpTrack.stream().map(o->o.size()).collect(Collectors.toList());
        int num = 100;
        for (int i : sizes) {
           if (i < num)
               num = i/10*10;
        }
        Pair<Integer, List<Integer>> candidates = maxMin(fpTrack, num);
        System.out.printf("original template num: %d, min: %d, %d candidates:%s\n",tIF.getSize(), candidates.getKey(), candidates.getValue().size(), candidates.getValue());

        List<KeyPoint> tKP = tIF.getObjectKeypoints().toList();
        List<KeyPoint> kps = new ArrayList<>();
        for (int i : candidates.getValue()) {
            kps.add(tKP.get(i));
        }
        Mat des = new Mat();
        MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
        matOfKeyPoint.fromList(kps);
        orb.compute(tImg, matOfKeyPoint, des);
        ImageFeature imageFeature = new ImageFeature(matOfKeyPoint, des);

//        for (Mat qImg : testImages) {
        File dir = new File(filePath); File[] directoryListing = dir.listFiles();
        if (directoryListing != null) {
            List<File> files = new ArrayList<>(Arrays.asList(directoryListing));
            files.sort(Comparator.comparing(File::getName));
            for (File f : files) {
                if (f.getName().equals(templateImg))
                    continue;
                Mat qImg = ImageUtil.BufferedImage2Mat(ImageIO.read(f));
                ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, 500);
                List<DMatch> matches = matchImage(qIF, imageFeature);
                MatOfDMatch m = new MatOfDMatch();
                m.fromList(matches);
//            System.out.printf("%f\n", (float) matches.size() / imageFeature.getSize());
                System.out.printf("%s: %f\n", f.getName(), (float) matches.size() / imageFeature.getSize());
                Mat display = new Mat();
                Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, imageFeature.getObjectKeypoints(), m, display);
                ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));
            }
        }
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
