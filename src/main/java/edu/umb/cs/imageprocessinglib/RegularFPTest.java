package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.opencv.core.*;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class RegularFPTest {

    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        extractObjectsInDir("src/main/resources/image/horse/");
//        testRegularFP("src/main/resources/image/horse/", "000.JPG");
        orb  = ORB.create(500, 1.2f, 8, 15, 0, 2, ORB.HARRIS_SCORE, 31, 20);
        testRegularFP("src/main/resources/image/Motorcycle/", "000.JPG");
    }

    static void testRegularFP(String filePath, String templateImg) throws IOException {
        //hyperparameter for picking random feature points
        int num = 500;

        File tImgFile = new File(filePath+templateImg);
        if (tImgFile == null || !tImgFile.isFile())
            return;
        Mat tImg = ImageUtil.BufferedImage2Mat(ImageIO.read(tImgFile));
        ImageFeature tIF = ImageProcessor.extractORBFeatures(tImg,300);
        List<KeyPoint> tKP = tIF.getObjectKeypoints().toList();
//        System.out.printf("template key points number: %d\n", tIF.getSize());
        File dir = new File(filePath);
        File[] directoryListing = dir.listFiles();
        long totalTime = 0;
        if (directoryListing != null) {
            List<File> files = new ArrayList<>(Arrays.asList(directoryListing));
            files.sort(Comparator.comparing(File::getName));
            for (File f : files) {
                if (f.getName().equals(templateImg))
                    continue;
                Mat qImg = ImageUtil.BufferedImage2Mat(ImageIO.read(f));
//                ImageFeature qIF = ImageProcessor.extractFeatures(qImg);
                ImageFeature qIF = ImageProcessor.extractORBFeatures(qImg, 300);
                List<KeyPoint> qKP = qIF.getObjectKeypoints().toList();

                //randomly pick num feature points in template and query image
                ImageFeature rTIF = pickNRandomFP(tImg, tIF, num);
                ImageFeature rQIF = pickNRandomFP(qImg, qIF, 500);
//                ImageFeature rQIF = qIF;
                long startTime = System.currentTimeMillis();
                MatOfDMatch m = ImageProcessor.BFMatchImages(rQIF, rTIF);
                totalTime += (System.currentTimeMillis()-startTime);

//                MatOfDMatch m = ImageProcessor.BFMatchImages(qIF, tIF);
//                MatOfDMatch m = ImageProcessor.matchImages(qIF, tIF);
//                MatOfDMatch m = ImageProcessor.matchWithRegression(qIF, tIF);
                List<DMatch> mL = new ArrayList<>();
//                List<DMatch> mL = m.toList();
                Map<Integer, List<DMatch>> recorder = new HashMap<>();

                for (DMatch match : m.toList()) {
                    if (match.distance < 300) {
                        if (recorder.get(match.trainIdx) == null) {
                            recorder.put(match.trainIdx, new ArrayList<>());
                        }
                        recorder.get(match.trainIdx).add(match);
//                        mL.add(match);
                    }
                }
                //if multiple query points are matched to the same query point, keep the match with minimum distance
                for (Integer i : recorder.keySet()) {
                    DMatch minDisMatch = null;
                    float minDis = 999999999f;
                    for (DMatch dMatch : recorder.get(i)) {
                        if (dMatch.distance < minDis) {
                            minDisMatch = dMatch;
                            minDis = dMatch.distance;
                        }
                    }
                    if (minDisMatch != null)
                        mL.add(minDisMatch);
                }
                m = new MatOfDMatch();
                m.fromList(mL);

//                System.out.printf("%s Match number: %d, Precision: %f\n\n", f.getName(), m.total(), (float)m.total()/ tIF.getSize());
                System.out.printf("%s %dx%d, Match number: %d, Precision: %f\n", f.getName(), rTIF.getSize(), rQIF.getSize(), m.total(), (float)m.total()/ tIF.getSize());
                //display matches
                Mat display = new Mat();
//                    Features2d.drawMatches(qImg, qIF.getObjectKeypoints(), tImg, tIF.getObjectKeypoints(), m, display);
                Features2d.drawMatches(qImg, rQIF.getObjectKeypoints(), tImg, rTIF.getObjectKeypoints(), m, display);
//                ImageUtil.displayImage(ImageUtil.Mat2BufferedImage(display));

                //print matched key points
//                mL.sort((o1, o2) -> {
//                    return (int)(tKP.get(o1.trainIdx).pt.x - tKP.get(o2.trainIdx).pt.x);
//                });
//                for (int i = 0; i < mL.size(); i++) {
//                        DMatch match = mL.get(i);
//                        System.out.printf("t: (%.2f, %.2f), q: (%.2f, %.2f), dis: %.2f\n",
//                                tKP.get(match.trainIdx).pt.x, tKP.get(match.trainIdx).pt.y,
//                                qKP.get(match.queryIdx).pt.x, qKP.get(match.queryIdx).pt.y,
//                                match.distance);
//                }
                mL.sort((o1, o2) -> {
                    return (int)(o1.trainIdx - o2.trainIdx);
                });
                for (int i = 0; i < mL.size(); i++) {
                    System.out.printf("%d\t", mL.get(i).trainIdx);
                    if ((i+1)%20==0) System.out.println();
                }
                System.out.println("\n---------");
            }
        }
//        System.out.printf("average time:%f", (float)totalTime/(float)(directoryListing.length-1)/1000.0);
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
        MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();;
        matOfKeyPoint.fromList(kps);
        orb.compute(image, matOfKeyPoint, des);
        return new ImageFeature(matOfKeyPoint, des);
    }

    public static List<KeyPoint> pickNRandom(List<KeyPoint> lst, int n) {
        List<KeyPoint> copy = new LinkedList<KeyPoint>(lst);
        Collections.shuffle(copy);
        return copy.subList(0, n);
    }
}
