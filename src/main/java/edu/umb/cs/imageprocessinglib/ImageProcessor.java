package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.feature.FeatureDetector;
import edu.umb.cs.imageprocessinglib.feature.FeatureMatcher;
import edu.umb.cs.imageprocessinglib.model.DescriptorType;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.opencv.core.*;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class ImageProcessor {

    static public List<Mat> rotatedImage(Mat image, float stepAngle, int num) {
        List<Mat> r = new ArrayList<>();
        for (int i = 1; i <= num; i++) {
            r.add(ImageUtil.rotateImage(image, (1 + i * stepAngle)));
        }
        return r;
    }

    static public List<Mat> scaleImage(Mat image, float stepScale, int num) {
        List<Mat> r = new ArrayList<>();
        for (int i = 1; i <= num; i++) {
            r.add(ImageUtil.scaleImage(image, (1 + i * stepScale)));
        }
        return r;
    }

    static public List<Mat> lightImage(Mat image, float stepLight, int num) {
        List<Mat> r = new ArrayList<>();
        for (int i = 1; i <= num; i++) {
            r.add(ImageUtil.lightImage(image, 1 + stepLight * i, 0));
        }
        return r;
    }

    //WARNING: stepPer * num should be less than Min(image.width, image.height) and larger than 0
    static public List<Mat> changeToRightPerspective(Mat image, float stepPer, int num) {
        List<Mat> r = new ArrayList<>();

        List<Point> originals = new ArrayList<>();
        originals.add(new Point(0, 0));
        originals.add(new Point(image.cols(), 0));
        originals.add(new Point(image.cols(), image.rows()));
        originals.add(new Point(0, image.rows()));
        int pixelStep = Math.min(image.rows(), image.cols());
        for (int i = 1; i <= num; i++) {
            List<Point> corners = new ArrayList<>();
            corners.add(new Point(stepPer*i, stepPer*i));
            corners.add(new Point(image.cols(), 0));
            corners.add(new Point(image.cols(), image.rows()));
            corners.add(new Point(stepPer*i, image.rows()-stepPer*i));

            r.add(ImageUtil.changeImagePerspective(image, originals, corners));
        }
        return r;
    }

    //WARNING: stepPer * num should be less than Min(image.width, image.height) and larger than 0
    static public List<Mat> changeToLeftPerspective(Mat image, float stepPer, int num) {
        List<Mat> r = new ArrayList<>();

        List<Point> originals = new ArrayList<>();
        originals.add(new Point(0, 0));
        originals.add(new Point(image.cols(), 0));
        originals.add(new Point(image.cols(), image.rows()));
        originals.add(new Point(0, image.rows()));
        for (int i = 1; i <= num; i++) {
            List<Point> corners = new ArrayList<>();
            corners.add(new Point(0, 0));
            corners.add(new Point(image.cols()-i*stepPer, i*stepPer));
            corners.add(new Point(image.cols()-i*stepPer, image.rows()-i*stepPer));
            corners.add(new Point(0, image.rows()));

            r.add(ImageUtil.changeImagePerspective(image, originals, corners));
        }
        return r;
    }

    //WARNING: stepPer * num should be less than Min(image.width, image.height) and larger than 0
    static public List<Mat> changeToTopPerspective(Mat image, float stepPer, int num) {
        List<Mat> r = new ArrayList<>();

        List<Point> originals = new ArrayList<>();
        originals.add(new Point(0, 0));
        originals.add(new Point(image.cols(), 0));
        originals.add(new Point(image.cols(), image.rows()));
        originals.add(new Point(0, image.rows()));
        for (int i = 1; i <= num; i++) {
            List<Point> corners = new ArrayList<>();
            corners.add(new Point(0, 0));
            corners.add(new Point(image.cols(), 0));
            corners.add(new Point(image.cols()-stepPer*i, image.rows()-stepPer*i));
            corners.add(new Point(stepPer*i, image.rows()-stepPer*i));

            r.add(ImageUtil.changeImagePerspective(image, originals, corners));
        }
        return r;
    }

    //WARNING: stepPer * num should be less than Min(image.width, image.height) and larger than 0
    static public List<Mat> changeToBottomPerspective(Mat image, float stepPer, int num) {
        List<Mat> r = new ArrayList<>();

        List<Point> originals = new ArrayList<>();
        originals.add(new Point(0, 0));
        originals.add(new Point(image.cols(), 0));
        originals.add(new Point(image.cols(), image.rows()));
        originals.add(new Point(0, image.rows()));
        for (int i = 1; i <= num; i++) {
            List<Point> corners = new ArrayList<>();
            corners.add(new Point(stepPer*i, stepPer*i));
            corners.add(new Point(image.cols()-i*stepPer, stepPer*i));
            corners.add(new Point(image.cols(), image.rows()));
            corners.add(new Point(0, image.rows()));

            r.add(ImageUtil.changeImagePerspective(image, originals, corners));
        }
        return r;
    }

    static public ImageFeature extractRobustFeatures(BufferedImage img) {
        Mat mat = ImageUtil.BufferedImage2Mat(img);
        return extractRobustFeatures(mat);
    }

    static public ImageFeature extractRobustFeatures(Mat img, int num) {
        return extractRobustFeatures(img, num, DescriptorType.ORB);
    }

    static public ImageFeature extractRobustFeatures(Mat img, int num, DescriptorType type) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
//        FeatureDetector featureDetector = new FeatureDetector(num);
//        featureDetector.extractRobustFeatures(img, kps, des, type, num);
        FeatureDetector.getInstance().extractRobustFeatures(img, kps, des, type, num);
        return new ImageFeature(kps, des, type);
    }

    /*
    Extract image feature points
     */
    static public ImageFeature extractRobustFeatures(Mat img, List<Mat> distortedImg, int num, DescriptorType type) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
//        FeatureDetector fd = new FeatureDetector(num);
        FeatureDetector.getInstance().extractRobustFeatures(img, distortedImg, kps, des, type, num);
        return new ImageFeature(kps, des, type);
    }

    /*
    Extract image feature points
     */
    static public ImageFeature extractRobustFeatures(Mat img) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractRobustFeatures(img, kps, des, DescriptorType.ORB);
        return new ImageFeature(kps, des);
    }

    /*
    Extract image feature points
     */
    static public ImageFeature extractORBFeatures(Mat img) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractORBFeatures(img, kps, des);
        return new ImageFeature(kps, des, DescriptorType.ORB);
    }

    /*
    Extract image feature points with ORB detector, the bound of the number of feature points is num
     */
    static public ImageFeature extractORBFeatures(Mat img, int num) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector fd = new FeatureDetector(num);
        fd.extractORBFeatures(img, kps, des);
        return new ImageFeature(kps, des, DescriptorType.ORB);
    }

    /*
    Extract image feature points
     */
    static public ImageFeature extractFeatures(Mat img) {
        return extractORBFeatures(img);
    }

    /*
    Extract image feature points
     */
    static public ImageFeature extractSURFFeatures(Mat img) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractSurfFeatures(img, kps, des);
        return new ImageFeature(kps, des, DescriptorType.SURF);
    }

    /*
    Match two images
     */
    static public MatOfDMatch matchImages(ImageFeature qIF, ImageFeature tIF) {
        if (qIF.getDescriptorType() != tIF.getDescriptorType()) {
            System.out.print("Can't match different feature descriptor types");
            return null;
        }
        return FeatureMatcher.getInstance().matchFeature(qIF.getDescriptors(), tIF.getDescriptors(),
                qIF.getObjectKeypoints(), tIF.getObjectKeypoints(), qIF.getDescriptorType());
//        return FeatureMatcher.getInstance().BFMatchFeature(qIF.getDescriptors(), tIF.getDescriptors());
    }

    static public MatOfDMatch myMatchImages(ImageFeature qIF, ImageFeature tIF, SimpleRegression rx, SimpleRegression ry) {
        if (qIF.getDescriptorType() != tIF.getDescriptorType()) {
            System.out.print("Can't match different feature descriptor types");
            return null;
        }
        return FeatureMatcher.getInstance().myMatchFeature(qIF.getDescriptors(), tIF.getDescriptors(),
                qIF.getObjectKeypoints(), tIF.getObjectKeypoints(), qIF.getDescriptorType(), rx, ry);
//        return FeatureMatcher.getInstance().BFMatchFeature(qIF.getDescriptors(), tIF.getDescriptors());
    }

    static public MatOfDMatch matchImages(Mat queryImg, Mat temImg) {
        ImageFeature qIF = extractFeatures(queryImg);
        ImageFeature tIF = extractFeatures(temImg);
        return matchImages(qIF, tIF);
    }

    /*
    Match two images
     */
    static public MatOfDMatch BFMatchImages(ImageFeature qIF, ImageFeature tIF) {
        if (qIF.getDescriptorType() != tIF.getDescriptorType()) {
            System.out.print("Can't match different feature descriptor types");
            return null;
        }
        return FeatureMatcher.getInstance().BFMatchFeature(qIF.getDescriptors(), tIF.getDescriptors(), qIF.getDescriptorType());
    }

     static public KeyPoint findKeyPoint(ImageFeature templateF, int idx){
        return findKeyPoint(templateF.getObjectKeypoints(),idx);
    }

    static public KeyPoint findKeyPoint(MatOfKeyPoint mkp, int idx){
        KeyPoint[] kps=mkp.toArray();
        return kps[idx];
    }

    static public MatOfDMatch matchWithRegression(ImageFeature qIF, ImageFeature tIF, int knnNum, float matchDisThd, int posThd) {
        if (qIF.getDescriptorType() != tIF.getDescriptorType()) {
            System.out.print("Can't match different feature descriptor types");
            return null;
        }
        return FeatureMatcher.getInstance().matchWithRegression(qIF.getDescriptors(), tIF.getDescriptors(),
                qIF.getObjectKeypoints(), tIF.getObjectKeypoints(), qIF.getDescriptorType(), knnNum, matchDisThd, posThd);
    }

    static public MatOfDMatch matchWithRegression(ImageFeature qIF, ImageFeature tIF) {
        return matchWithRegression(qIF, tIF, 5, 500, 20);
    }
}

