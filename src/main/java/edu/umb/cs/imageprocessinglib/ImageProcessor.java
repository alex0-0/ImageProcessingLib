package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.feature.FeatureDetector;
import edu.umb.cs.imageprocessinglib.feature.FeatureMatcher;
import edu.umb.cs.imageprocessinglib.model.DescriptorType;
import edu.umb.cs.imageprocessinglib.model.ImageFeature;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.imgcodecs.Imgcodecs;

import java.awt.image.BufferedImage;

public class ImageProcessor {

    static public ImageFeature extractDistinctFeatures(BufferedImage img) {
        Mat mat = ImageUtil.BufferedImage2Mat(img);
        return extractDistinctFeatures(mat);
    }

    static public ImageFeature extractDistinctFeatures(Mat img, int num) {
        return extractDistinctFeatures(img, num, DescriptorType.ORB);
    }

    static public ImageFeature extractDistinctFeatures(Mat img, int num, DescriptorType type) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractDistinctFeatures(img, kps, des, type, num);
        return new ImageFeature(kps, des, type);
    }

    /*
    Extract image feature points
     */
    static public ImageFeature extractDistinctFeatures(Mat img) {
        MatOfKeyPoint kps = new MatOfKeyPoint();
        Mat des = new Mat();
        FeatureDetector.getInstance().extractDistinctFeatures(img, kps, des);
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
    static public MatOfDMatch matcheImages(ImageFeature qIF, ImageFeature tIF) {
        if (qIF.getDescriptorType() != tIF.getDescriptorType()) {
            System.out.printf("Can't match different feature descriptor types");
            return null;
        }
        return FeatureMatcher.getInstance().matchFeature(qIF.getDescriptors(), tIF.getDescriptors(),
                qIF.getObjectKeypoints(), tIF.getObjectKeypoints(), qIF.getDescriptorType());
//        return FeatureMatcher.getInstance().BFMatchFeature(qIF.getDescriptors(), tIF.getDescriptors());
    }

    static public MatOfDMatch myMatcheImages(ImageFeature qIF, ImageFeature tIF, SimpleRegression rx, SimpleRegression ry) {
        if (qIF.getDescriptorType() != tIF.getDescriptorType()) {
            System.out.printf("Can't match different feature descriptor types");
            return null;
        }
        return FeatureMatcher.getInstance().myMatchFeature(qIF.getDescriptors(), tIF.getDescriptors(),
                qIF.getObjectKeypoints(), tIF.getObjectKeypoints(), qIF.getDescriptorType(), rx, ry);
//        return FeatureMatcher.getInstance().BFMatchFeature(qIF.getDescriptors(), tIF.getDescriptors());
    }

    static public MatOfDMatch matcheImages(Mat queryImg, Mat temImg) {
        ImageFeature qIF = extractFeatures(queryImg);
        ImageFeature tIF = extractFeatures(temImg);
        return matcheImages(qIF, tIF);
    }

    /*
    Match two images
     */
    static public MatOfDMatch BFMatcheImages(ImageFeature qIF, ImageFeature tIF) {
        if (qIF.getDescriptorType() != tIF.getDescriptorType()) {
            System.out.printf("Can't match different feature descriptor types");
            return null;
        }
        return FeatureMatcher.getInstance().BFMatchFeature(qIF.getDescriptors(), tIF.getDescriptors(), qIF.getDescriptorType());
    }

//    static public MatOfDMatch matcheImages(Bitmap queryImg, Bitmap temImg) {
//        feature qIF = extractFeatures(queryImg);
//        feature tIF = extractFeatures(temImg);
//        return matcheImages(qIF, tIF);
//    }

     static public KeyPoint findKeyPoint(ImageFeature templateF, int idx){
        return findKeyPoint(templateF.getObjectKeypoints(),idx);
    }

    static public KeyPoint findKeyPoint(MatOfKeyPoint mkp, int idx){
        KeyPoint[] kps=mkp.toArray();
        return kps[idx];
    }
}

