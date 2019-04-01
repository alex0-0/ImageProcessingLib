package edu.umb.cs.imageprocessinglib.util;


import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import org.apache.commons.io.IOUtils;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;

/**
 * Util class for image processing.
 */
public class ImageUtil {
    private static ImageUtil imageUtil;
    // Default output directory
    static String OUTPUT_DIR = "./sample";

    private ImageUtil() {
        File dir = new File(OUTPUT_DIR);
        if (!dir.exists()) {
            dir.mkdir();
        }
    }

    /**
     * It returns the singleton instance of this class.
     * @return ImageUtil instance
     */
    public static ImageUtil getInstance() {
        if (imageUtil == null) {
            imageUtil = new ImageUtil();
        }

        return imageUtil;
    }

    /**
     * Label image with classes and predictions given by the ThensorFLow
     * @param image buffered image to label
     * @param recognitions list of recognized objects
     */
    public static void labelAndSaveImage(final byte[] image, final List<Recognition> recognitions, final String fileName, float modelInSize) {
        labelAndSaveImage(image, recognitions, fileName, modelInSize, OUTPUT_DIR);
    }

    /**
     * Label image with classes and predictions given by the ThensorFLow
     * @param image buffered image to label
     * @param recognitions list of recognized objects
     */
    public static void labelAndSaveImage(final byte[] image, final List<Recognition> recognitions, final String fileName, float modelInSize, final String dirPath) {
        BufferedImage bufferedImage = imageUtil.createImageFromBytes(image);
        float scaleX = (float) bufferedImage.getWidth() / modelInSize;
        float scaleY = (float) bufferedImage.getHeight() / modelInSize;
        Graphics2D graphics = (Graphics2D) bufferedImage.getGraphics();

        for (Recognition recognition: recognitions) {
            BoxPosition box = recognition.getScaledLocation(scaleX, scaleY);
            //draw text
            graphics.drawString(recognition.getTitle() + " " + recognition.getConfidence(), box.getLeft(), box.getTop() - 7);
            // draw bounding box
            graphics.drawRect(box.getLeftInt(),box.getTopInt(), box.getWidthInt(), box.getHeightInt());
        }

        graphics.dispose();

        File dir = new File(dirPath);
        if (!dir.exists()) {
            dir.mkdir();
        }
        saveImage(bufferedImage, dirPath + "/" + fileName);
    }

    public static void saveImage(final BufferedImage image, final String target) {
        try {
            ImageIO.write(image,"jpg", new File(target));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static BufferedImage loadImage(final String filePath) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(filePath));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return img;
    }

    static public Mat loadMatImage(String file){
        Imgcodecs imageCodecs = new Imgcodecs();
        return imageCodecs.imread(file);
    }

    public static BufferedImage deepCopyBufferedImage(BufferedImage i) {
        BufferedImage deepCopy = new BufferedImage(i.getWidth(), i.getHeight(), i.getType());
        // Draw the subimage onto the new, empty copy
        Graphics2D g = deepCopy.createGraphics();
        try {
            g.drawImage(i, 0, 0, null);
        }
        finally {
            g.dispose();
        }
        return deepCopy;
    }

    public static BufferedImage createImageFromBytes(final byte[] imageData) {
        ByteArrayInputStream bais = new ByteArrayInputStream(imageData);
        try {
            return ImageIO.read(bais);
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
    }

    public static byte[] extractBytes (final String filePath, Class c) {
        try {
            File f = new File(filePath);
            return Files.readAllBytes(f.toPath());
        } catch (IOException | NullPointerException ex) {
            ex.printStackTrace();
            throw new RuntimeException();
        }
    }

    public static BufferedImage Mat2BufferedImage(Mat m) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 ) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels()*m.cols()*m.rows();
        byte [] b = new byte[bufferSize];
        m.get(0,0,b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    public static Mat BufferedImage2Mat(BufferedImage bi) {
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
        byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        mat.put(0, 0, data);
        return mat;
    }

    //display image
    public static void displayImage(BufferedImage img) {
        Toolkit toolkit =  Toolkit.getDefaultToolkit ();
        Dimension dim = toolkit.getScreenSize();
        Image image = img;
        if (img.getHeight() > dim.height || img.getWidth() > dim.width) {
            float scale = Math.min((float)dim.height/img.getHeight(), (float)dim.width/img.getWidth());
            image = img.getScaledInstance((int)(img.getWidth()*scale), (int)(img.getHeight()*scale),
                    Image.SCALE_SMOOTH);
        }
        ImageIcon icon=new ImageIcon(image);
        JFrame frame=new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(image.getWidth(null)+50, image.getHeight(null)+50);
        JLabel lbl=new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static void BGR2RGB(byte[] data) {
        for (int i = 0; i < data.length; i += 3) {
            byte tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }
    }

    /**
     * Rotate original image to generate a group of distorted image
     * @param image     original image
     * @return          a list containing rotated images
     */
    public static Mat rotateImage(Mat image, float angle) {
        Mat rotatedImg = new Mat();
        Size size = image.size();
        Point center = new Point(size.width/2, size.height/2);
        Mat matrix = Imgproc.getRotationMatrix2D(center, angle, 1);
        Imgproc.warpAffine(image, rotatedImg, matrix, size);
        matrix.release();
        return rotatedImg;
    }

    /**
     * Scale original image to generate a group of distorted image
     * @param image     original image
     * @param scale     scale
     * @return          a list containing scaled images
     */
    public static Mat scaleImage(Mat image, float scale) {
        Mat scaledImage = new Mat();
        Size size = image.size();
        double rows = size.width;
        double cols = size.height;
        Size newSize = new Size(rows * scale, cols * scale);
        Imgproc.resize(image, scaledImage, newSize);
        return scaledImage;
    }

    /**
     * Adjust original image bright. p(i,j) = α⋅p(i,j)+β
     * @param image     original image
     * @param alpha     alpha
     * @param beta      beta
     * @return          a list containing scaled images
     */
    public static Mat lightImage(Mat image, float alpha, int beta) {
        Mat newImage = new Mat();
        image.convertTo(newImage, -1, alpha, beta);
        return newImage;
    }

    /**
     * Affine original image to generate a group of distorted image
     * refer to https://stackoverflow.com/questions/10962228/whats-the-best-way-of-understanding-opencvs-warpperspective-and-warpaffine?rq=1 for more information
     * @param image     original image
     * @param originalPoints     original points position, at least 4 points, containing left-top, right-top, right-bottom, left-bottom point
     * @param targetPoints       target points position, at least 4 points, containing left-top, right-top, right-bottom, left-bottom point
     * @return          a image of changed perspective.
     */
    public static Mat changeImagePerspective(Mat image, List<Point> originalPoints, List<Point> targetPoints) {
        Mat r = new Mat();
        Mat cornersMat = Converters.vector_Point2f_to_Mat(originalPoints);
        Mat targetMat = Converters.vector_Point2f_to_Mat(targetPoints);
        Mat trans = Imgproc.getPerspectiveTransform(cornersMat, targetMat);

        Imgproc.warpPerspective(image, r, trans, new Size(image.cols(), image.rows()));
        //clean resource
        cornersMat.release();
        targetMat.release();
        trans.release();

        return r;
    }

    /**
     * Affine original image to generate a group of distorted image
     * @param image     original image
     * @param originalPoints     original points position, at least 4 points
     * @param targetPoints       target points position, at least 4 points
     * @return          an affined image
     */
    public static Mat affineImage(Mat image, List<Point> originalPoints, List<Point> targetPoints) {
        MatOfPoint2f originalMat = new MatOfPoint2f();
        originalMat.fromList(originalPoints);

        MatOfPoint2f targetMat = new MatOfPoint2f();
        targetMat.fromList(targetPoints);

        //calculate the affine transformation matrix,
        //refer to https://stackoverflow.com/questions/22954239/given-three-points-compute-affine-transformation
        Mat affineTransform = Imgproc.getAffineTransform(originalMat, targetMat);

        Mat affine = new Mat();
        Imgproc.warpAffine(image, affine, affineTransform, new Size(image.cols(), image.rows()));

        //release resources
        affineTransform.release();
        targetMat.release();
        originalMat.release();

        return affine;
    }
}
