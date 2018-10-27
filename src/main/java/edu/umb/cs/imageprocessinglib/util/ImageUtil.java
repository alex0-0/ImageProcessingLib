package edu.umb.cs.imageprocessinglib.util;


import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import org.apache.commons.io.IOUtils;
import org.opencv.core.Mat;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
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

    public static BufferedImage createImageFromBytes(final byte[] imageData) {
        ByteArrayInputStream bais = new ByteArrayInputStream(imageData);
        try {
            return ImageIO.read(bais);
        } catch (IOException ex) {
            throw new ServiceException("Unable to create image from bytes!", ex);
        }
    }

    public static byte[] extractBytes (final String filePath, Class c) {
        try {
            return IOUtils.toByteArray(c.getResourceAsStream(filePath));
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

    //display image
    public static void displayImage(Image img) {
        ImageIcon icon=new ImageIcon(img);
        JFrame frame=new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(img.getWidth(null)+50, img.getHeight(null)+50);
        JLabel lbl=new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
    
}
