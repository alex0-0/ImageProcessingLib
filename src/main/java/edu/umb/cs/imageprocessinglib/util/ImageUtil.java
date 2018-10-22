package edu.umb.cs.imageprocessinglib.util;


import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import org.apache.commons.io.IOUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Util class for image processing.
 */
public class ImageUtil {
    private static ImageUtil imageUtil;
    // Output directory
    String OUTPUT_DIR = "./sample";

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
    public void labelAndSaveImage(final byte[] image, final List<Recognition> recognitions, final String fileName, float modelInSize) {
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
        saveImage(bufferedImage, OUTPUT_DIR + "/" + fileName);
    }

    public void saveImage(final BufferedImage image, final String target) {
        try {
            ImageIO.write(image,"jpg", new File(target));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private BufferedImage createImageFromBytes(final byte[] imageData) {
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
}
