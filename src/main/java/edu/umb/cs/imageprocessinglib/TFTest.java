package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.opencv.core.Core;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TFTest {
    public static void main(String[] args) throws Exception {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        rotationTest();
        sizeTest();
    }

    static void rotationTest(){

    }

    static void sizeTest() throws IOException {
        String imgPath = "src/main/resources/image/dog_cat.jpg";
//        String imgPath = "src/main/resources/image/standing/140.jpg";
        BufferedImage img = ImageUtil.loadImage(imgPath);
        ObjectDetector objectDetector = new ObjectDetector();
        objectDetector.init();
        List<Recognition> recognitions = objectDetector.recognizeImage(img);

        for (Recognition r : recognitions) {
            System.out.printf("Object: %s - confidence: %f box: %s\n",
                    r.getTitle(), r.getConfidence(), r.getLocation());
            BoxPosition bp = r.getScaledLocation((float)img.getWidth()/r.getModelSize(), (float)img.getHeight()/r.getModelSize());
            BufferedImage t = img.getSubimage(bp.getLeftInt(), bp.getTopInt(), bp.getWidthInt(), bp.getHeightInt());
            ImageUtil.displayImage(t);
        }

        List<BufferedImage> images = divideImage(img, 3, 3);
        for (BufferedImage i : images) {
            List<Recognition> rs = objectDetector.recognizeImage(i);

            for (Recognition r : rs) {
                System.out.printf("Object: %s - confidence: %f box: %s\n",
                        r.getTitle(), r.getConfidence(), r.getLocation());
                BoxPosition bp = r.getScaledLocation((float)i.getWidth()/r.getModelSize(), (float)i.getHeight()/r.getModelSize());
                BufferedImage t = i.getSubimage(bp.getLeftInt(), bp.getTopInt(), bp.getWidthInt(), bp.getHeightInt());
                ImageUtil.displayImage(t);
            }

        }

    }

    /**
     *
     * @param img   image
     * @param hn    horizontal division number
     * @param vn    vertical division number
     * @return
     */
    static List<BufferedImage> divideImage(BufferedImage img, int hn, int vn) {
        List<BufferedImage> imgs = new ArrayList();
        int w = img.getWidth()/hn;
        int h = img.getHeight()/vn;
        for (int i=0; i < hn; i++) {
            for (int d = 0; d < vn; d++) {
                BufferedImage t = img.getSubimage(i * w, d * h, w, h);
                imgs.add(t);
            }
        }
        for (int i=0; i < hn-1; i++) {
            for (int d = 0; d < vn-1; d++) {
                BufferedImage t = img.getSubimage(i * w + w/2, d * h + h/2, w, h);
                imgs.add(t);
            }
        }

        List<BufferedImage> r = new ArrayList<>();
        //deep copy
        for (BufferedImage i : imgs) {
            BufferedImage deepCopy = new BufferedImage(i.getWidth(), i.getHeight(), i.getType());

        // Draw the subimage onto the new, empty copy
            Graphics2D g = deepCopy.createGraphics();
            try {
                g.drawImage(i, 0, 0, null);
            }
            finally {
                g.dispose();
            }
            r.add(deepCopy);
        }

        return r;
    }

}
