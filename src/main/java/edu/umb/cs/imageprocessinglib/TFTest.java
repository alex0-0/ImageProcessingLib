package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.opencv.core.Core;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

public class TFTest {
    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
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

    }

}
