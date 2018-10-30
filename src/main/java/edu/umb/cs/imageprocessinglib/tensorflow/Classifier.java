package edu.umb.cs.imageprocessinglib.tensorflow;

import edu.umb.cs.imageprocessinglib.model.Recognition;

import java.awt.image.BufferedImage;
import java.util.List;

public interface Classifier {
    List<Recognition> recognizeImage(BufferedImage image);

//    void enableStatLogging(final boolean debug);
//
//    String getStatString();

//    void close();
}

