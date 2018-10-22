package edu.umb.cs.imageprocessinglib.tensorflow;

import edu.umb.cs.imageprocessinglib.model.Recognition;

import java.util.List;

public interface Classifier {
    List<Recognition> recognizeImage(byte[] image);

//    void enableStatLogging(final boolean debug);
//
//    String getStatString();

//    void close();
}

