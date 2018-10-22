package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.model.Recognition;

import java.io.IOException;
import java.util.List;

public class Main {
//    private final static String IMAGE = "/image/cow-and-bird.jpg";
    private final static String IMAGE = "/image/eagle.jpg";

    public static void main(String[] args) throws IOException {
        ImageProcessor imageProcessor = new ImageProcessor();
        imageProcessor.initObjectDetector();
        List<Recognition> recognitions = imageProcessor.recognizeImage(IMAGE);
        printToConsole(recognitions);
    }
    /**
     * Prints out the recognize objects and its confidence
     * @param recognitions list of recognitions
     */
    private static void printToConsole(final List<Recognition> recognitions) {
        for (Recognition recognition : recognitions) {
            System.out.printf("Object: %s - confidence: %f", recognition.getTitle(), recognition.getConfidence());
        }
    }
}
