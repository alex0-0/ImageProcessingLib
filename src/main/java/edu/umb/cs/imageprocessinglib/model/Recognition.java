package edu.umb.cs.imageprocessinglib.model;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.awt.image.BufferedImage;

/**
 * An immutable result returned by a recognizer describing what was recognized.
 */
public final class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final Integer id;
    private final String title;
    private final Float confidence;
    private BoxPosition location;
    private Mat img=null;

    public Recognition(final Integer id, final String title,
                       final Float confidence, final BoxPosition location) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
    }

    public Integer getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    public BoxPosition getScaledLocation(final float scaleX, final float scaleY) {
        return new BoxPosition(location, scaleX, scaleY);
    }

    public Mat getPixels(){ return img;}

    public BoxPosition getLocation() {
        return new BoxPosition(location);
    }

    public void setLocation(BoxPosition location) {
        this.location = location;
    }

    @Override
    public String toString() {
        return "Recognition{" +
                "id=" + id +
                ", title='" + title + '\'' +
                ", confidence=" + confidence +
                ", location=" + location +
                '}';
    }

    public void loadPiexels(Mat oriImage, int modelInSize){
        float scaleX = (float) oriImage.size().width / modelInSize;
        float scaleY = (float) oriImage.size().height / modelInSize;

        BoxPosition slocation = getScaledLocation(scaleX, scaleY);
        Rect rect = new Rect(slocation.getLeftInt(), slocation.getTopInt(), slocation.getWidthInt(), slocation.getHeightInt());
//        img = oriImage.getSubimage(slocation.getLeftInt(),slocation.getTopInt(),slocation.getWidthInt(),slocation.getHeightInt());
        img = new Mat(oriImage, rect);
    }
}
