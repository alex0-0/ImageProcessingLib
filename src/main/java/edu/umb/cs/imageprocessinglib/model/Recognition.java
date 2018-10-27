package edu.umb.cs.imageprocessinglib.model;

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
    private BufferedImage img=null;

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

    public BufferedImage getPixels(){ return img;}

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

    public void loadPiexels(BufferedImage bufferedImage, int modelInSize){
        float scaleX = (float) bufferedImage.getWidth() / modelInSize;
        float scaleY = (float) bufferedImage.getHeight() / modelInSize;

        BoxPosition slocation=getScaledLocation(scaleX, scaleY);
        img=bufferedImage.getSubimage(slocation.getLeftInt(),slocation.getTopInt(),slocation.getWidthInt(),slocation.getHeightInt());
    }
}
