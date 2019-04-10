package edu.umb.cs.imageprocessinglib;

import edu.umb.cs.imageprocessinglib.model.BoxPosition;
import edu.umb.cs.imageprocessinglib.model.Recognition;
import edu.umb.cs.imageprocessinglib.util.ImageUtil;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.List;

public class TFTest {
    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        sizeTest();
        generateTestData(10);
    }


    static void generateTestData(int num) throws IOException {
        int ratioType = 5;  //five types of ratio for each row, i.e, 1:1:1, 2:1, 1:2, 2:3, 3:2
        int[] rowType = {1, 2, 3};  //three cases of the number of rows, i.e, 1, 2, 3
        float[] scaleType = {1, 1.5f, 2, 2.5f};  //three types of the scaling, i.e, 1, 1.5, 2, 2.5

        String path = "src/main/resources/image/";
//        String[] dirNames = {"lego_man", "shoe", "furry_elephant", "toy_bear", "van_gogh", "furry_bear", "duck_cup", "furry_dog", "baby_cream", "girl_statue"};
        String[] dirNames = {"toy_bear", "van_gogh", "furry_bear", "duck_cup", "girl_statue"};
        String[] groundTruth = {"teddy bear","person","teddy bear","cup","person"};

        String saveDir = "merged_img";
        File sdir = new File(path+saveDir);
        if (!sdir.exists())
            sdir.mkdir();

        File logFile = new File(path+saveDir+"/ground_truth");
        logFile.createNewFile();
        PrintWriter pw = new PrintWriter(new FileOutputStream(logFile, false));

        path += "single_distortion/";
        Mat[] mats = new Mat[dirNames.length];
        for (int i=0; i<dirNames.length; i++) {
            String dir = dirNames[i];
            mats[i] = ImageUtil.loadMatImage(path+dir+"/0.png");
        }

        Random rand = new Random();
        for (int i=0; i<num; i++) {
            float scale = scaleType[rand.nextInt(100)%scaleType.length];
            int rowNum = rowType[rand.nextInt(100)%rowType.length];

//            int height = (int)(mats[0].height()*scale);
            int width = (int)(mats[0].width()*scale);
            ArrayList<String> labels = new ArrayList<>();

            ArrayList<Mat> vImgs = new ArrayList<>();
            for (int k=0; k < rowNum; k++) {
                int r = rand.nextInt(100)%ratioType;
                int index;
                //five types of ratio for each row, i.e, 1:1:1, 2:1, 1:2, 2:3, 3:2
                switch (r) {
                    case 0://1:1:1
                    {
                        Mat resized = new Mat();
                        ArrayList<Mat> hImgs = new ArrayList<>();
                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        hImgs.add(mats[index]);
                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        hImgs.add(mats[index]);
                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        hImgs.add(mats[index]);
                        Core.hconcat(hImgs, resized);
                        Mat ret = new Mat();
                        Imgproc.resize(resized, ret, new Size(width, (float)width/resized.width()*resized.height()));
                        vImgs.add(ret);
                        break;
                    }
                    case 1://2:1
                    case 2://1:2
                    {
                        Mat resized = new Mat();
                        ArrayList<Mat> hImgs = new ArrayList<>();
                        //vertical 2 images
                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        hImgs.add(mats[index]);
                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        hImgs.add(mats[index]);
                        Mat t = new Mat();
                        Core.vconcat(hImgs, t);
                        //make sure the two images have the same height
                        Imgproc.resize(t, t, new Size(t.width()/2,t.height()/2));
                        hImgs.clear();

                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        if (r == 1) {
                            hImgs.add(t);
                            hImgs.add(mats[index]);
                        } else {
                            hImgs.add(mats[index]);
                            hImgs.add(t);
                        }
                        //horizontal two images
                        Core.hconcat(hImgs, resized);
                        Mat ret = new Mat();
                        Imgproc.resize(resized, ret, new Size(width, (float)width/resized.width()*resized.height()));
                        vImgs.add(ret);
                        break;
                    }
                    case 3://2:3
                    case 4://3:2
                    {
                        Mat resized = new Mat();
                        ArrayList<Mat> hImgs = new ArrayList<>();
                        //vertical 2 images
                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        hImgs.add(mats[index]);
                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        hImgs.add(mats[index]);
                        Mat t1 = new Mat();
                        Core.vconcat(hImgs, t1);
                        //make sure the two images have the same height
                        Imgproc.resize(t1, t1, new Size(t1.width()/2,t1.height()/2));
                        hImgs.clear();
                        hImgs = new ArrayList<>();
                        //vertical 3 images
                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        hImgs.add(mats[index]);
                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        hImgs.add(mats[index]);
                        index = rand.nextInt(100)%mats.length;
                        labels.add(groundTruth[index]);
                        hImgs.add(mats[index]);
                        Mat t2 = new Mat();
                        Core.vconcat(hImgs, t2);
                        //make sure the two images have the same height
                        Imgproc.resize(t2, t2, new Size(t2.width()/3,t2.height()/3));
                        hImgs.clear();
                        if (r == 3) {
                            hImgs.add(t1);
                            hImgs.add(t2);
                        } else {
                            hImgs.add(t2);
                            hImgs.add(t1);
                        }
                        //horizontal two images
                        Core.hconcat(hImgs, resized);
                        Mat ret = new Mat();
                        Imgproc.resize(resized, ret, new Size(width, (float)width/resized.width()*resized.height()));
                        vImgs.add(ret);
                        break;
                    }
                    default:
                        break;
                }
            }
            Mat ret = new Mat();
            Core.vconcat(vImgs, ret);

            Imgcodecs.imwrite(String.format("src/main/resources/image/%s/%d.png", saveDir, i), ret);
            pw.printf("%d\t%s\n",labels.size(), labels);
        }
        pw.close();
    }

    static void sizeTest() throws IOException {
//        String imgPath = "src/main/resources/image/dog_cat.jpg";
        String imgPath = "src/main/resources/image/single_distortion/furry_elephant/0.png";
//        String imgPath = "src/main/resources/image/standing/140.jpg";
//        BufferedImage img = ImageUtil.loadImage(imgPath);
        BufferedImage img = ImageUtil.Mat2BufferedImage(ImageUtil.rotateImage(ImageUtil.loadMatImage(imgPath), 90));
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

//        List<BufferedImage> images = divideImage(img, 3, 3);
//        for (BufferedImage i : images) {
//            List<Recognition> rs = objectDetector.recognizeImage(i);
//
//            for (Recognition r : rs) {
//                System.out.printf("Object: %s - confidence: %f box: %s\n",
//                        r.getTitle(), r.getConfidence(), r.getLocation());
//                BoxPosition bp = r.getScaledLocation((float)i.getWidth()/r.getModelSize(), (float)i.getHeight()/r.getModelSize());
//                BufferedImage t = i.getSubimage(bp.getLeftInt(), bp.getTopInt(), bp.getWidthInt(), bp.getHeightInt());
//                ImageUtil.displayImage(t);
//            }
//
//        }

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
