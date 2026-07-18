# Crosswalk Detection Using Classical Image Processing Techniques

## 1. Project Description

This project detects pedestrian crossings in road images using classical image processing techniques. The implementation is written in Python with OpenCV and does not use neural networks or trained models.

The topic is relevant to applied computer vision in the automotive field because it focuses on identifying an important element of road infrastructure: the pedestrian crossing.

## 2. Objective

The objective of the project is to automatically detect a pedestrian crossing in an image and highlight it in the original image.

The program must:

- read an input image;
- highlight white road markings;
- identify rectangular regions that may represent crosswalk stripes;
- group the detected stripes;
- display and save the final result.

## 3. Technologies Used

- Python
- OpenCV
- NumPy

## 4. Method

The algorithm uses a classical image-processing pipeline:

1. **Image Loading**
   - The image is loaded using `cv2.imread`.

2. **Grayscale Conversion**
   - The color image is converted to grayscale for simpler processing.

3. **Gaussian Blur**
   - A Gaussian filter is applied to reduce image noise.

4. **Thresholding**
   - Bright or white regions are extracted because pedestrian crossing markings are usually white.

5. **Morphological Operations**
   - Morphological closing and opening are applied to connect nearby regions and remove noise.

6. **Contour Detection**
   - Contours are detected in the binary mask.

7. **Candidate Stripe Filtering**
   - Only regions with an elongated shape and a sufficiently large area are kept.

8. **Crosswalk Area Estimation**
   - If at least three candidate stripes are detected, the image is considered to contain a pedestrian crossing.
   - A common bounding rectangle is created around the detected stripes.

## 5. Project Structure

```text
crosswalk_detection_project/
│
├── src/
│   └── main.py
│
├── input/
│   └── place test images here
│
├── output/
│   └── generated results are saved here
│
├── requirements.txt
└── README.md
```

## 6. Installation

Create a virtual environment, optionally:

```bash
python -m venv venv
```

Activate the virtual environment.

On Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 7. Running the Project

Place an image in the `input` directory, for example:

```text
input/crosswalk.jpg
```

Run the program:

```bash
python src/main.py --image input/crosswalk.jpg
```

You can also specify the output directory:

```bash
python src/main.py --image input/crosswalk.jpg --output output
```

## 8. Generated Results

For an image named `crosswalk.jpg`, the program generates:

```text
output/crosswalk_01_gray.jpg
output/crosswalk_02_threshold.jpg
output/crosswalk_03_cleaned_mask.jpg
output/crosswalk_04_result.jpg
```

The final image is:

```text
crosswalk_04_result.jpg
```

It contains:

- green rectangles around candidate stripes;
- a red rectangle around the final detected area;
- the text `Crosswalk detected` when the detection is successful.

## 9. Advantages

- Simple implementation that is easy to understand and explain.
- Does not require a large dataset or AI model training.
- Produces visual results that are easy to include in a presentation.
- Uses important image-processing concepts.

## 10. Limitations

The algorithm may encounter difficulties in the following situations:

- the pedestrian crossing is heavily faded;
- the image contains strong shadows;
- the markings are covered by vehicles or pedestrians;
- the camera angle is significantly different;
- other similar white road markings are present in the image.

## 11. Possible Improvements

- Use the HSV color space for more accurate white-color segmentation.
- Apply the Hough transform to detect parallel lines.
- Add a region of interest to analyze only the road area.
- Process video frame by frame.
- Integrate an object detection model for more robust results.

## 12. Conclusion

This project demonstrates how classical computer vision techniques can be used to detect pedestrian crossings. Although the method does not use artificial intelligence, it can produce good results on clear images and provides a useful foundation for driver-assistance systems or smart-city applications.
