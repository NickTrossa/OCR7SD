# OCR7SD
## Optical Character Recognition of Seven Segment Display (Python based)

This is a command-line guided Python script to automatically aquire pictures from a webcam pointing at a seven segment display and return the number displayed.

OCRmain.py loads functions from OCRauxiliar.py and needs OpenCV module installed to run properly (appart from numpy and matplotlib). It saves the results in a "out.txt" file, where there are 2\*N + 1 columns. The first one is the time of the measurement, the next N are the N digits recognized using "method 1" and the other N using "method 2". The latter seems so work better.

The macro-structure of the code is the following:
1. Substract backround from pictures before binarization.
2. Manually input number of digits and selecting ROI.
3. Application of three quantisized methods:
    - Pixel-by-pixel comparation of the binarized digits to the "base" (images of each of the possible digits displayed)
    - Evaluation of state of individual segments by measuring on small, specific areas.
    - Comparation of black-covered fraction of the digit to the ones in the base (to discard options) **TO DO**
4. Loop the process and save to file.

The more detailed structure is:
1. Capture backround (display OFF) to improve quality of segmentation process and convert to gray.
2. Capture display ON picture and convert to gray.
3. Substract backround of first display ON picture.
4. Manually select ROI (Region Of Interest, where the numbers are displayed, that is, the 4 corners of the ROI plus two points marking the space between digits) and input of the number of digits.
5. Load base (picture with the ten possible digits)
6. Resize base digits to the shape of the yet unknown digits of the ROI.
7. Transform ROI to a rectangular area in order to "untilt" digits, so they become easier to analyze.
8. Loop:
    - Capture image, substract backround of item 1), binarize and separate digits according to item 4).
    - Apply methods and save results.
    - Save into file "out.txt" and continue.
