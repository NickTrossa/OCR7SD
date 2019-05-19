# OCR7SD
## Optical Character Recognition of Seven Segment Display (Python based)

This is a command-line guided Python script to recognize the numbers in a seven segment digits display. Currently, there are two modes of operation:
1. Load pictures.
2. Aquire pictures using webcam.

OCRmain.py loads functions from OCRauxiliar.py and needs OpenCV module installed to run properly (appart from numpy and matplotlib). It displays the results in the prompt.

### Mode 1 (example.py)
The macro-structure of the code in Mode 1 is the following:
1. Load a photo or picture with the display.
2. Selection of ROI, manually input number of digits and segmentation.
3. Resize of base digits (/img/numeros_base.png) to the shape of the yet unknown digits of the ROI.
4. Load picture and application of method to each segmented digit:
    - Pixel-by-pixel comparation of the binarized digits to the "base" (images of each of the possible digits displayed)
5. Print result as an array where the last column contains the recognized numbers. A second array is printed, which contains the "distance" of the recognized digit to the next possible result. High values correlate with high confidence and lower ones with low confidence.
6. Loop items 4 and 5 using configuration parameters from items 2 and 3.

### Mode 2
The macro-structure of the code in Mode 2 is the following:
1. Substract backround from pictures before binarization.
2. Manually input number of digits and selection of ROI.
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
5. Load base "numeros_base.png" (picture with the ten possible digits) located in /img.
6. Resize base digits to the shape of the yet unknown digits of the ROI.
7. Transform ROI to a rectangular area in order to "untilt" digits, so they become easier to analyze.
8. Loop:
    - Capture image, substract backround of item 1), binarize and separate digits according to item 4).
    - Apply methods.
