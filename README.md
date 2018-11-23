# OCR7SD
Optical Character Recognition of Seven Segment Display (Python based)

This is a command-line guided Python script to automatically aquire pictures from a webcam pointing at a seven segment display and return the number displayed.

The macro-structure of the code is the following:
1) Substract backround from pictures before binarization.
3) Manually input number of digits and selecting ROI.
4) Application of three quantisized methods:
  i. Pixel-by-pixel comparation of the binarized digits to the "base" (images of each of the possible digits displayed)
  ii. Evaluation of state of individual segments by measuring on small, specific areas.
  iii. Comparation of black-covered fraction of the digit to the ones in the base (to discard options)
5) Loop the process and save to file.

The more-detailed-structure is:
1) Capture backround (display OFF) to improve quality of segmentation process and convert to gray.
2) Capture display ON picture and convert to gray.
3) Substract backround of first display ON picture.
4) Manually select ROI (Region Of Interest, where the numbers are displayed) and input of the number of digits.
5) Load base (picture with the ten possible digits)
6) Resize base digits to the shape of the yet unknown digits of the ROI.
7) Loop:
  Capture image, substract backround of item 1), binarize and separate digits according to item 4).
  Apply methods and choose the best result (the one more confident).
  Save and continue
