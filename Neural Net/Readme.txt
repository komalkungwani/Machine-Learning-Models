The main.py file contains the code that uses neural network along with tensorflow that works on mnist dataset to predict the number it looks at (0,1,2,3,4,5,6,7,8, or 9).
Mnist dataset contains 60,000 training samples and 10,000 testing samples of hand-written and labeled digits, 0 through 9.
It has the images, which are purely black and white, thresholded, images, of size 28 x 28, or 784 pixels total.
Features will be the pixel values for each pixel, thresholded. Either the pixel is "blank" (nothing there, a 0), or there is something there (1). Those are the features.
Neural network will create an inner-model of the relationships between pixels, and be able to look at new examples of digits and predict them to a high degree.
