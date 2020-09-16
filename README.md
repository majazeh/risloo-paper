# Multiple choice scanner
This is a project for scanning a multiple choice bubble sheet. 
## Demo Application
To run the Demo file, follow the below commands:

1. Install Python >=3.7 
2. Install required libraries with this command `pip install -r ./requirements.txt`  
3. Run `risloo_paper.py --image_files [path of image files of your test]`. 
   * For example  for a test with a single page :  `risloo_paper.py --image_files ./test1.jpg `
   * For a test with multiple pages : `risloo_paper.py --image_files ./test1_page1.jpg ./test1_page2.png ./test1_page3.jpg`
4. The edited images will be saved in the `./Adjusted_tests` directory 
4. The final .Json files  will be saved in the `./Answers` directory  

* If any problem occurs, The corresponding error will be printed and you should correct it!
### All possible errors 
1. `This directory was not found!` : The folder which you have passed to the `--image_folder` arguement doesn't exits.
2. `You didn't enter any argument !` : It's clear!
3. `This is not a file : [file] !` : The path of [file] you passed in the argument is not a file ; Probably It's a directory  
4. `This is not an image file : [file] !` : The path of [file] you passed in the argument is not an image file (it doesn't end with .jpg or .png).  
5. `4 points of sorrounding contour can't be found in [file], You should take a better picture of paper !` : The image [file] you have scanned doesn't capture all surrounding rectangle of the paper. 
6. `QR code detected but didn't read its information in [file] ,You should take a better picture of paper !` : The [file] is not clear enough to read QRcode.
7. `No QR code detected in [file] ,You should take a better picture of paper !` : The [file] is not clear enough to detect QRcode.
8. `Please Scan Again! markers not found completely in [file]!` : The [file] is not clear enough to detect markers!
9. `Please Scan Again! choices not found completely in [file] !` : The [file] is not clear enough to detect options contours!

