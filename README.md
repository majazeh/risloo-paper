# Multiple choice scanner
This is a project for scanning a multiple choice bubble sheet. 
## Demo Application
To run the Demo file, follow the below commands:

1. Install Python >=3.7 
2. Install required libraries with this command `pip install -r ./requirements.txt`  
2. Put all of your scanned sheet images in the `./Tests` directory or an another specific directory .
3. Run `Scanner.py --image_folder [The folder which your sheets are in that directory ("./ Tests" directory or your own directory)]`  
4. The edited images will be saved in the `./Adjusted_tests` directory 
4. The final .Json files  will be saved in the `./Answers` directory  

* If any problem occurs, The corresponding error will be printed and you should correct it!
### All possible errors 
1. `This directory was not found!` : The folder which you have passed to the `--image_folder` arguement doesn't exits.
2. `4 points of sorrounding contour can't be found in [file], You should take a better picture of paper !` : The image [file] you have scanned doesn't capture all surrounding rectangle of the paper. 
3. `QR code detected but didn't read its information in [file] ,You should take a better picture of paper !` : The [file] is not clear enough to read QRcode.
4. `No QR code detected in [file] ,You should take a better picture of paper !` : The [file] is not clear enough to detect QRcode.
5. `Please Scan Again! markers not found completely in [file]!` : The [file] is not clear enough to detect markers!
6. `Please Scan Again! choices not found completely in [file] !` : The [file] is not clear enough to detect options contours!

