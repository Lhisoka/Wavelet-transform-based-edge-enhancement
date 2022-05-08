# Wavelet-transform-based-edge-enhancement
## Wavelet Transform based edge enhancement through Daubechies wavelet as mother wavelet

1) Firstly, install all the dependencies from 'requirements.txt' 
2) Clone this repo or download the zip file
3) Open the command prompt terminal and navigate to the directory location named "GNR602"
4) Let the image on which you want to do the edge enhancement operation be 'Carriage.jpg' (contained in the "sample" folder)
5) Type the following in the terminal:
   python model.py --inputimage sample/Carriage.jpg --outputpath output
6) The following message will be displayed: "Your Image is processing......please be patient"
7) Once the program runs successfully, it will show: "Image Processing completed......check the output directory that you have provided."
8) After this, check the output file from the "output" directory
9) If you wish to also output the intermediate images which include: vertical, horizontal, diagonal and overall edgemaps along with the output, change the command to the following:
   python model.py --inputimage sample/Carriage.jpg --outputpath output --need_edgemap True
10) If you wish to change the default values of weight, db4 threshold and overall threshold, change the command to the following:
    python model.py --inputimage sample/Carriage.jpg --outputpath output --need_edgemap True --weight new_weight --overall_threshold new_overall_threshold --db4_threshold new_db4_threshold 
  (Where new_weight, new_overall_threshold, new_db4_threshold are the numerical values of the new weight, new overall threshold and new db4 threshold respectively)
  You can tune these threshold values and see which values give the best results.
