#----------------- Parse Arguements ----------------
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--inputimage", dest="input", default=None)
parser.add_option("--outputpath", dest="output", default=None)
parser.add_option("--weight", dest="weight", default=0.15)
parser.add_option("--overall_threshold", dest="overall_threshold", default=1e3)
parser.add_option("--db4_threshold", dest="db4_threshold", default=15)
parser.add_option("--need_edgemap", dest="edgemap", default=False)

(options, args) = parser.parse_args()

if options.input is None:
    raise ValueError("Provide the path of input image")
if options.output is None:
    raise ValueError("Provide the path of output directory")


print("Your Image is processing......please be patient")

# --------------------Importing the Libs
import math

import cv2
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt

# --------------------------------------------- Defining Sampling Operation --------------------------------------------
def downSample(img_input, direction):
    if direction == direction_rows:
        h, w = img_input.shape
        img_output = np.zeros((int(h / 2),int(w)))
        i = 0
        new = i
        while i < h:
            img_output[new, :] = img_input[i, :]
            new = new + 1
            i = i + 2

        return img_output

    elif direction == direction_columns:
        h, w = img_input.shape
        img_output = np.zeros((int(h), int(w / 2)))
        i = 0
        new = i
        while i < w:
            img_output[:, new] = img_input[:, i]
            new = new + 1
            i = i + 2

        return img_output


def upSample(img_input, order):
    img_output = scipy.ndimage.zoom(img_input, order)
    return img_output


# ----------------------------------------- Implementation of db4 Wavelet Filter---------------------------------------
def db4ConvolveRowsLowPass(img_input):
    #coeff: -0.0105974018 0.0328830117 0.0308413818 -0.1870348117 -0.0279837694 0.6308807679 0.7148465706 0.2303778133
    
    img_input_pad = cv2.copyMakeBorder(img_input, 0, 0, 3, 4, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape


    img_db4_convolved = np.zeros((height, width))

    for x in range(0, height):
        for y in range(3, width - 4):
            img_db4_convolved[x, y] = img_input_pad[x, y - 3] * (-0.0105974018) + img_input_pad[x, y - 2] * (0.0328830117) + img_input_pad[x, y - 1] * (0.0308413818) + \
                                          img_input_pad[x, y] * (-0.1870348117) + img_input_pad[x, y + 1] * (-0.0279837694) + \
                                          img_input_pad[x, y + 2] * (0.6308807679) + img_input_pad[x, y + 3] * (0.7148465706) + img_input_pad[x, y + 4] * (0.2303778133)
    return img_db4_convolved[0:height, 3:width - 4]


def db4ConvolveColumnsLowPass(img_input):
    #coeff: -0.0105974018 0.0328830117 0.0308413818 -0.1870348117 -0.0279837694 0.6308807679 0.7148465706 0.2303778133
    
    img_input_pad = cv2.copyMakeBorder(img_input, 3, 4, 0, 0, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape


    img_db4_convolved = np.zeros((height, width))

    for y in range(0, width):
        for x in range(3, height - 4):
            img_db4_convolved[x, y] = img_input_pad[x-3, y] * (-0.0105974018) + img_input_pad[x-2, y] * (0.0328830117) + img_input_pad[x-1, y] * (0.0308413818) + \
                                          img_input_pad[x, y] * (-0.1870348117) + img_input_pad[x+1, y] * (-0.0279837694) + \
                                          img_input_pad[x+2, y] * (0.6308807679) + img_input_pad[x+3, y] * (0.7148465706) + img_input_pad[x+4, y] * (0.2303778133)

    return img_db4_convolved[3:height - 4, 0:width]


def db4ConvolveRowsHighPass(img_input):
     #coeff: -0.2303778133 0.7148465706 -0.6308807679 -0.0279837694  0.1870348117  0.0308413818  -0.0328830117  -0.0105974018
    
    img_input_pad = cv2.copyMakeBorder(img_input, 0, 0, 3, 4, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape


    img_db4_convolved = np.zeros((height, width))

    for x in range(0, height):
        for y in range(3, width - 4):
            img_db4_convolved[x, y] = img_input_pad[x, y - 3] * (-0.2303778133) + img_input_pad[x, y - 2] * (0.7148465706) + img_input_pad[x, y - 1] * (-0.6308807679) + \
                                          img_input_pad[x, y] * (-0.0279837694) + img_input_pad[x, y + 1] * (0.1870348117) + \
                                          img_input_pad[x, y + 2] * (0.0308413818) + img_input_pad[x, y + 3] * (-0.0328830117) + img_input_pad[x, y + 4] * (-0.0105974018)
    return img_db4_convolved[0:height, 3:width - 4]



def db4ConvolveColumnsHighPass(img_input):
    #coeff: -0.2303778133 0.7148465706 -0.6308807679 -0.0279837694  0.1870348117  0.0308413818  -0.0328830117  -0.0105974018
    
    img_input_pad = cv2.copyMakeBorder(img_input, 3, 4, 0, 0, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape


    img_db4_convolved = np.zeros((height, width))

    for y in range(0, width):
        for x in range(3, height - 4):
            img_db4_convolved[x, y] = img_input_pad[x-3, y] * (-0.2303778133) + img_input_pad[x-2, y] * (0.7148465706) + img_input_pad[x-1, y] * (-0.6308807679) + \
                                          img_input_pad[x, y] * (-0.0279837694) + img_input_pad[x+1, y] * (0.1870348117) + \
                                          img_input_pad[x+2, y] * (0.0308413818) + img_input_pad[x+3, y] * (-0.0328830117) + img_input_pad[x+4, y] * (-0.0105974018)
    return img_db4_convolved[3:height - 4, 0:width]



# ----------------------------------------- Implementation of Coiflet Wavelet Filter------------------------------------
def coifletConvolveRowsLowPass(img_input):
    # -0.0157   -0.0727    0.3849      0.8526    0.3379   -0.0727
    img_input_pad = cv2.copyMakeBorder(img_input, 0, 0, 2, 3, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape
    img_coiflet_convolved = np.zeros((height, width))

    for x in range(0, height):
        for y in range(2, width - 3):
            img_coiflet_convolved[x, y] = img_input_pad[x, y - 2] * (-0.0157) + img_input_pad[x, y - 1] * (-0.0727) + \
                                          img_input_pad[x, y] * (0.3849) + img_input_pad[x, y + 1] * (0.8526) + \
                                          img_input_pad[x, y + 2] * (0.3379) + img_input_pad[x, y + 3] * (-0.0727)

    return img_coiflet_convolved[0:height, 2:width - 3]


def coifletConvolveColumnsLowPass(img_input):
    # -0.0157   -0.0727    0.3849      0.8526    0.3379   -0.0727
    img_input_pad = cv2.copyMakeBorder(img_input, 2, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape
    img_coiflet_convolved = np.zeros((height, width))

    for x in range(2, height - 3):
        for y in range(0, width):
            img_coiflet_convolved[x, y] = img_input_pad[x - 2, y] * (-0.0157) + img_input_pad[x - 1, y] * (-0.0727) + \
                                          img_input_pad[x, y] * (0.3849) + img_input_pad[x + 1, y] * (0.8526) + \
                                          img_input_pad[x + 2, y] * (0.3379) + img_input_pad[x + 3, y] * (-0.0727)

    return img_coiflet_convolved[2:height - 3, 0:width]


def coifletConvolveRowsHighPass(img_input):
    # 0.0727    0.3379   -0.8526    0.3849    0.0727   -0.0157; High
    img_input_pad = cv2.copyMakeBorder(img_input, 0, 0, 2, 3, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape
    img_coiflet_convolved = np.zeros((height, width))

    for x in range(0, height):
        for y in range(2, width - 3):
            img_coiflet_convolved[x, y] = img_input_pad[x, y - 2] * (0.0727) + img_input_pad[x, y - 1] * (0.3379) + \
                                          img_input_pad[x, y] * (-0.8526) + img_input_pad[x, y + 1] * (0.3849) + \
                                          img_input_pad[x, y + 2] * (0.0727) + img_input_pad[x, y + 3] * (-0.0157)

    return img_coiflet_convolved[0:height, 2:width - 3]


def coifletConvolveColumnsHighPass(img_input):
    # 0.0727    0.3379   -0.8526    0.3849    0.0727   -0.0157; High
    img_input_pad = cv2.copyMakeBorder(img_input, 2, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)

    height, width = img_input_pad.shape
    img_coiflet_convolved = np.zeros((height, width))

    for x in range(2, height - 3):
        for y in range(0, width):
            img_coiflet_convolved[x, y] = img_input_pad[x - 2, y] * (0.0727) + img_input_pad[x - 1, y] * (0.3379) + \
                                          img_input_pad[x, y] * (-0.8526) + img_input_pad[x + 1, y] * (0.3849) + \
                                          img_input_pad[x + 2, y] * (0.0727) + img_input_pad[x + 3, y] * (-0.0157)
    return img_coiflet_convolved[2:height - 3, 0:width]


# ----------------------------------- Implementation of Wavelet Filters (LL, LH, HL, HH) -------------------------------
def performLL(img_input, filter_wavelet):
    if filter_wavelet == filter_db4:
        img_L_rows = db4ConvolveRowsLowPass(img_input)
        img_L_rows_downsampled = downSample(img_L_rows, direction_rows)

        img_L_columns = db4ConvolveColumnsLowPass(img_L_rows_downsampled)
        img_L_columns_downsampled = downSample(img_L_columns, direction_columns)

        return img_L_columns_downsampled

    elif filter_wavelet == filter_coiflet:
        img_L_rows = coifletConvolveRowsLowPass(img_input)
        img_L_rows_downsampled = downSample(img_L_rows, direction_rows)

        img_L_columns = coifletConvolveColumnsLowPass(img_L_rows_downsampled)
        img_L_columns_downsampled = downSample(img_L_columns, direction_columns)

        return img_L_columns_downsampled


def performLH(img_input, filter_wavelet):
    if filter_wavelet == filter_db4:
        img_L_rows = db4ConvolveRowsLowPass(img_input)
        img_L_rows_downsampled = downSample(img_L_rows, direction_rows)

        img_H_columns = db4ConvolveColumnsHighPass(img_L_rows_downsampled)
        img_H_columns_downsampled = downSample(img_H_columns, direction_columns)

        return img_H_columns_downsampled

    elif filter_wavelet == filter_coiflet:
        img_L_rows = coifletConvolveRowsLowPass(img_input)
        img_L_rows_downsampled = downSample(img_L_rows, direction_rows)

        img_H_columns = coifletConvolveColumnsHighPass(img_L_rows_downsampled)
        img_H_columns_downsampled = downSample(img_H_columns, direction_columns)

        return img_H_columns_downsampled


def performHL(img_input, filter_wavelet):
    if filter_wavelet == filter_db4:
        img_H_rows = db4ConvolveRowsHighPass(img_input)
        img_H_rows_downsampled = downSample(img_H_rows, direction_rows)

        img_L_columns = db4ConvolveColumnsLowPass(img_H_rows_downsampled)
        img_L_columns_downsampled = downSample(img_L_columns, direction_columns)

        return img_L_columns_downsampled

    elif filter_wavelet == filter_coiflet:
        img_H_rows = coifletConvolveRowsHighPass(img_input)
        img_H_rows_downsampled = downSample(img_H_rows, direction_rows)

        img_L_columns = coifletConvolveColumnsLowPass(img_H_rows_downsampled)
        img_L_columns_downsampled = downSample(img_L_columns, direction_columns)

        return img_L_columns_downsampled


def performHH(img_input, filter_wavelet):
    if filter_wavelet == filter_db4:
        img_H_rows = db4ConvolveRowsHighPass(img_input)
        img_H_rows_downsampled = downSample(img_H_rows, direction_rows)

        img_H_columns = db4ConvolveColumnsHighPass(img_H_rows_downsampled)
        img_H_columns_downsampled = downSample(img_H_columns, direction_columns)

        return img_H_columns_downsampled

    elif filter_wavelet == filter_coiflet:
        img_H_rows = coifletConvolveRowsHighPass(img_input)
        img_H_rows_downsampled = downSample(img_H_rows, direction_rows)

        img_H_columns = coifletConvolveColumnsHighPass(img_H_rows_downsampled)
        img_H_columns_downsampled = downSample(img_H_columns, direction_columns)

        return img_H_columns_downsampled


# --------------------------------------------- Pipeline for Wavelet Edge Detection-------------------------------------
def pipeline(img_input, filter_wavelet, threshold_subband):
    # ------------------------ Step 1 : Applying Wavelet Filter, Threshold and Down Sampling ---------------------------
    # Level 1 : Perform filters LL, LH, HL and HH on Input Image of size [M, N] and generate [M/2, N/2] size Image
    img_LL_2 = performLL(img_input, filter_wavelet)
    img_LH_2 = performLH(img_input, filter_wavelet)
    img_HL_2 = performHL(img_input, filter_wavelet)
    img_HH_2 = performHH(img_input, filter_wavelet)


    # Applying threshold to LH, HL and HH subbands
    img_LH_2 = performThresholding(img_LH_2, threshold_subband, False)
    img_HL_2 = performThresholding(img_HL_2, threshold_subband, False)
    img_HH_2 = performThresholding(img_HH_2, threshold_subband, False)


    # Level 2 : Perform filters LL, LH, HL and HH on LL Image of size [M/2, N/2] and generate [M/4, N/4] size Image
    img_LL_4 = performLL(img_LL_2, filter_wavelet)
    img_LH_4 = performLH(img_LL_2, filter_wavelet)
    img_HL_4 = performHL(img_LL_2, filter_wavelet)
    img_HH_4 = performHH(img_LL_2, filter_wavelet)


    # Applying threshold to LH, HL and HH subbands
    img_LH_4 = performThresholding(img_LH_4, threshold_subband, False)
    img_HL_4 = performThresholding(img_HL_4, threshold_subband, False)
    img_HH_4 = performThresholding(img_HH_4, threshold_subband, False)


    # Level 3 : Perform filters LL, LH, HL and HH on LL Image of size [M/4, N/4] and generate [M/8, N/8] size Image
    img_LL_8 = performLL(img_LL_4, filter_wavelet)
    img_LH_8 = performLH(img_LL_4, filter_wavelet)
    img_HL_8 = performHL(img_LL_4, filter_wavelet)
    img_HH_8 = performHH(img_LL_4, filter_wavelet)


    # Applying threshold to LH, HL and HH subbands
    img_LH_8 = performThresholding(img_LH_8, threshold_subband, False)
    img_HL_8 = performThresholding(img_HL_8, threshold_subband, False)
    img_HH_8 = performThresholding(img_HH_8, threshold_subband, False)


    # Level 4 : Perform filters LL, LH, HL and HH on LL Image of size [M/8, N/8] and generate [M/16, N/16] size Image
    img_LL_16 = performLL(img_LL_8, filter_wavelet)
    img_LH_16 = performLH(img_LL_8, filter_wavelet)
    img_HL_16 = performHL(img_LL_8, filter_wavelet)
    img_HH_16 = performHH(img_LL_8, filter_wavelet)


    # Applying threshold to LH, HL and HH subbands
    img_LH_16 = performThresholding(img_LH_16, threshold_subband, False)
    img_HL_16 = performThresholding(img_HL_16, threshold_subband, False)
    img_HH_16 = performThresholding(img_HH_16, threshold_subband, False)


    # --------------------------------- Step 2 : Up Sampling and Image Matrix Multiplication ---------------------------
    # Blowing Up Level 4 to Level 3 : Up Sampling LH, HL and HH Images of size [M/16, N/16] to size [M/8, N/8]
    img_LH_16_blown = upSample(img_LH_16, scale_order)
    img_HL_16_blown = upSample(img_HL_16, scale_order)
    img_HH_16_blown = upSample(img_HH_16, scale_order)


    # Element wise Multiplication Level 4 blown up Images of size [M/8, N/8] with original Level 3 Images of size [M/8, N/8]
    img_LH_8_new = img_LH_16_blown * img_LH_8
    img_HL_8_new = img_HL_16_blown * img_HL_8
    img_HH_8_new = img_HH_16_blown * img_HH_8


    # Blowing Up Level 3 to Level 2 : Up Sampling LH, HL and HH Images of size [M/8, N/8] to size [M/4, N/4]
    img_LH_8_blown = upSample(img_LH_8_new, scale_order)
    img_HL_8_blown = upSample(img_HL_8_new, scale_order)
    img_HH_8_blown = upSample(img_HH_8_new, scale_order)


    # Element wise Multiplication Level 3 blown up Images of size [M/4, N/4] with original Level 2 Images of size [M/4, N/4]
    img_LH_4_new = img_LH_8_blown * img_LH_4
    img_HL_4_new = img_HL_8_blown * img_HL_4
    img_HH_4_new = img_HH_8_blown * img_HH_4

   
    # Blowing Up Level 2 to Level 1 : Up Sampling LH, HL and HH Images of size [M/4, N/4] to size [M/2, N/2]
    img_LH_4_blown = upSample(img_LH_4_new, scale_order)
    img_HL_4_blown = upSample(img_HL_4_new, scale_order)
    img_HH_4_blown = upSample(img_HH_4_new, scale_order)


    # Element wise Multiplication Level 2 blown up Images of size [M/2, N/2] with original Level 1 Images of size [M/2, N/2]
    img_LH_2_new = img_LH_4_blown * img_LH_2
    img_HL_2_new = img_HL_4_blown * img_HL_2
    img_HH_2_new = img_HH_4_blown * img_HH_2



    # --------------------------------------- Step 3 : Generating Final Edge map ---------------------------------------
    img_LH_final = np.multiply(img_LH_2_new, img_LH_2_new)
    img_HL_final = np.multiply(img_HL_2_new, img_HL_2_new)
    img_HH_final = np.multiply(img_HH_2_new, img_HH_2_new)
    img_final_edges = np.sqrt(img_LH_final + img_HL_final + img_HH_final)

    return True, img_final_edges, img_LH_final, img_HL_final, img_HH_final


# ---------------------------------------------- Applying Threshold -------------------------------------------------
def performThresholding(img_input, val_threshold, isFinalImg):
    low_values_indices = img_input < val_threshold  # Where values are low
    img_input[low_values_indices] = 0  # All low values set to 0

    if isFinalImg:
        high_values_indices = img_input > 0  # Where values are low
        img_input[high_values_indices] = 0  # All low values set to 0
        img_input[low_values_indices] = 255
        

    return img_input


# ----------------------------------------- Printing Final Comparison Results ------------------------------------------
def printFinalComparisonResults(img_orig_db4_edges, img_orig_coiflet_edges):
    img_orig_db4_edges = performThresholding(img_orig_db4_edges, threshold_final, True)
    img_orig_coiflet_edges = performThresholding(img_orig_coiflet_edges, threshold_final, True)


    plt.subplot(231), plt.imshow(img_orig_db4_edges, cmap='gray'), plt.title('Original Image with db4 Filter')
    plt.xticks([]), plt.yticks([])


    plt.subplot(234), plt.imshow(img_orig_coiflet_edges, cmap='gray'), plt.title('Original Image with Coiflet Filter')
    plt.xticks([]), plt.yticks([])

    plt.show()
    


# ---------------------------------------------- Variable Declarations -------------------------------------------------
direction_rows = 0
direction_columns = 1

scale_order = 2

filter_db4 = "db4 Filter"
filter_coiflet = "Coiflet Filter"

threshold_final = options.overall_threshold

threshold_original_db4 = options.db4_threshold
threshold_original_coiflet = 14
weightage = float(options.weight)
outputpath = options.output
edgemap = options.edgemap

# ----------------------------------------------- Program Main Function ------------------------------------------------
Images = [options.input]
 
Filters = [filter_db4]

   
for image in Images:
    img_original = cv2.imread(image)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    img_original_db4_edges = img_original
    img_original_coiflet_edges = img_original

        
            
    for filter_wavelet in Filters:
        if filter_wavelet == filter_db4:
            success, img_original_db4_edges, img_db4_LH, img_db4_HL, img_db4_HH = pipeline(img_original, filter_wavelet, threshold_original_db4)

        elif filter_wavelet == filter_coiflet:
            success, img_original_coiflet_edges = pipeline(img_original, filter_wavelet, threshold_original_coiflet)

    img_orig_db4_edges = performThresholding(img_original_db4_edges, threshold_final, True)
        
    img = cv2.resize(img_original, (img_original.shape[0]//2, img_original.shape[1]//2)) 
    img_original = img
    img = img + weightage*img_orig_db4_edges
        
    # plt.subplot(231), plt.imshow(img_original, cmap='gray'), plt.title('original')
    # plt.xticks([]), plt.yticks([])
    # print(img_original.shape)
        
    # plt.subplot(234), plt.imshow(img, cmap='gray'), plt.title('enhanced db4 img')
    # plt.xticks([]), plt.yticks([])
    # print(img.shape)
        
    # cv2_imshow(img_db4_HL)
    # cv2_imshow(img_db4_LH)
    # cv2_imshow(img_db4_HH)
    # cv2_imshow(img_orig_db4_edges)
    # cv2_imshow(img_original)
    # cv2_imshow(img)
    # cv2.waitKey(0)
    
    cv2.imwrite(outputpath + '/output.jpg', img)
    if edgemap:
        cv2.imwrite(outputpath + '/vertical_edgemap.jpg', img_db4_HL)
        cv2.imwrite(outputpath + '/horizontal_edgemap.jpg', img_db4_LH)
        cv2.imwrite(outputpath + '/diagonal_edgemap.jpg', img_db4_HH)
        cv2.imwrite(outputpath + '/overall_edgemap.jpg', img_orig_db4_edges)
        
        
print("Image Processing completed......check the output directory that you have provided.")
    
    
    

    
    

