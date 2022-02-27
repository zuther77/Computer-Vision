import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def generate_hist(img, bins):  # to make a histogram (count distribution frequency)
    h,w = img.shape[:2]
    val = [0]*bins
    for i in range(h):
        for j in range(w):
            val[img[i,j]]+=1
    return np.asarray(val) , np.arange(bins)


def create_pdf(hist, N):
    #divide histogram by number of pizels
    pdf = hist / N
    return pdf


def create_cdf(hist ,N):
    pdf = create_pdf(hist ,N)
    cdf = np.zeros((pdf.shape))
    cdf[0] = pdf[0]

    #get cummulative sum
    for i in range(1,len(pdf)):
        cdf[i] = cdf[i-1] + pdf[i]

    return cdf


def histogram_equalization(img, hist,N):
    cdf = create_cdf(hist ,N)

    #pixel mapping
    normalized_cdf =  np.floor(255 * cdf).astype("uint8")

    img_to_list = list(img.flatten())
    equalized_img = [ normalized_cdf[i] for i in img_to_list]

    #need to reshape since we used flatted image during mapping
    equalized_img = np.reshape(np.asarray(equalized_img) , img.shape)

    return equalized_img




if __name__ == '__main__':
    
    img = cv2.imread(sys.argv[1],0)
    h,w = img.shape[:2]


    #convert to uint8 
    img_uint8 = ( (img - np.min(img)) * 1/(np.max(img) - np.min(img))  * 255  ).astype('uint8')


    #genetate histogram
    hist_og , bins = generate_hist(img_uint8, bins = 256 )
    
    #equalized Histogram
    img_equalized = histogram_equalization(img_uint8, hist_og , h*w)
    H,W= img_equalized.shape[:2]



    #show output
    cv2.imshow("Original Image", img_uint8)
    cv2.imshow("Eqaulized Image", img_equalized)
    cv2.imwrite("a1_uint8.png", img_uint8)
    cv2.imwrite("a1_equalized.png", img_equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    #plot original histogram and Equalized
    plt.figure(1)
    plt.title("Original Histogram")
    plt.xlabel("Pixel Intensity")
    plt.xlim([0,256])
    plt.ylabel("Frequency")
    #bins is an array of size 257 so we ignore the last bin 
    plt.plot(bins, hist_og ) 

    hist_equalized , bins = generate_hist(img_equalized, bins = 256 )
    plt.figure(2)
    plt.title("Equalized Histogram")
    plt.xlabel("Pixel Intensity")
    plt.xlim([0,256])
    plt.ylabel("Frequency")
    plt.plot(bins, hist_equalized) 


    #plot original vs equalized pdf
    pdf = create_pdf(hist_og , h*w)
    pdf_e = create_pdf(hist_equalized, H*W)
    plt.figure(3)
    plt.title("Original PDF vs Equalized PDF")
    plt.plot(pdf_e, label = "Equalized")
    plt.plot(pdf , label = "Original")
    plt.legend()

    #plot original vs equalized cdf
    cdf = create_cdf(hist_og, h*w)
    cdf_e = create_cdf(hist_equalized, img_equalized.shape[-1] * img_equalized.shape[0])
    plt.figure(4)
    plt.title("Original CDF vs Equalized CDF")
    plt.plot(cdf_e, label = "Equalized")
    plt.plot(cdf , label = "Original")
    plt.legend()
    plt.show()
