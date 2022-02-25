import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_hist(img, bins):  # to make a histogram (count distribution frequency)
    h,w = img.shape[:2]
    val = [0]*bins
    for i in range(h):
        for j in range(w):
            val[img[i,j]]+=1
    return np.asarray(val) , np.arange(bins)



def manual_threshold(im_in, threshold, OTSU):
# Threshold image with the threshold of your choice
    final_image = im_in.copy()
    final_image[ im_in > threshold] = 255
    final_image[im_in<threshold] = 0
    if OTSU:
        cv2.imshow("Output from Otsu Thresholding", final_image)
        cv2.imwrite("Otsu_out.png", final_image)
        cv2.waitKey(0)
        return
    
    cv2.imshow("Output from Manual Thresholding", final_image)
    cv2.waitKey(0)
    cv2.imwrite("Manual_out.png", final_image)

    
    return None


def otsu_threshold(im_in):
    # Create Otsu thresholded image
    h,w = img.shape[:2]
    mean_weight = 1/ (h*w)
    hist , bins = generate_hist(im_in, bins = 256)

    print(type(hist))
    print(type(bins))
    print(bins.shape)

    optimal_threshold = -1000
    inter_class_variance = -1000

    #to plot the variance
    plot_variance = []

    pixel_intensity = np.asarray(range(0,256))
    for i in range(len(bins[0:-1])):
        probability_0 = np.sum(hist[:i])
        probability_1 = np.sum(hist[i:])
        w0 = probability_0 * mean_weight
        w1 = probability_1 * mean_weight

        mean_u0 = np.sum( pixel_intensity[:i] * hist[:i] ) / probability_0
        mean_u1 = np.sum( pixel_intensity[i:] * hist[i:] ) / float(probability_1)

        temp = w0 * w1 * ((mean_u0 - mean_u1) **2)
        plot_variance.append(temp)

        if temp>inter_class_variance:
            optimal_threshold = i
            inter_class_variance = temp
        

    print("\nOptimal value of Thresholding is ",optimal_threshold)
    print("\nMaximum Inter-class Varinace is ", "{:.2f}".format(inter_class_variance))

    manual_threshold(im_in, optimal_threshold, True)

    #ploting histograms
    plt.figure(1)
    plt.title("Original Histogram")
    plt.xlabel("Pixel Intensity")
    plt.xlim([0,256])
    plt.ylabel("Frequency")
    #bins is an array of size 257 so we ignore the last bin 
    plt.plot(bins, hist )

    #plotting inter-class variance vs threshold
    plt.figure(2)
    plt.title("Inter-Class Variance vs Threshold")
    plt.xlabel("Threshold")
    plt.xlim([0,255])
    plt.ylabel("Inter-Class Variance")
    plt.plot(range(0,255), plot_variance)



    plt.show()


    return None




if __name__ == '__main__':
    img = cv2.imread('a2_c.png',0)

    cv2.imshow("Original Image" , img)


    #manual thresholding at value 169
    manual_threshold(img, 169, False)

    otsu_threshold(img)

    