import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr



def License_Detection(id_Img, count):

    filename = f"vehicle_{id_Img}_{count}.png" 
            
    reader = easyocr.Reader(['en'])

    image = cv2.imread('OutputImage/'+filename)

    results_1 = reader.readtext(image)

  
    plate_image = None
    padding = -4
    License_ID = ""
    pre_license_ID  = ""

    if results_1: ###Cropping the image and enlarging the details
        for (bbox, text, prob) in results_1:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            top_right = tuple(map(int, top_right))
            bottom_right = tuple(map(int, bottom_right))
            bottom_left = tuple(map(int, bottom_left))

            x_min = max(0, min(top_left[0], bottom_left[0]) - padding)
            x_max = min(image.shape[1], max(top_right[0], bottom_right[0]) + padding)
            y_min = max(0, min(top_left[1], top_right[1]) - padding)
            y_max = min(image.shape[0], max(bottom_left[1], bottom_right[1]) + padding)
            plate_image = image[y_min:y_max, x_min:x_max]
            break

        if plate_image is not None: ###Image Processing and Filtering Steps 
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            scale_factor = 10
            high_res = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            blur = cv2.GaussianBlur(high_res, (5, 5), 0)

            kernel = np.array([[0, -1,  0], [-1, 4, -1], [0, -1,  0]])
            laplacian = cv2.filter2D(blur, -1, kernel)
            sharp = cv2.addWeighted(high_res, 1.8, laplacian, -0.3, 0)

            kernel = np.ones((3, 3), np.uint8)
            morph = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, kernel)
            binary_morph = cv2.bitwise_not(morph)

            result_2 = reader.readtext(binary_morph)

            if result_2:
                top_left = tuple(result_2[0][0][0])
                bottom_right = tuple(result_2[0][0][2])
                text = result_2[0][1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = morph
                img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
                img = cv2.putText(img, text, top_left, font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                for detection in result_2:
                    pre_license_ID += detection[1]

                reversed_License_ID = pre_license_ID[::-1]
                reversed_License_ID_no_spaces = reversed_License_ID.replace(' ', '')
                reversed_License_ID = reversed_License_ID_no_spaces[0:6]
                License_ID = reversed_License_ID[::-1]
                print(License_ID)
                return License_ID
            
            else:
                return "Undefined!"
        else:
            return "Undefined!"
    else:
        return "Undefined!"
