import face_recognition
import cv2


rgb_hat=cv2.imread("rgb_hat.jpg",-1)
a=cv2.imread("hat_alpha.jpg",-1)



# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

face_locations = []



while True:
    ret, frame = video_capture.read()
    # # Resize frame of video to 1/4 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame, model="cnn")
    for face_location in face_locations:
         # Print the location of each face in this image
         top, right, bottom, left = face_location
         # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,right))

         # Put the blurred face region back into the frame image
         # frame[top:bottom, left:right] = face_image

    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        # top *= 4
        # right *= 4
        # bottom *= 4
        # left *= 4
        x, y, w, h =  left, top, right - left, bottom - top
        # face_image = frame[top:bottom, left:right]

        eyes_center = ((x+(w//2))*4, y // 2)
        factor = 1.5
        resized_hat_h = int(round(rgb_hat.shape[0] * w*4/ rgb_hat.shape[1] * factor))
        resized_hat_w = int(round(rgb_hat.shape[1] * w*4 / rgb_hat.shape[1] * factor))

        if resized_hat_h > y:
            resized_hat_h = y - 1

            # 根据人脸大小调整帽子大小
        resized_hat = cv2.resize(rgb_hat, (resized_hat_w, resized_hat_h))
        # 帽子相对与人脸框上线的偏移量
        mask = cv2.resize(a, (rgb_hat.shape[1], rgb_hat.shape[0]))
        mask_inv = cv2.bitwise_not(mask)
        # face_image = cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        dh=100
        dw=0
        roi = frame[y + dh - resized_hat_h:y + dh,
            (eyes_center[0] - resized_hat_w // 3):(eyes_center[0] + resized_hat_w // 3 * 2)]
        roi = roi.astype(float)
        mask_inv = cv2.merge((mask_inv, mask_inv, mask_inv))
        alpha = mask_inv.astype(float) / 255

        # 相乘之前保证两者大小一致（可能会由于四舍五入原因不一致）
        alpha = cv2.resize(alpha, (roi.shape[1], roi.shape[0]))
        roi = cv2.multiply(alpha, roi)
        roi = roi.astype('uint8')

        # 提取帽子区域
        hat = cv2.bitwise_and(rgb_hat, rgb_hat, mask=mask)

        # 相加之前保证两者大小一致（可能会由于四舍五入原因不一致）
        hat = cv2.resize(hat, (roi.shape[1], roi.shape[0]))
        # 两个ROI区域相加
        add_hat = cv2.add(roi, hat)

        frame[y + dh - resized_hat_h:y + dh,
            (eyes_center[0] - resized_hat_w // 3):(eyes_center[0] + resized_hat_w // 3 * 2)]= add_hat


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cv2.destroyAllWindows()
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()