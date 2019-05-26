import cv2


class OpenCV():
    def __init__(self, cascade_path):
        self.PATH = cascade_path

    def crop(self, image_path):
        cascade = cv2.CascadeClassifier(self.PATH)
        print(cascade)
        image = cv2.imread(image_path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_img, 1.3, 7, minSize=(30, 30))
        print(faces)
        faces_arr = list()
        count = 0
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r)
            count += 1
            face_img = image[y - int(nr // 1.11):y + h + nr, x - nr:x + w + nr]
            path = 'cropped_images/image' + str(count) + '.jpg'
            cv2.imwrite(path, face_img)
            faces_arr.append(path)
        return faces_arr
