def QR():
    import cv2
    import numpy as np
    from pyzbar.pyzbar import decode

    def decoder(image):
        gray_img = cv2.cvtColor(image, 0)
        barcode = decode(gray_img)

        for obj in barcode:
            points = obj.polygon
            (x, y, w, h) = obj.rect
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 3)

            barcodeData = obj.data.decode("utf-8")
            barcodeType = obj.type
            string = "Data " + str(barcodeData) + " | Type " + str(barcodeType)

            cv2.putText(frame, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            print("Barcode: " + barcodeData + " | Type: " + barcodeType)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 960)
    while True:
        ret, frame = cap.read()
        decoder(frame)
        cv2.imshow('Image', frame)
        code = cv2.waitKey(10)
        if code == 27:
            break


def getQR():
    import cv2
    import numpy as np
    from pyzbar.pyzbar import decode


    def decoder(image):
        gray_img = cv2.cvtColor(image, 0)
        barcode = decode(gray_img)

        result = ""

        for obj in barcode:
            points = obj.polygon
            (x, y, w, h) = obj.rect
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 3)

            barcodeData = obj.data.decode("utf-8")
            barcodeType = obj.type
            string = "Data " + str(barcodeData) + " | Type " + str(barcodeType)

            cv2.putText(frame, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            print("Barcode: " + barcodeData + " | Type: " + barcodeType)

            result = barcodeData
            break

        return result


    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 960)

    result = ""

    while result == "":
        ret, frame = cap.read()
        result = decoder(frame)
        cv2.imshow('Image', frame)
        code = cv2.waitKey(10)
        if code == 27:
            break

    print(result)

# getQR()

QR()