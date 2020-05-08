import time

import cv2
import pyscreenshot as screen
screenshot_png = "123.png"
while True:
    try:
        time.sleep(3)
        im = screen.grab(bbox=(10, 10, 800, 800))
        im.save(screenshot_png)

        image = cv2.imread(screenshot_png)

        cv2.imshow("Input", image)
        cv2.waitKey(5)
    except Exception as e:
        print("Error ...." + str(e))
        pass
    # input("Press Enter to continue...")
cv2.destroyAllWindows()