import cv2
from matplotlib import pyplot as plt

image_path = r"C:\Users\admin\Desktop\stophi.png"
cascade_path = r"C:\Users\admin\Downloads\stop_sign_classifier_2.xml"

# Load the image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Image file could not be loaded. Check the path.")

# Convert to grayscale and enhance contrast
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.equalizeHist(img_gray)

# Convert to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load Haar cascade
stop_data = cv2.CascadeClassifier(cascade_path)
if stop_data.empty():
    raise FileNotFoundError("Cascade file could not be loaded. Check the path.")

# Detect stop signs with tuned parameters
stop_signs = stop_data.detectMultiScale(
    img_gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30, 30),
    maxSize=(400, 400)
)

# Draw rectangles around detected signs
if len(stop_signs) == 0:
    print("No stop signs detected. Try adjusting parameters or using a better cascade.")
else:
    for (x, y, w, h) in stop_signs:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 5)
    print(f"Detected {len(stop_signs)} stop sign(s).")

# Display result
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
