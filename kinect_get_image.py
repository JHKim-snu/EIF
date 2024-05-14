###########################
# Check on the viewer first!!!
# to install, follow: https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1263
# cd Azure-Kinect-Sensor-SDK/scripts
# k4aviewer
###########################
# https://github.com/etiennedub/pyk4a

from pyk4a import PyK4A

# Load camera with the default config
k4a = PyK4A()
k4a.start()


# exit()

# Get the next capture (blocking function)
capture = k4a.get_capture()
img_color = capture.color

# Display with pyplot
from matplotlib import pyplot as plt
# plt.imshow(img_color[:, :, 2::-1]) # BGRA to RGB
plt.imsave('azure_test.png',img_color[:, :, 2::-1])
# plt.show()

k4a.stop()

exit()
