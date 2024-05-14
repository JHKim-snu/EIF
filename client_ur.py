import sys
import cv2
import base64
import socket
import numpy as np
from ur_controller import RobotController
import time
import urx

# socket
HOST = '147.47.200.146'
PORT = 9998

def main():
    robot = urx.Robot('192.168.1.117') # Network IP Address
    time.sleep(0.2)
    print("roobt successfully connected")
    cli_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli_sock.connect((HOST,PORT))
    # rospy.init_node('rt0_client', anonymous=False, disable_signals=True)

    rc = RobotController()
    # rospy.loginfo('Start')
    rc.arm.to_home_pose()
    rc.arm.finger_open()
    
    while 1:
        try:
            input('Press any key to get images')
            capture = rc.camera.get_capture()
            cv_img = capture.color[:, :, :3]
            
            # send captured image
            _, buf = cv2.imencode('.png', cv_img) # , [cv2.IMWRITE_PNG_STRATEGY_DEFAULT]
            data = np.array(buf)
            #
            b64_data = base64.b64encode(data)
            length = str(len(b64_data))
            cli_sock.sendall(length.encode('utf-8').ljust(64))
            cli_sock.sendall(b64_data)
            # rospy.loginfo('Wait for the server')

            # get action
            data = cli_sock.recv(1024)
            str_data = data.decode()
            action = str_data.strip().split(';')
            action = [float(v) for v in action]
            if sum(action) != 0:
                rc.arm.move_gripper(action)    
            else:
                rc.arm.to_home_pose()
                time.sleep(1.0)
                rc.arm.finger_open()
                time.sleep(1.0)

        except: #KeyboardInterrupt
            rc.camera.stop()
            break


if __name__ == '__main__':
	main()
