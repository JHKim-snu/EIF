import sys
import socket
import cv2
import numpy as np
import base64

import time
from grasp_2dbbox import realsense, grasp

HOST = "147.47.200.155"
PORT = 9998


def main():

    cli_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli_sock.connect((HOST, PORT))
    # rospy.init_node('visual_grounding_client', anonymous=False, disable_signals=True)

    agent = grasp()
    camera = realsense(1280, 720, 30)
    save_image_path = "./realsense.png"
    idx = 500
    while 1:
        try:
            print("To Camera Pose")
            agent.to_camera_pose()
            time.sleep(2.0)
            input("Place Things and Get Visual Info (Press any key)")
            # Take visual info and send image to the server
            frames = camera.pipe.wait_for_frames()
            frames = camera.align.process(frames)
            aligned_depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Send Captured Image
            retval, buf = cv2.imencode(".png", color_image)
            data = np.array(buf)
            b64_data = base64.b64encode(data)
            length = str(len(b64_data))
            cli_sock.sendall(length.encode("utf-8").ljust(64))
            cli_sock.sendall(b64_data)
            print("Wait for the server")

            # Receive bounding box Info
            data = cli_sock.recv(1024)
            str_data = data.decode()
            bbox = str_data.strip().split(";")
            TL_X = int(bbox[0])
            TL_Y = int(bbox[1])
            BR_X = int(bbox[2])
            BR_Y = int(bbox[3])
            PL_X = int(bbox[4])
            PL_Y = int(bbox[5])
            # Visualize received bbox on input image
            # cv_img = cv2.rectangle(cv_img, (TL_X, TL_Y), (BR_X, BR_Y), (0, 0, 255))
            # cv_img = cv2.circle(cv_img, (PL_X, PL_Y), 8, (0, 255, 0), -1)
            # cv2.imwrite('result/vg_{}.png'.format(idx), cv_img)
            print("Pick bbox: {} {} {} {}".format(TL_X, TL_Y, BR_X, BR_Y))
            print("Place coordinate: {} {}".format(PL_X, PL_Y))
            # Manipulation (Pick-n-Place)

            point = [int((TL_X + BR_X) / 2), int((TL_Y + BR_Y) / 2)]
            print("target pick point: {}".format(point))
            print("image resolusion: {}".format(color_image.shape))
            tx, ty, tz = (0, 0, 0)
            while tx == 0 and ty == 0 and tz == 0:
                cv2.circle(
                    color_image, (int(point[0]), int(point[1])), 20, (0, 0, 255), 3
                )
                cv2.imwrite(save_image_path, color_image)

                tx, ty, tz = camera.convert_pixel_to_3d_world(
                    aligned_depth_frame, point[0], point[1]
                )

                print("point based on the camera is ({},{},{})".format(tx, ty, tz))

            agent.pick(x=tx, y=ty, z=tz)

            agent.to_camera_pose()
            idx += 1
            value = input("Continue? [y/n]")
            if value != "y":
                break
        except KeyboardInterrupt:
            print("\nClient Ctrl-c")
            break
        except IOError and ValueError:
            print("\nServer closed")
            break

    cli_sock.close()
    return


if __name__ == "__main__":

    main()
