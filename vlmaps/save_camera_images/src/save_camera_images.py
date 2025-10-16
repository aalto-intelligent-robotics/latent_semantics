#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def image_message_to_cv_image(image_message):
    # Converts the image from the message to OpenCV format
    # http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding="passthrough")

    return cv_image


def save_image(image_message, directory):
    image = image_message_to_cv_image(image_message)

    filename = "{}/{}_{}_{}.jpg".format(
        directory, image_message.header.stamp, image_message.height, image_message.width
    )
    rospy.loginfo("Writing to file " + filename)

    success = cv2.imwrite(filename, image)
    rospy.loginfo("Image written:" + str(success))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        help="Directory in which the images need to be saved.",
        required=True,
    )
    args = parser.parse_args()
    return args


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + " Image: %s", data.header.stamp)

    save_image(data, parse_args().img_dir)


def main():
    rospy.init_node("save_camera_images", anonymous=True)
    rospy.Subscriber("/sensorring_cam3d/rgb/image_raw", Image, callback)
    rospy.spin()


if __name__ == "__main__":
    main()
