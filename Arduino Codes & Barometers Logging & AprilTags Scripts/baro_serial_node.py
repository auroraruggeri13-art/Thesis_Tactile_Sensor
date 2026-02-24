#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple ROS node for Arduino barometer (MKR Zero + DPS310)

Arduino now prints:
    Time(ms),B1_P,B1_T,B2_P,B2_T,B3_P,B3_T,B4_P,B4_T,B5_P,B5_T,B6_P,B6_T
    723456,1019.98,24.5,1020.12,24.6,...

This node:
    - Opens /dev/ttyACM* at 115200 baud
    - Sends 'S' handshake to start the Arduino stream
    - Skips Arduino's own header line
    - Logs to a text file with header:

      PcTime,Epoch_s,Time_ms,
             b1_P,b1_T,b2_P,b2_T,b3_P,b3_T,
             b4_P,b4_T,b5_P,b5_T,b6_P,b6_T

    - Publishes the raw Arduino data line on /baro6_raw (std_msgs/String)

You can choose the output filename from the terminal with:
    _filename:=barometers_trialXXXX.txt
"""

import os
import datetime
import rospy
import serial
from std_msgs.msg import String


def main():
    rospy.init_node("baro_serial_node")

    # -----------------------------
    # 1) ROS parameters
    # -----------------------------
    port = rospy.get_param("~port", "/dev/ttyACM0")
    baud = int(rospy.get_param("~baud", 115200))
    save_dir = rospy.get_param("~save_dir", "/catkin_ws/data/aurora/test")
    topic_name = rospy.get_param("~topic_name", "/baro6_raw")
    filename = rospy.get_param("~filename", "")

    # -----------------------------
    # 2) Prepare output folder
    # -----------------------------
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if filename == "":
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "datalog_%s.txt" % ts

    out_path = os.path.join(save_dir, filename)

    # -----------------------------
    # 3) Open serial port
    # -----------------------------
    rospy.loginfo("Opening %s @ %d baud", port, baud)
    ser = serial.Serial(port, baudrate=baud, timeout=0.1)

    # Send handshake to Arduino (so it starts streaming)
    try:
        ser.write(b'S')
        ser.flush()
        rospy.loginfo("Handshake 'S' sent to Arduino.")
    except Exception as e:
        rospy.logwarn("Handshake failed: %s", e)

    # -----------------------------
    # 4) Open log file
    # -----------------------------
    rospy.loginfo("Saving Arduino data to %s", out_path)
    f = open(out_path, "w", buffering=1)

    # New header matching the new Arduino format (pressure + temperature)
    f.write(
        "PcTime,Epoch_s,Time_ms,"
        "b1_P,b1_T,b2_P,b2_T,b3_P,b3_T,"
        "b4_P,b4_T,b5_P,b5_T,b6_P,b6_T\n"
    )

    # -----------------------------
    # 5) ROS publisher
    # -----------------------------
    pub = rospy.Publisher(topic_name, String, queue_size=100)
    rospy.loginfo("baro_serial_node is running. Press Ctrl+C to stop.")

    try:
        # -----------------------------
        # 6) Main loop
        # -----------------------------
        while not rospy.is_shutdown():
            line = ser.readline()
            if not line:
                continue

            # Decode bytes -> string
            try:
                line = line.decode("utf-8", "ignore").strip()
            except Exception:
                continue

            if not line:
                continue

            # Skip Arduino's own header line: "Time(ms),B1_P,B1_T,..."
            if line.lower().startswith("time(") or line.lower().startswith("time(ms)"):
                rospy.loginfo("Skipping Arduino header line: %s", line)
                continue

            # Split the Arduino line
            parts = [p.strip() for p in line.split(",")]

            # Expect: 1 Time_ms + 12 values (B1_P,B1_T,...,B6_P,B6_T) = 13 columns
            if len(parts) != 13:
                rospy.logwarn_throttle(
                    5.0,
                    "Unexpected Arduino line format (got %d cols, expected 13): %s",
                    len(parts), line
                )
                continue

            time_ms = parts[0]
            baro_vals = parts[1:]  # B1_P,B1_T,...,B6_P,B6_T

            # PC timestamps
            now = datetime.datetime.now()
            pc_ts = now.strftime("%H:%M:%S.%f")[:-3]  # e.g. 15:22:40.669
            epoch_s = (now - datetime.datetime(1970, 1, 1)).total_seconds()

            # Build row for file
            row = "%s,%.6f,%s,%s\n" % (pc_ts, epoch_s, time_ms, ",".join(baro_vals))
            f.write(row)

            # Publish raw Arduino line on /baro6_raw
            pub.publish(line)

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        rospy.loginfo("Stopping baro_serial_node...")
        try:
            f.close()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

