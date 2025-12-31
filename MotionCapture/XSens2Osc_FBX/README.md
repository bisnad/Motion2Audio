## AI-Toolbox - Motion Utilities - XSens2Osc

![XSens2Osc](data/media/XSens2Osc.JPG)

Figure 1: The figure shows a screenshot of the XSens2Osc software.

### Summary

XSens2Osc is a small C++-based software that receives motion capture data from the [XSens MVN](https://www.movella.com/products/motion-capture/mvn-analyze) Software and forwards this data as [OSC](https://en.wikipedia.org/wiki/Open_Sound_Control) messages. 

### Installation

For simply executing the software, no installation is required. The software runs on any Windows 10 or 11 operating system. If the user wants to compile the software from source, both a C++ IDE such as [Visual Studio](https://visualstudio.microsoft.com/vs/community/) and the [openFrameworks](https://openframeworks.cc/) creative coding environment need to be installed beforehand. Installation instructions for Visual Studio and openFrameworks are available in the [AI Toolbox github repository](https://github.com/bisnad/AIToolbox). 

The software can be downloaded by cloning the [MotionUtilities Github repository](https://github.com/bisnad/MotionUtilities). After cloning, the software is located in the MotionUtilities / XSens2Osc directory.

### Directory Structure

- XSens2Osc (contains theVisual Studio project file)

  - data 
    - media (contains media used in this Readme)

  - bin (contains the software  executable and dynamic libraries)
    - data (contains a configuration file that specifies several software settings)
  - src (contains the source code files)

### Usage

#### Start

The software can be started by double clicking the xsens2osc.exe file. During startup, the software reads a configuration file entitled "config.json". This file defines the addresses and ports for sending and receiving OSC messages. 

#### Functionality

The software receives mocap data through network streaming from the XSens MVN software and forwards this data via OSC. The software supports for an arbitrary number of performers the following type of mocap data: 3D joint positions, joint rotations as Euler angles, joint rotations as quaternions, linear and angular joint velocities and accelerations, and raw inertial sensor data. In order for the software to forward this data, the network streamer options in the MVN  software have to be set accordingly. Figure 2 depicts a screenshot of the network streaming options in the MVN software with all mocap data types that are supported by the XSens2Osc software selected. To avoid overloading XSens2Osc with data, it can be helpful to limit the stream rate to a Maximum of 60Hz. The stream rate can also be set in the network streamer options in the MVN   software. 

![XSens_MVN_network_streamer_window](data/media/XSens_MVN_network_streamer_window.JPG)

Figure 2: The figure shows a screenshot of the Network Streamer Options Window in the XSens MVN software (Analyze Pro 2023).

#### Graphical User Interface

The software possesses a minimal graphical user interface that displays the outgoing OSC messages. For each OSC message, the message address and the first three values are shown. 

### OSC Communication

The software sends only OSC messages for those mocap data that has been selected for sending in the XSens MVN Network Streamer Options window. For each mocap data type that is selected for sending, the resulting OSC messages contain for each performer all data grouped together. For example, an OSC message that sends the joint positions of a performer as 3D vectors in world coordinates looks like this (with N representing the number of joints): `/mocap/0/joint/pos_world <float j1x> <float j1y> <float j1z> .... <float jNx> <float jNy> <float jNz>` . Similarity, an OSC message that sends for example the rotations of all sensors worn by a performer as quaternions in world coordinates looks like this (with N representing the number of sensors):`/mocap/skelID/tracker/rot_world <float s1w> <float s1x> <float s1y> <float s1z> .... <float sNw> <float sNx> <float sNy> <float sNz>` 

The following sections list for each type of motion data that can be selected in the XSens MVN Network Streamer Options window and that is supported by the Xsens2Osc software the resulting OSC messages. 

##### Position + Orientation (Quaternion)

- joint positions as list of 3D vectors in world coordinates: `/mocap/0/joint/pos_world <float j1x> <float j1y> <float j1z> .... <float jNx> <float jNy> <float jNz>` 
- joint rotations as list of Quaternions in world coordinates: `/mocap/0/joint/rot_world <float j1w> <float j1x> <float j1y> <float j1z> .... <float jNw> <float jNx> <float jNy> <float jNz>` 

##### Position+ Orientation (Euler)

- joint positions as list of 3D vectors in world coordinates: `/mocap/0/joint/pos_world <float j1x> <float j1y> <float j1z> .... <float jNx> <float jNy> <float jNz>` 
- joint rotations as list of Euler values in world coordinates: `/mocap/0/joint/rot_world_euler <float j1x> <float j1y> <float j1z> .... <float jNx> <float jNy> <float jNz>` 

##### Linear Segment Kinematics

- joint positions as list of 3D vectors in world coordinates: `/mocap/0/joint/pos_world <float j1x> <float j1y> <float j1z> .... <float jNx> <float jNy> <float jNz>` 
- joint linear velocities as list of 3D vectors in world coordinates: `/mocap/0/joint/lin_vel <float j1x> <float j1y> <float j1z> .... <float jNx> <float jNy> <float jNz>` 
- joint linear accelerations as list of 3D vectors in world coordinates: `/mocap/0/joint/lin_acc <float j1x> <float j1y> <float j1z> .... <float jNx> <float jNy> <float jNz>` 

##### Angular Segment Kinematics

- joint rotations as list of Quaternions in world coordinates: `/mocap/0/joint/rot_world <float j1w> <float j1x> <float j1y> <float j1z> .... <float jNw> <float jNx> <float jNy> <float jNz>` 
- joint rotational velocities as list of 3D vectors in world coordinates: `/mocap/0/joint/rot_vel <float j1x> <float j1y> <float j1z> .... <float jNx> <float jNy> <float jNz>` 
- joint rotational accelerations as list of 3D vectors in world coordinates: `/mocap/0/joint/rot_acc <float j1x> <float j1y> <float j1z> .... <float jNx> <float jNy> <float jNz>` 

##### Tracker Kinematics

- tracker rotations as list of Quaternions: `/mocap/0/tracker/rot_world <float j1w> <float j1x> <float j1y> <float j1z> .... <float jNw> <float jNx> <float jNy> <float jNz>` 
- tracker free accelerations as list of 3D vectors: `/mocap/0/tracker/accel_world <float j1x> <float j1y> <float j1z> ....  <float jNx> <float jNy> <float jNz>` 
- tracker magnetic field values as list of 3D vectors: `/mocap/0/tracker/magnet <float j1x> <float j1y> <float j1z> ....  <float jNx> <float jNy> <float jNz>` 

### Limitations

XSens2Osc doesn't provide a GUI for changing the network address and port address for sending OSC messages to. These settings need to be changed by editing the "config.json" file in the bin/data/ Folder.

XSens2OSC collapses at mocap frames rates higher than 60 Hz.

### Dependencies

To compile the XSens2Osc tool from source, one additional Addon that is not part of the openFrameworks default distribution need to be present in the addons directory of openFrameworks. This addon can be downloaded from its own dedicated online repository:

- [ofxDabBase](https://github.com/bisnad/ofxDabBase) 

  ofxDabBase provides some basic functionality in the form of classes that deal with multidimensional data, text file parsing, and information flow. 

