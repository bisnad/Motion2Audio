#include "dab_xsens_stream_manager.h"
#include "dab_xsens_mocap_skeleton.h"

using namespace dab;
using namespace dab::xsens;

const bool StreamManager::sBigEndian = true;
const unsigned int StreamManager::sTrackerCount = 17; // TODO: likely also deal with versions that include gloves
const unsigned int StreamManager::sSegmentCount = 23; // TODO: likely also deal with versions that include gloves
const unsigned int StreamManager::sDatagramMaxSize = 2000;
const unsigned int StreamManager::sDatagramHeaderSize = 24;
const unsigned int StreamManager::sDatagramPoseDataEulerSizePerSegment = 28;
const unsigned int StreamManager::sDatagramPoseDataQuaternionSizePerSegment = 32;
const unsigned int StreamManager::sDatagramLinearKinematicsSizePerSegment = 40;
const unsigned int StreamManager::sDatagramAngularKinematicsSizePerSegment = 44;
const unsigned int StreamManager::sDatagramMotionTrackerKinematicsSizePerTracker = 68;
const unsigned int StreamManager::sDatagramJointAnglesSizePerSegment = 20;

StreamManager::StreamManager()
{}

StreamManager::~StreamManager()
{
	mUDPReceiver.Close();
}

void 
StreamManager::setupUDPReceiver(unsigned int pPort) throw (dab::Exception)
{
	mReceivePort = pPort;

	ofxUDPSettings settings;
	settings.receiveOn(mReceivePort);
	settings.blocking = false;

	bool success;

	success = mUDPReceiver.Setup(settings);

	if (success == false) throw dab::Exception("Network Error: failed to open udp receiver on port " + std::to_string(mReceivePort), __FILE__, __FUNCTION__, __LINE__);
}

void 
StreamManager::update() throw (dab::Exception)
{
	char udpMessage[sDatagramMaxSize];
	int rec = mUDPReceiver.Receive(udpMessage, sDatagramMaxSize);

	if (rec == 0 || rec == SOCKET_TIMEOUT || rec == SOCKET_ERROR) return;

	std::string message_type = std::string({ udpMessage[4], udpMessage[5] });

	//std::cout << "message_type " << message_type << "\n";

	try
	{
		unsigned char* datagram = (unsigned char*)(&udpMessage);

		//std::cout << "message_type " << message_type << "\n";

		if (message_type == "01") updatePoseDataEuler(datagram);
		else if (message_type == "02") updatePoseDataQuaternion(datagram);
		else if (message_type == "20") updateJointAngles(datagram);
		else if (message_type == "21") updateLinearSegmentKinematics(datagram);
		else if (message_type == "22") updateAngularSegmentKinematics(datagram);
		else if (message_type == "23") updateMotionTrackerKinematics(datagram);
		//else throw dab::Exception("Network Error: message type " + message_type + " not supported\n", __FILE__, __FUNCTION__, __LINE__);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("Network Error: failed to update stream manager\n", __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void 
StreamManager::updatePoseDataEuler(unsigned char* pDatagram) throw (dab::Exception)
{
	uint8_t character_id = pDatagram[16];

	std::vector<float> segmentPositions(sSegmentCount * 3);
	std::vector<float> segmentRotationsEuler(sSegmentCount * 3);

	float segment_position_x;
	float segment_position_y;
	float segment_position_z;
	float segment_rotation_euler_x;
	float segment_rotation_euler_y;
	float segment_rotation_euler_z;

	if (sBigEndian == true)
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sSegmentCount; mI += sDatagramPoseDataEulerSizePerSegment, sI += 1)
		{
			unsigned char segment_position_x_bytes[] = { pDatagram[mI + 7], pDatagram[mI + 6], pDatagram[mI + 5], pDatagram[mI + 4] };
			unsigned char segment_position_y_bytes[] = { pDatagram[mI + 11], pDatagram[mI + 10], pDatagram[mI + 9], pDatagram[mI + 8] };
			unsigned char segment_position_z_bytes[] = { pDatagram[mI + 15], pDatagram[mI + 14], pDatagram[mI + 13], pDatagram[mI + 12] };
			unsigned char segment_rotation_euler_x_bytes[] = { pDatagram[mI + 19], pDatagram[mI + 18], pDatagram[mI + 17], pDatagram[mI + 16] };
			unsigned char segment_rotation_euler_y_bytes[] = { pDatagram[mI + 23], pDatagram[mI + 22], pDatagram[mI + 21], pDatagram[mI + 20] };
			unsigned char segment_rotation_euler_z_bytes[] = { pDatagram[mI + 27], pDatagram[mI + 26], pDatagram[mI + 25], pDatagram[mI + 24] };

			memcpy(&segment_position_x, segment_position_x_bytes, sizeof(float));
			memcpy(&segment_position_y, segment_position_y_bytes, sizeof(float));
			memcpy(&segment_position_z, segment_position_z_bytes, sizeof(float));
			memcpy(&segment_rotation_euler_x, segment_rotation_euler_x_bytes, sizeof(float));
			memcpy(&segment_rotation_euler_y, segment_rotation_euler_y_bytes, sizeof(float));
			memcpy(&segment_rotation_euler_z, segment_rotation_euler_z_bytes, sizeof(float));

			//segmentPositions[sI * 3] = segment_position_x;
			//segmentPositions[sI * 3 + 1] = segment_position_y;
			//segmentPositions[sI * 3 + 2] = segment_position_z;

			// The euler version employs a different coordinate system than the quaternion version
			// I change here from euler to quaternion convention
			// since the quaternion convention is also the one used for the kinematics
			segmentPositions[sI * 3] = segment_position_z / 100.0;
			segmentPositions[sI * 3 + 1] = segment_position_x / 100.0;
			segmentPositions[sI * 3 + 2] = segment_position_y / 100.0;
			segmentRotationsEuler[sI * 3] = segment_rotation_euler_x;
			segmentRotationsEuler[sI * 3 + 1] = segment_rotation_euler_y;
			segmentRotationsEuler[sI * 3 + 2] = segment_rotation_euler_z;
		}
	}
	else
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sSegmentCount; mI += sDatagramPoseDataEulerSizePerSegment, sI += 1)
		{
			unsigned char segment_position_x_bytes[] = { pDatagram[mI + 4], pDatagram[mI + 5], pDatagram[mI + 6], pDatagram[mI + 7] };
			unsigned char segment_position_y_bytes[] = { pDatagram[mI + 8], pDatagram[mI + 9], pDatagram[mI + 10], pDatagram[mI + 11] };
			unsigned char segment_position_z_bytes[] = { pDatagram[mI + 12], pDatagram[mI + 13], pDatagram[mI + 14], pDatagram[mI + 15] };
			unsigned char segment_rotation_euler_x_bytes[] = { pDatagram[mI + 16], pDatagram[mI + 17], pDatagram[mI + 18], pDatagram[mI + 19] };
			unsigned char segment_rotation_euler_y_bytes[] = { pDatagram[mI + 20], pDatagram[mI + 21], pDatagram[mI + 22], pDatagram[mI + 23] };
			unsigned char segment_rotation_euler_z_bytes[] = { pDatagram[mI + 24], pDatagram[mI + 25], pDatagram[mI + 26], pDatagram[mI + 27] };

			memcpy(&segment_position_x, segment_position_x_bytes, sizeof(float));
			memcpy(&segment_position_y, segment_position_y_bytes, sizeof(float));
			memcpy(&segment_position_z, segment_position_z_bytes, sizeof(float));
			memcpy(&segment_rotation_euler_x, segment_rotation_euler_x_bytes, sizeof(float));
			memcpy(&segment_rotation_euler_y, segment_rotation_euler_y_bytes, sizeof(float));
			memcpy(&segment_rotation_euler_z, segment_rotation_euler_z_bytes, sizeof(float));

			//segmentPositions[sI * 3] = segment_position_x;
			//segmentPositions[sI * 3 + 1] = segment_position_y;
			//segmentPositions[sI * 3 + 2] = segment_position_z;
			segmentPositions[sI * 3] = segment_position_z;
			segmentPositions[sI * 3 + 1] = segment_position_x;
			segmentPositions[sI * 3 + 2] = segment_position_y;
			segmentRotationsEuler[sI * 3] = segment_rotation_euler_x;
			segmentRotationsEuler[sI * 3 + 1] = segment_rotation_euler_y;
			segmentRotationsEuler[sI * 3 + 2] = segment_rotation_euler_z;
		}
	}


	try
	{
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/pos_world", segmentPositions);
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/rot_world_euler", segmentRotationsEuler);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("Network Error: failed to update pose data euler\n", __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void 
StreamManager::updatePoseDataQuaternion(unsigned char* pDatagram) throw (dab::Exception)
{
	uint8_t character_id = pDatagram[16];

	std::vector<float> segmentPositions(sSegmentCount * 3);
	std::vector<float> segmentRotationsQuat(sSegmentCount * 4);

	float segment_position_x;
	float segment_position_y;
	float segment_position_z;
	float segment_rotation_quat_1;
	float segment_rotation_quat_2;
	float segment_rotation_quat_3;
	float segment_rotation_quat_4;

	if (sBigEndian == true)
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sSegmentCount; mI += sDatagramPoseDataQuaternionSizePerSegment, sI += 1)
		{
			unsigned char segment_position_x_bytes[] = { pDatagram[mI + 7], pDatagram[mI + 6], pDatagram[mI + 5], pDatagram[mI + 4] };
			unsigned char segment_position_y_bytes[] = { pDatagram[mI + 11], pDatagram[mI + 10], pDatagram[mI + 9], pDatagram[mI + 8] };
			unsigned char segment_position_z_bytes[] = { pDatagram[mI + 15], pDatagram[mI + 14], pDatagram[mI + 13], pDatagram[mI + 12] };
			unsigned char segment_rotation_quat_1_bytes[] = { pDatagram[mI + 19], pDatagram[mI + 18], pDatagram[mI + 17], pDatagram[mI + 16] };
			unsigned char segment_rotation_quat_2_bytes[] = { pDatagram[mI + 23], pDatagram[mI + 22], pDatagram[mI + 21], pDatagram[mI + 20] };
			unsigned char segment_rotation_quat_3_bytes[] = { pDatagram[mI + 27], pDatagram[mI + 26], pDatagram[mI + 25], pDatagram[mI + 24] };
			unsigned char segment_rotation_quat_4_bytes[] = { pDatagram[mI + 31], pDatagram[mI + 30], pDatagram[mI + 29], pDatagram[mI + 28] };

			memcpy(&segment_position_x, segment_position_x_bytes, sizeof(float));
			memcpy(&segment_position_y, segment_position_y_bytes, sizeof(float));
			memcpy(&segment_position_z, segment_position_z_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_1, segment_rotation_quat_1_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_2, segment_rotation_quat_2_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_3, segment_rotation_quat_3_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_4, segment_rotation_quat_4_bytes, sizeof(float));

			segmentPositions[sI * 3] = segment_position_x;
			segmentPositions[sI * 3 + 1] = segment_position_y;
			segmentPositions[sI * 3 + 2] = segment_position_z;
			//segmentPositions[sI * 3] = segment_position_x * 100.0;
			//segmentPositions[sI * 3 + 1] = segment_position_y * 100.0;
			//segmentPositions[sI * 3 + 2] = segment_position_z * 100.0;
			segmentRotationsQuat[sI * 4] = segment_rotation_quat_1;
			segmentRotationsQuat[sI * 4 + 1] = segment_rotation_quat_2;
			segmentRotationsQuat[sI * 4 + 2] = segment_rotation_quat_3;
			segmentRotationsQuat[sI * 4 + 3] = segment_rotation_quat_4;
		}
	}
	else
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sSegmentCount; mI += sDatagramPoseDataQuaternionSizePerSegment, sI += 1)
		{
			unsigned char segment_position_x_bytes[] = { pDatagram[mI + 4], pDatagram[mI + 5], pDatagram[mI + 6], pDatagram[mI + 7] };
			unsigned char segment_position_y_bytes[] = { pDatagram[mI + 8], pDatagram[mI + 9], pDatagram[mI + 10], pDatagram[mI + 11] };
			unsigned char segment_position_z_bytes[] = { pDatagram[mI + 12], pDatagram[mI + 13], pDatagram[mI + 14], pDatagram[mI + 15] };
			unsigned char segment_rotation_quat_1_bytes[] = { pDatagram[mI + 16], pDatagram[mI + 17], pDatagram[mI + 18], pDatagram[mI + 19] };
			unsigned char segment_rotation_quat_2_bytes[] = { pDatagram[mI + 20], pDatagram[mI + 21], pDatagram[mI + 22], pDatagram[mI + 23] };
			unsigned char segment_rotation_quat_3_bytes[] = { pDatagram[mI + 24], pDatagram[mI + 25], pDatagram[mI + 26], pDatagram[mI + 27] };
			unsigned char segment_rotation_quat_4_bytes[] = { pDatagram[mI + 28], pDatagram[mI + 29], pDatagram[mI + 30], pDatagram[mI + 31] };

			memcpy(&segment_position_x, segment_position_x_bytes, sizeof(float));
			memcpy(&segment_position_y, segment_position_y_bytes, sizeof(float));
			memcpy(&segment_position_z, segment_position_z_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_1, segment_rotation_quat_1_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_2, segment_rotation_quat_2_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_3, segment_rotation_quat_3_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_4, segment_rotation_quat_4_bytes, sizeof(float));

			segmentPositions[sI * 3] = segment_position_x;
			segmentPositions[sI * 3 + 1] = segment_position_y;
			segmentPositions[sI * 3 + 2] = segment_position_z;
			segmentRotationsQuat[sI * 4] = segment_rotation_quat_1;
			segmentRotationsQuat[sI * 4 + 1] = segment_rotation_quat_2;
			segmentRotationsQuat[sI * 4 + 2] = segment_rotation_quat_3;
			segmentRotationsQuat[sI * 4 + 3] = segment_rotation_quat_4;
		}
	}

	try
	{
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/pos_world", segmentPositions);
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/rot_world", segmentRotationsQuat);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("Network Error: failed to update pose data euler\n", __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void 
StreamManager::updateLinearSegmentKinematics(unsigned char* pDatagram) throw (dab::Exception)
{
	uint8_t character_id = pDatagram[16];

	std::vector<float> segmentPositions(sSegmentCount * 3);
	std::vector<float> segmentVelocities(sSegmentCount * 3);
	std::vector<float> segmentAccelerations(sSegmentCount * 3);

	float segment_position_x;
	float segment_position_y;
	float segment_position_z;
	float segment_velocity_x;
	float segment_velocity_y;
	float segment_velocity_z;
	float segment_acceleration_x;
	float segment_acceleration_y;
	float segment_acceleration_z;

	if (sBigEndian == true)
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sSegmentCount; mI += sDatagramLinearKinematicsSizePerSegment, sI += 1)
		{
			unsigned char segment_position_x_bytes[] = { pDatagram[mI + 7], pDatagram[mI + 6], pDatagram[mI + 5], pDatagram[mI + 4] };
			unsigned char segment_position_y_bytes[] = { pDatagram[mI + 11], pDatagram[mI + 10], pDatagram[mI + 9], pDatagram[mI + 8] };
			unsigned char segment_position_z_bytes[] = { pDatagram[mI + 15], pDatagram[mI + 14], pDatagram[mI + 13], pDatagram[mI + 12] };
			unsigned char segment_velocity_x_bytes[] = { pDatagram[mI + 19], pDatagram[mI + 18], pDatagram[mI + 16], pDatagram[mI + 16] };
			unsigned char segment_velocity_y_bytes[] = { pDatagram[mI + 23], pDatagram[mI + 22], pDatagram[mI + 21], pDatagram[mI + 20] };
			unsigned char segment_velocity_z_bytes[] = { pDatagram[mI + 27], pDatagram[mI + 26], pDatagram[mI + 25], pDatagram[mI + 24] };
			unsigned char segment_acceleration_x_bytes[] = { pDatagram[mI + 31], pDatagram[mI + 30], pDatagram[mI + 29], pDatagram[mI + 28] };
			unsigned char segment_acceleration_y_bytes[] = { pDatagram[mI + 35], pDatagram[mI + 34], pDatagram[mI + 33], pDatagram[mI + 32] };
			unsigned char segment_acceleration_z_bytes[] = { pDatagram[mI + 39], pDatagram[mI + 38], pDatagram[mI + 37], pDatagram[mI + 36] };

			memcpy(&segment_position_x, segment_position_x_bytes, sizeof(float));
			memcpy(&segment_position_y, segment_position_y_bytes, sizeof(float));
			memcpy(&segment_position_z, segment_position_z_bytes, sizeof(float));
			memcpy(&segment_velocity_x, segment_velocity_x_bytes, sizeof(float));
			memcpy(&segment_velocity_y, segment_velocity_y_bytes, sizeof(float));
			memcpy(&segment_velocity_z, segment_velocity_z_bytes, sizeof(float));
			memcpy(&segment_acceleration_x, segment_acceleration_x_bytes, sizeof(float));
			memcpy(&segment_acceleration_y, segment_acceleration_y_bytes, sizeof(float));
			memcpy(&segment_acceleration_z, segment_acceleration_z_bytes, sizeof(float));

			segmentPositions[sI * 3] = segment_position_x;
			segmentPositions[sI * 3 + 1] = segment_position_y;
			segmentPositions[sI * 3 + 2] = segment_position_z;
			segmentVelocities[sI * 3] = segment_velocity_x;
			segmentVelocities[sI * 3 + 1] = segment_velocity_y;
			segmentVelocities[sI * 3 + 2] = segment_velocity_z;
			segmentAccelerations[sI * 3] = segment_acceleration_x;
			segmentAccelerations[sI * 3 + 1] = segment_acceleration_y;
			segmentAccelerations[sI * 3 + 2] = segment_acceleration_z;
		}
	}
	else
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sSegmentCount; mI += sDatagramLinearKinematicsSizePerSegment, sI += 1)
		{
			unsigned char segment_position_x_bytes[] = { pDatagram[mI + 4], pDatagram[mI + 5], pDatagram[mI + 6], pDatagram[mI + 7] };
			unsigned char segment_position_y_bytes[] = { pDatagram[mI + 8], pDatagram[mI + 9], pDatagram[mI + 10], pDatagram[mI + 11] };
			unsigned char segment_position_z_bytes[] = { pDatagram[mI + 12], pDatagram[mI + 13], pDatagram[mI + 14], pDatagram[mI + 15] };
			unsigned char segment_velocity_x_bytes[] = { pDatagram[mI + 16], pDatagram[mI + 17], pDatagram[mI + 18], pDatagram[mI + 19] };
			unsigned char segment_velocity_y_bytes[] = { pDatagram[mI + 20], pDatagram[mI + 21], pDatagram[mI + 22], pDatagram[mI + 23] };
			unsigned char segment_velocity_z_bytes[] = { pDatagram[mI + 24], pDatagram[mI + 25], pDatagram[mI + 26], pDatagram[mI + 27] };
			unsigned char segment_acceleration_x_bytes[] = { pDatagram[mI + 28], pDatagram[mI + 29], pDatagram[mI + 30], pDatagram[mI + 31] };
			unsigned char segment_acceleration_y_bytes[] = { pDatagram[mI + 32], pDatagram[mI + 33], pDatagram[mI + 34], pDatagram[mI + 35] };
			unsigned char segment_acceleration_z_bytes[] = { pDatagram[mI + 36], pDatagram[mI + 37], pDatagram[mI + 38], pDatagram[mI + 38] };

			memcpy(&segment_position_x, segment_position_x_bytes, sizeof(float));
			memcpy(&segment_position_y, segment_position_y_bytes, sizeof(float));
			memcpy(&segment_position_z, segment_position_z_bytes, sizeof(float));
			memcpy(&segment_velocity_x, segment_velocity_x_bytes, sizeof(float));
			memcpy(&segment_velocity_y, segment_velocity_y_bytes, sizeof(float));
			memcpy(&segment_velocity_z, segment_velocity_z_bytes, sizeof(float));
			memcpy(&segment_acceleration_x, segment_acceleration_x_bytes, sizeof(float));
			memcpy(&segment_acceleration_y, segment_acceleration_y_bytes, sizeof(float));
			memcpy(&segment_acceleration_z, segment_acceleration_z_bytes, sizeof(float));

			segmentPositions[sI * 3] = segment_position_x;
			segmentPositions[sI * 3 + 1] = segment_position_y;
			segmentPositions[sI * 3 + 2] = segment_position_z;
			segmentVelocities[sI * 3] = segment_velocity_x;
			segmentVelocities[sI * 3 + 1] = segment_velocity_y;
			segmentVelocities[sI * 3 + 2] = segment_velocity_z;
			segmentAccelerations[sI * 3] = segment_acceleration_x;
			segmentAccelerations[sI * 3 + 1] = segment_acceleration_y;
			segmentAccelerations[sI * 3 + 2] = segment_acceleration_z;
		}
	}

	try
	{
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/pos_world", segmentPositions);
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/lin_vel", segmentVelocities);
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/lin_acc", segmentAccelerations);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("Network Error: failed to update linear segment kinematics\n", __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void 
StreamManager::updateAngularSegmentKinematics(unsigned char* pDatagram) throw (dab::Exception)
{
	uint8_t character_id = pDatagram[16];

	std::vector<float> segmentRotationsQuat(sSegmentCount * 4);
	std::vector<float> segmentVelocities(sSegmentCount * 3);
	std::vector<float> segmentAccelerations(sSegmentCount * 3);

	//int segment_id;
	float segment_rotation_quat_1;
	float segment_rotation_quat_2;
	float segment_rotation_quat_3;
	float segment_rotation_quat_4;
	float segment_position_z;
	float segment_velocity_x;
	float segment_velocity_y;
	float segment_velocity_z;
	float segment_acceleration_x;
	float segment_acceleration_y;
	float segment_acceleration_z;

	if (sBigEndian == true)
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sSegmentCount; mI += sDatagramAngularKinematicsSizePerSegment, sI += 1)
		{
			//unsigned char segment_id_bytes[] = { pDatagram[mI + 3], pDatagram[mI + 2], pDatagram[mI + 1], pDatagram[mI + 0] };
			unsigned char segment_rotation_q1_bytes[] = { pDatagram[mI + 7], pDatagram[mI + 6], pDatagram[mI + 5], pDatagram[mI + 4] };
			unsigned char segment_rotation_q2_bytes[] = { pDatagram[mI + 11], pDatagram[mI + 10], pDatagram[mI + 9], pDatagram[mI + 8] };
			unsigned char segment_rotation_q3_bytes[] = { pDatagram[mI + 15], pDatagram[mI + 14], pDatagram[mI + 13], pDatagram[mI + 12] };
			unsigned char segment_rotation_q4_bytes[] = { pDatagram[mI + 19], pDatagram[mI + 18], pDatagram[mI + 17], pDatagram[mI + 16] };
			unsigned char segment_velocity_x_bytes[] = { pDatagram[mI + 23], pDatagram[mI + 22], pDatagram[mI + 21], pDatagram[mI + 20] };
			unsigned char segment_velocity_y_bytes[] = { pDatagram[mI + 27], pDatagram[mI + 26], pDatagram[mI + 25], pDatagram[mI + 24] };
			unsigned char segment_velocity_z_bytes[] = { pDatagram[mI + 31], pDatagram[mI + 30], pDatagram[mI + 29], pDatagram[mI + 28] };
			unsigned char segment_acceleration_x_bytes[] = { pDatagram[mI + 35], pDatagram[mI + 34], pDatagram[mI + 33], pDatagram[mI + 32] };
			unsigned char segment_acceleration_y_bytes[] = { pDatagram[mI + 39], pDatagram[mI + 38], pDatagram[mI + 37], pDatagram[mI + 36] };
			unsigned char segment_acceleration_z_bytes[] = { pDatagram[mI + 43], pDatagram[mI + 42], pDatagram[mI + 41], pDatagram[mI + 40] };

			//memcpy(&segment_id, segment_id_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_1, segment_rotation_q1_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_2, segment_rotation_q2_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_3, segment_rotation_q3_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_4, segment_rotation_q4_bytes, sizeof(float));
			memcpy(&segment_velocity_x, segment_velocity_x_bytes, sizeof(float));
			memcpy(&segment_velocity_y, segment_velocity_y_bytes, sizeof(float));
			memcpy(&segment_velocity_z, segment_velocity_z_bytes, sizeof(float));
			memcpy(&segment_acceleration_x, segment_acceleration_x_bytes, sizeof(float));
			memcpy(&segment_acceleration_y, segment_acceleration_y_bytes, sizeof(float));
			memcpy(&segment_acceleration_z, segment_acceleration_z_bytes, sizeof(float));

			segmentRotationsQuat[sI * 4] = segment_rotation_quat_1;
			segmentRotationsQuat[sI * 4 + 1] = segment_rotation_quat_2;
			segmentRotationsQuat[sI * 4 + 2] = segment_rotation_quat_3;
			segmentRotationsQuat[sI * 4 + 3] = segment_rotation_quat_4;
			segmentVelocities[sI * 3] = segment_velocity_x;
			segmentVelocities[sI * 3 + 1] = segment_velocity_y;
			segmentVelocities[sI * 3 + 2] = segment_velocity_z;
			segmentAccelerations[sI * 3] = segment_acceleration_x;
			segmentAccelerations[sI * 3 + 1] = segment_acceleration_y;
			segmentAccelerations[sI * 3 + 2] = segment_acceleration_z;

			//std::cout << "sI " << sI << " sId " << segment_id << "\n";
		}
	}
	else
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sSegmentCount; mI += sDatagramAngularKinematicsSizePerSegment, sI += 1)
		{
			unsigned char segment_rotation_q1_bytes[] = { pDatagram[mI + 3], pDatagram[mI + 5], pDatagram[mI + 6], pDatagram[mI + 7] };
			unsigned char segment_rotation_q2_bytes[] = { pDatagram[mI + 8], pDatagram[mI + 9], pDatagram[mI + 10], pDatagram[mI + 11] };
			unsigned char segment_rotation_q3_bytes[] = { pDatagram[mI + 12], pDatagram[mI + 13], pDatagram[mI + 14], pDatagram[mI + 15] };
			unsigned char segment_rotation_q4_bytes[] = { pDatagram[mI + 16], pDatagram[mI + 17], pDatagram[mI + 18], pDatagram[mI + 19] };
			unsigned char segment_velocity_x_bytes[] = { pDatagram[mI + 20], pDatagram[mI + 21], pDatagram[mI + 22], pDatagram[mI + 23] };
			unsigned char segment_velocity_y_bytes[] = { pDatagram[mI + 24], pDatagram[mI + 25], pDatagram[mI + 26], pDatagram[mI + 27] };
			unsigned char segment_velocity_z_bytes[] = { pDatagram[mI + 28], pDatagram[mI + 29], pDatagram[mI + 30], pDatagram[mI + 31] };
			unsigned char segment_acceleration_x_bytes[] = { pDatagram[mI + 32], pDatagram[mI + 33], pDatagram[mI + 34], pDatagram[mI + 35] };
			unsigned char segment_acceleration_y_bytes[] = { pDatagram[mI + 36], pDatagram[mI + 37], pDatagram[mI + 38], pDatagram[mI + 39] };
			unsigned char segment_acceleration_z_bytes[] = { pDatagram[mI + 40], pDatagram[mI + 41], pDatagram[mI + 42], pDatagram[mI + 43] };

			memcpy(&segment_rotation_quat_1, segment_rotation_q1_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_2, segment_rotation_q2_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_3, segment_rotation_q3_bytes, sizeof(float));
			memcpy(&segment_rotation_quat_4, segment_rotation_q4_bytes, sizeof(float));
			memcpy(&segment_velocity_x, segment_velocity_x_bytes, sizeof(float));
			memcpy(&segment_velocity_y, segment_velocity_y_bytes, sizeof(float));
			memcpy(&segment_velocity_z, segment_velocity_z_bytes, sizeof(float));
			memcpy(&segment_acceleration_x, segment_acceleration_x_bytes, sizeof(float));
			memcpy(&segment_acceleration_y, segment_acceleration_y_bytes, sizeof(float));
			memcpy(&segment_acceleration_z, segment_acceleration_z_bytes, sizeof(float));

			segmentRotationsQuat[sI * 4] = segment_rotation_quat_1;
			segmentRotationsQuat[sI * 4 + 1] = segment_rotation_quat_2;
			segmentRotationsQuat[sI * 4 + 2] = segment_rotation_quat_3;
			segmentRotationsQuat[sI * 4 + 3] = segment_rotation_quat_4;
			segmentVelocities[sI * 3] = segment_velocity_x;
			segmentVelocities[sI * 3 + 1] = segment_velocity_y;
			segmentVelocities[sI * 3 + 2] = segment_velocity_z;
			segmentAccelerations[sI * 3] = segment_acceleration_x;
			segmentAccelerations[sI * 3 + 1] = segment_acceleration_y;
			segmentAccelerations[sI * 3 + 2] = segment_acceleration_z;
		}
	}

	try
	{
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/rot_world", segmentRotationsQuat);
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/rot_vel", segmentVelocities);
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/rot_acc", segmentAccelerations);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("Network Error: failed to update angular segment kinematics\n", __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void
StreamManager::updateMotionTrackerKinematics(unsigned char* pDatagram) throw (dab::Exception)
{
	uint8_t character_id = pDatagram[16];

	std::vector<float> trackerIds(sTrackerCount);
	std::vector<float> trackerRotationsQuat(sTrackerCount * 4);
	std::vector<float> trackerGlobalAccelerations(sTrackerCount * 3);
	std::vector<float> trackerAccelerations(sTrackerCount * 3);
	std::vector<float> trackerAngularVelocities(sTrackerCount * 3);
	std::vector<float> trackerMagneticFields(sTrackerCount * 3);

	int tracker_id;
	float tracker_rotation_quat_1;
	float tracker_rotation_quat_2;
	float tracker_rotation_quat_3;
	float tracker_rotation_quat_4;
	float tracker_global_acceleration_x;
	float tracker_global_acceleration_y;
	float tracker_global_acceleration_z;
	float tracker_acceleration_x;
	float tracker_acceleration_y;
	float tracker_acceleration_z;
	float tracker_angular_velocity_x;
	float tracker_angular_velocity_y;
	float tracker_angular_velocity_z;
	float tracker_magnetic_field_x;
	float tracker_magnetic_field_y;
	float tracker_magnetic_field_z;

	if (sBigEndian == true)
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sTrackerCount; mI += sDatagramMotionTrackerKinematicsSizePerTracker, sI += 1)
		{
			unsigned char tracker_id_bytes[] = { pDatagram[mI + 3], pDatagram[mI + 2], pDatagram[mI + 1], pDatagram[mI + 0] };
			unsigned char tracker_rotation_q1_bytes[] = { pDatagram[mI + 7], pDatagram[mI + 6], pDatagram[mI + 5], pDatagram[mI + 4] };
			unsigned char tracker_rotation_q2_bytes[] = { pDatagram[mI + 11], pDatagram[mI + 10], pDatagram[mI + 9], pDatagram[mI + 8] };
			unsigned char tracker_rotation_q3_bytes[] = { pDatagram[mI + 15], pDatagram[mI + 14], pDatagram[mI + 13], pDatagram[mI + 12] };
			unsigned char tracker_rotation_q4_bytes[] = { pDatagram[mI + 19], pDatagram[mI + 18], pDatagram[mI + 17], pDatagram[mI + 16] };
			unsigned char tracker_global_acceleration_x_bytes[] = { pDatagram[mI + 23], pDatagram[mI + 22], pDatagram[mI + 21], pDatagram[mI + 20] };
			unsigned char tracker_global_acceleration_y_bytes[] = { pDatagram[mI + 27], pDatagram[mI + 26], pDatagram[mI + 25], pDatagram[mI + 24] };
			unsigned char tracker_global_acceleration_z_bytes[] = { pDatagram[mI + 31], pDatagram[mI + 30], pDatagram[mI + 29], pDatagram[mI + 28] };
			unsigned char tracker_acceleration_x_bytes[] = { pDatagram[mI + 35], pDatagram[mI + 34], pDatagram[mI + 33], pDatagram[mI + 32] };
			unsigned char tracker_acceleration_y_bytes[] = { pDatagram[mI + 39], pDatagram[mI + 38], pDatagram[mI + 37], pDatagram[mI + 36] };
			unsigned char tracker_acceleration_z_bytes[] = { pDatagram[mI + 43], pDatagram[mI + 42], pDatagram[mI + 41], pDatagram[mI + 40] };
			unsigned char tracker_angular_velocity_x_bytes[] = { pDatagram[mI + 47], pDatagram[mI + 46], pDatagram[mI + 45], pDatagram[mI + 44] };
			unsigned char tracker_angular_velocity_y_bytes[] = { pDatagram[mI + 51], pDatagram[mI + 50], pDatagram[mI + 49], pDatagram[mI + 48] };
			unsigned char tracker_angular_velocity_z_bytes[] = { pDatagram[mI + 55], pDatagram[mI + 54], pDatagram[mI + 53], pDatagram[mI + 52] };
			unsigned char tracker_magnetic_field_x_bytes[] = { pDatagram[mI + 59], pDatagram[mI + 58], pDatagram[mI + 57], pDatagram[mI + 56] };
			unsigned char tracker_magnetic_field_y_bytes[] = { pDatagram[mI + 63], pDatagram[mI + 62], pDatagram[mI + 61], pDatagram[mI + 60] };
			unsigned char tracker_magnetic_field_z_bytes[] = { pDatagram[mI + 67], pDatagram[mI + 66], pDatagram[mI + 65], pDatagram[mI + 64] };

			memcpy(&tracker_id, tracker_id_bytes, sizeof(int));
			memcpy(&tracker_rotation_quat_1, tracker_rotation_q1_bytes, sizeof(float));
			memcpy(&tracker_rotation_quat_2, tracker_rotation_q2_bytes, sizeof(float));
			memcpy(&tracker_rotation_quat_3, tracker_rotation_q3_bytes, sizeof(float));
			memcpy(&tracker_rotation_quat_4, tracker_rotation_q4_bytes, sizeof(float));
			memcpy(&tracker_global_acceleration_x, tracker_global_acceleration_x_bytes, sizeof(float));
			memcpy(&tracker_global_acceleration_y, tracker_global_acceleration_y_bytes, sizeof(float));
			memcpy(&tracker_global_acceleration_z, tracker_global_acceleration_z_bytes, sizeof(float));
			memcpy(&tracker_acceleration_x, tracker_acceleration_x_bytes, sizeof(float));
			memcpy(&tracker_acceleration_y, tracker_acceleration_y_bytes, sizeof(float));
			memcpy(&tracker_acceleration_z, tracker_acceleration_z_bytes, sizeof(float));
			memcpy(&tracker_angular_velocity_x, tracker_angular_velocity_x_bytes, sizeof(float));
			memcpy(&tracker_angular_velocity_y, tracker_angular_velocity_y_bytes, sizeof(float));
			memcpy(&tracker_angular_velocity_z, tracker_angular_velocity_z_bytes, sizeof(float));
			memcpy(&tracker_magnetic_field_x, tracker_magnetic_field_x_bytes, sizeof(float));
			memcpy(&tracker_magnetic_field_y, tracker_magnetic_field_y_bytes, sizeof(float));
			memcpy(&tracker_magnetic_field_z, tracker_magnetic_field_z_bytes, sizeof(float));

			trackerIds[sI] = static_cast<float>(tracker_id);
			trackerRotationsQuat[sI * 4] = tracker_rotation_quat_1;
			trackerRotationsQuat[sI * 4 + 1] = tracker_rotation_quat_2;
			trackerRotationsQuat[sI * 4 + 2] = tracker_rotation_quat_3;
			trackerRotationsQuat[sI * 4 + 3] = tracker_rotation_quat_4;
			trackerGlobalAccelerations[sI * 3] = tracker_global_acceleration_x;
			trackerGlobalAccelerations[sI * 3 + 1] = tracker_global_acceleration_y;
			trackerGlobalAccelerations[sI * 3 + 2] = tracker_global_acceleration_z;
			trackerAccelerations[sI * 3] = tracker_acceleration_x;
			trackerAccelerations[sI * 3 + 1] = tracker_acceleration_y;
			trackerAccelerations[sI * 3 + 2] = tracker_acceleration_z;
			trackerAngularVelocities[sI * 3] = tracker_angular_velocity_x;
			trackerAngularVelocities[sI * 3 + 1] = tracker_angular_velocity_y;
			trackerAngularVelocities[sI * 3 + 2] = tracker_angular_velocity_z;
			trackerMagneticFields[sI * 3] = tracker_magnetic_field_x;
			trackerMagneticFields[sI * 3 + 1] = tracker_magnetic_field_y;
			trackerMagneticFields[sI * 3 + 2] = tracker_magnetic_field_z;
		}
	}
	else
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sTrackerCount; mI += sDatagramMotionTrackerKinematicsSizePerTracker, sI += 1)
		{
			unsigned char tracker_id_bytes[] = { pDatagram[mI + 0], pDatagram[mI + 1], pDatagram[mI + 2], pDatagram[mI + 3] };
			unsigned char tracker_rotation_q1_bytes[] = { pDatagram[mI + 4], pDatagram[mI + 5], pDatagram[mI + 6], pDatagram[mI + 7] };
			unsigned char tracker_rotation_q2_bytes[] = { pDatagram[mI + 8], pDatagram[mI + 9], pDatagram[mI + 10], pDatagram[mI + 11] };
			unsigned char tracker_rotation_q3_bytes[] = { pDatagram[mI + 12], pDatagram[mI + 13], pDatagram[mI + 14], pDatagram[mI + 15] };
			unsigned char tracker_rotation_q4_bytes[] = { pDatagram[mI + 16], pDatagram[mI + 17], pDatagram[mI + 18], pDatagram[mI + 19] };
			unsigned char tracker_global_acceleration_x_bytes[] = { pDatagram[mI + 20], pDatagram[mI + 21], pDatagram[mI + 22], pDatagram[mI + 23] };
			unsigned char tracker_global_acceleration_y_bytes[] = { pDatagram[mI + 24], pDatagram[mI + 25], pDatagram[mI + 26], pDatagram[mI + 27] };
			unsigned char tracker_global_acceleration_z_bytes[] = { pDatagram[mI + 28], pDatagram[mI + 29], pDatagram[mI + 30], pDatagram[mI + 31] };
			unsigned char tracker_acceleration_x_bytes[] = { pDatagram[mI + 32], pDatagram[mI + 33], pDatagram[mI + 34], pDatagram[mI + 35] };
			unsigned char tracker_acceleration_y_bytes[] = { pDatagram[mI + 36], pDatagram[mI + 37], pDatagram[mI + 38], pDatagram[mI + 39] };
			unsigned char tracker_acceleration_z_bytes[] = { pDatagram[mI + 40], pDatagram[mI + 41], pDatagram[mI + 42], pDatagram[mI + 43] };
			unsigned char tracker_angular_velocity_x_bytes[] = { pDatagram[mI + 44], pDatagram[mI + 45], pDatagram[mI + 46], pDatagram[mI + 47] };
			unsigned char tracker_angular_velocity_y_bytes[] = { pDatagram[mI + 48], pDatagram[mI + 49], pDatagram[mI + 50], pDatagram[mI + 51] };
			unsigned char tracker_angular_velocity_z_bytes[] = { pDatagram[mI + 52], pDatagram[mI + 53], pDatagram[mI + 54], pDatagram[mI + 55] };
			unsigned char tracker_magnetic_field_x_bytes[] = { pDatagram[mI + 56], pDatagram[mI + 57], pDatagram[mI + 58], pDatagram[mI + 59] };
			unsigned char tracker_magnetic_field_y_bytes[] = { pDatagram[mI + 60], pDatagram[mI + 61], pDatagram[mI + 62], pDatagram[mI + 63] };
			unsigned char tracker_magnetic_field_z_bytes[] = { pDatagram[mI + 64], pDatagram[mI + 65], pDatagram[mI + 66], pDatagram[mI + 67] };

			memcpy(&tracker_id, tracker_id_bytes, sizeof(int));
			memcpy(&tracker_rotation_quat_1, tracker_rotation_q1_bytes, sizeof(float));
			memcpy(&tracker_rotation_quat_2, tracker_rotation_q2_bytes, sizeof(float));
			memcpy(&tracker_rotation_quat_3, tracker_rotation_q3_bytes, sizeof(float));
			memcpy(&tracker_rotation_quat_4, tracker_rotation_q4_bytes, sizeof(float));
			memcpy(&tracker_global_acceleration_x, tracker_global_acceleration_x_bytes, sizeof(float));
			memcpy(&tracker_global_acceleration_y, tracker_global_acceleration_y_bytes, sizeof(float));
			memcpy(&tracker_global_acceleration_z, tracker_global_acceleration_z_bytes, sizeof(float));
			memcpy(&tracker_acceleration_x, tracker_acceleration_x_bytes, sizeof(float));
			memcpy(&tracker_acceleration_y, tracker_acceleration_y_bytes, sizeof(float));
			memcpy(&tracker_acceleration_z, tracker_acceleration_z_bytes, sizeof(float));
			memcpy(&tracker_angular_velocity_x, tracker_angular_velocity_x_bytes, sizeof(float));
			memcpy(&tracker_angular_velocity_y, tracker_angular_velocity_y_bytes, sizeof(float));
			memcpy(&tracker_angular_velocity_z, tracker_angular_velocity_z_bytes, sizeof(float));
			memcpy(&tracker_magnetic_field_x, tracker_magnetic_field_x_bytes, sizeof(float));
			memcpy(&tracker_magnetic_field_y, tracker_magnetic_field_y_bytes, sizeof(float));
			memcpy(&tracker_magnetic_field_z, tracker_magnetic_field_z_bytes, sizeof(float));

			trackerIds[sI] = static_cast<float>(tracker_id);
			trackerRotationsQuat[sI * 4] = tracker_rotation_quat_1;
			trackerRotationsQuat[sI * 4 + 1] = tracker_rotation_quat_2;
			trackerRotationsQuat[sI * 4 + 2] = tracker_rotation_quat_3;
			trackerRotationsQuat[sI * 4 + 3] = tracker_rotation_quat_4;
			trackerGlobalAccelerations[sI * 3] = tracker_global_acceleration_x;
			trackerGlobalAccelerations[sI * 3 + 1] = tracker_global_acceleration_y;
			trackerGlobalAccelerations[sI * 3 + 2] = tracker_global_acceleration_z;
			trackerAccelerations[sI * 3] = tracker_acceleration_x;
			trackerAccelerations[sI * 3 + 1] = tracker_acceleration_y;
			trackerAccelerations[sI * 3 + 2] = tracker_acceleration_z;
			trackerAngularVelocities[sI * 3] = tracker_angular_velocity_x;
			trackerAngularVelocities[sI * 3 + 1] = tracker_angular_velocity_y;
			trackerAngularVelocities[sI * 3 + 2] = tracker_angular_velocity_z;
			trackerMagneticFields[sI * 3] = tracker_magnetic_field_x;
			trackerMagneticFields[sI * 3 + 1] = tracker_magnetic_field_y;
			trackerMagneticFields[sI * 3 + 2] = tracker_magnetic_field_z;
		}
	}

	try
	{
		MocapSkeletonManager::get().updateProperty(character_id, "/tracker/id", trackerIds);
		MocapSkeletonManager::get().updateProperty(character_id, "/tracker/rot_world", trackerRotationsQuat);
		MocapSkeletonManager::get().updateProperty(character_id, "/tracker/accel_world", trackerGlobalAccelerations);
		//MocapSkeletonManager::get().updateProperty(character_id, "/tracker/accel", trackerAccelerations);
		//MocapSkeletonManager::get().updateProperty(character_id, "/tracker/angular_vel", trackerAngularVelocities);
		MocapSkeletonManager::get().updateProperty(character_id, "/tracker/magnet", trackerMagneticFields);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("Network Error: failed to update angular segment kinematics\n", __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}

void 
StreamManager::updateJointAngles(unsigned char* pDatagram) throw (dab::Exception)
{
	uint8_t character_id = pDatagram[16];

	std::vector<int> parentIds(sSegmentCount);
	std::vector<int> childIds(sSegmentCount);
	std::vector<float> jointAngles(sSegmentCount * 3);

	uint32_t parent_id;
	uint32_t child_id;
	float joint_angle_x;
	float joint_angle_y;
	float joint_angle_z;

	if (sBigEndian == true)
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sSegmentCount; mI += sDatagramJointAnglesSizePerSegment, sI += 1)
		{
			unsigned char parent_id_bytes[] = { pDatagram[mI + 7], pDatagram[mI + 6], pDatagram[mI + 5], pDatagram[mI + 4] };
			unsigned char child_id_bytes[] = { pDatagram[mI + 7], pDatagram[mI + 6], pDatagram[mI + 5], pDatagram[mI + 4] };
			unsigned char joint_angle_x_bytes[] = { pDatagram[mI + 11], pDatagram[mI + 10], pDatagram[mI + 9], pDatagram[mI + 8] };
			unsigned char joint_angle_y_bytes[] = { pDatagram[mI + 15], pDatagram[mI + 14], pDatagram[mI + 13], pDatagram[mI + 12] };
			unsigned char joint_angle_z_bytes[] = { pDatagram[mI + 19], pDatagram[mI + 18], pDatagram[mI + 17], pDatagram[mI + 16] };

			memcpy(&parent_id, parent_id_bytes, sizeof(uint32_t));
			memcpy(&child_id, child_id_bytes, sizeof(uint32_t));
			memcpy(&joint_angle_x, joint_angle_x_bytes, sizeof(float));
			memcpy(&joint_angle_y, joint_angle_y_bytes, sizeof(float));
			memcpy(&joint_angle_z, joint_angle_z_bytes, sizeof(float));

			parentIds[sI] = parent_id;
			childIds[sI] = child_id;

			jointAngles[sI * 3] = joint_angle_x;
			jointAngles[sI * 3 + 1] = joint_angle_y;
			jointAngles[sI * 3 + 2] = joint_angle_z;
		}
	}
	else
	{
		for (int mI = sDatagramHeaderSize, sI = 0; sI < sSegmentCount; mI += sDatagramAngularKinematicsSizePerSegment, sI += 1)
		{
			unsigned char parent_id_bytes[] = { pDatagram[mI + 3], pDatagram[mI + 5], pDatagram[mI + 6], pDatagram[mI + 7] };
			unsigned char child_id_bytes[] = { pDatagram[mI + 8], pDatagram[mI + 9], pDatagram[mI + 10], pDatagram[mI + 11] };
			unsigned char joint_angle_x_bytes[] = { pDatagram[mI + 12], pDatagram[mI + 13], pDatagram[mI + 14], pDatagram[mI + 15] };
			unsigned char joint_angle_y_bytes[] = { pDatagram[mI + 16], pDatagram[mI + 17], pDatagram[mI + 18], pDatagram[mI + 19] };
			unsigned char joint_angle_z_bytes[] = { pDatagram[mI + 20], pDatagram[mI + 21], pDatagram[mI + 22], pDatagram[mI + 23] };
	
			memcpy(&parent_id, parent_id_bytes, sizeof(uint32_t));
			memcpy(&child_id, child_id_bytes, sizeof(uint32_t));
			memcpy(&joint_angle_x, joint_angle_x_bytes, sizeof(float));
			memcpy(&joint_angle_y, joint_angle_y_bytes, sizeof(float));
			memcpy(&joint_angle_z, joint_angle_z_bytes, sizeof(float));

			parentIds[sI] = parent_id;
			childIds[sI] = child_id;

			jointAngles[sI * 3] = joint_angle_x;
			jointAngles[sI * 3 + 1] = joint_angle_y;
			jointAngles[sI * 3 + 2] = joint_angle_z;
		}
	}

	try
	{
		MocapSkeletonManager::get().updateProperty(character_id, "/joint/angles", jointAngles);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("Network Error: failed to update angular segment kinematics\n", __FILE__, __FUNCTION__, __LINE__);
		throw e;
	}
}