#pragma once

#include "ofxNetwork.h"
#include "dab_singleton.h"
#include "dab_exception.h"

namespace dab
{

namespace xsens
{

class StreamManager : public Singleton<StreamManager>
{
public:
	StreamManager();
	~StreamManager();

	void setupUDPReceiver(unsigned int pPort) throw (dab::Exception);
	void update() throw (dab::Exception);

protected:
	static const bool sBigEndian;
	static const unsigned int sTrackerCount;
	static const unsigned int sSegmentCount;
	static const unsigned int sDatagramMaxSize;
	static const unsigned int sDatagramHeaderSize;
	static const unsigned int sDatagramPoseDataEulerSizePerSegment;
	static const unsigned int sDatagramPoseDataQuaternionSizePerSegment;
	static const unsigned int sDatagramLinearKinematicsSizePerSegment;
	static const unsigned int sDatagramAngularKinematicsSizePerSegment;
	static const unsigned int sDatagramMotionTrackerKinematicsSizePerTracker;
	static const unsigned int sDatagramJointAnglesSizePerSegment;

	unsigned int mReceivePort;
	ofxUDPManager mUDPReceiver;

	void updatePoseDataEuler(unsigned char* pDatagram) throw (dab::Exception);
	void updatePoseDataQuaternion(unsigned char* pDatagram) throw (dab::Exception);
	void updateLinearSegmentKinematics(unsigned char* pDatagram) throw (dab::Exception);
	void updateAngularSegmentKinematics(unsigned char* pDatagram) throw (dab::Exception);
	void updateMotionTrackerKinematics(unsigned char* pDatagram) throw (dab::Exception);
	void updateJointAngles(unsigned char* pDatagram) throw (dab::Exception);
};

};

};