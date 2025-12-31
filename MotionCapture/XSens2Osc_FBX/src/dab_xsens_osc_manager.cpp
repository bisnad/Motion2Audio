#include "dab_xsens_osc_manager.h"
#include "dab_xsens_mocap_skeleton.h"
#include "dab_xsens_2_fbx_manager.h"

using namespace dab;
using namespace dab::xsens;

# pragma mark OscSender definition

OscSender::OscSender(const std::string& pIpAddress, unsigned int pPort, float pSendRate)
	: mIpAddress(pIpAddress)
	, mPort(pPort)
{
	mSendInterval = unsigned int(1000000 / pSendRate);

	mSender.setup(mIpAddress, mPort);
}

OscSender::~OscSender()
{}

void 
OscSender::start()
{
	if (isThreadRunning() == false) startThread();
}

void 
OscSender::stop()
{
	if (isThreadRunning() == true) stopThread();
}

void 
OscSender::send()
{
	mLock.lock();

	try
	{
		MocapSkeletonManager& skeletonManager = MocapSkeletonManager::get();
		Xsens2FbxManager& fbxManager = Xsens2FbxManager::get();
		int jointCount = fbxManager.getJointCount();

		const std::vector<unsigned int>& skeletonIds = skeletonManager.skeletonIds();

		for (auto skel_id : skeletonIds)
		{
			std::shared_ptr<MocapSkeleton> skeleton = skeletonManager.skeleton(skel_id);
			const std::vector<std::string>& propertyNames = skeleton->propertyNames();

			for (auto properyName : propertyNames)
			{

				std::vector<float> property = skeleton->property(properyName);

				if (properyName == "/joint/pos_world")
				{
					// convert vector<float> to vector<array<float, 3>>
					std::vector<std::array<float, 3>> jointPositionsXSens(jointCount);

					for (int jI = 0, dI = 0; jI < jointCount; ++jI)
					{
						for (int d = 0; d < 3; ++d, ++dI)
						{
							jointPositionsXSens[jI][d] = property[dI];
						}
					}

					// joint index remap from mvn to fbx 
					std::vector<std::array<float, 3>> jointPositionsWorldRemapped = fbxManager.remapJointIndices(jointPositionsXSens);
					// joint positions world to local conversion
					std::vector<std::array<float, 3>> jointPositionsLocalRemapped = fbxManager.convertWorld2Local(jointPositionsWorldRemapped);
					// swap coordinates
					std::vector<std::array<float, 3>> jointPositionsWorldFBX = fbxManager.swapCoordinates(jointPositionsWorldRemapped);
					std::vector<std::array<float, 3>> jointPositionsLocalFBX = fbxManager.swapCoordinates(jointPositionsLocalRemapped);

					// convert vector<array<float, 3>> to vector<float>
					std::vector<float> jointPosWorldValues(jointCount * 3);
					std::vector<float> jointPosLocalValues(jointCount * 3);

					for (int jI = 0, dI = 0; jI < jointCount; ++jI)
					{
						for (int d = 0; d < 3; ++d, ++dI)
						{
							jointPosWorldValues[dI] = jointPositionsWorldFBX[jI][d];
							jointPosLocalValues[dI] = jointPositionsLocalFBX[jI][d];
						}
					}

					// send joint values
					sendData(std::string("/mocap/") + std::to_string(skel_id) + "/joint/pos_world", jointPosWorldValues);
					sendData(std::string("/mocap/") + std::to_string(skel_id) + "/joint/pos_local", jointPosLocalValues);

				}
				else if(properyName == "/joint/rot_world")
				{ 
					// convert vector<float> to vector<array<float, 4>>
					std::vector<std::array<float, 4>> jointRotationsXSens(jointCount);

					for (int jI = 0, dI = 0; jI < jointCount; ++jI)
					{
						for (int d = 0; d < 4; ++d, ++dI)
						{
							jointRotationsXSens[jI][d] = property[dI];
						}
					}

					// joint index remap from mvn to fbx 
					std::vector<std::array<float, 4>> jointRotationsWorldRemapped = fbxManager.remapJointIndices(jointRotationsXSens);
					// joint rotation world to local conversion
					std::vector<std::array<float, 4>> jointRotationsLocalRemapped = fbxManager.convertWorld2Local(jointRotationsWorldRemapped);
					// swap coordinates
					std::vector<std::array<float, 4>> jointRotationsWorldFBX = fbxManager.swapCoordinates(jointRotationsWorldRemapped);
					std::vector<std::array<float, 4>> jointRotationsLocalFBX = fbxManager.swapCoordinates(jointRotationsLocalRemapped);

					// convert vector<array<float, 3>> to vector<float>
					std::vector<float> jointRotWorldValues(jointCount * 4);
					std::vector<float> jointRotLocalValues(jointCount * 4);

					for (int jI = 0, dI = 0; jI < jointCount; ++jI)
					{
						for (int d = 0; d < 4; ++d, ++dI)
						{
							jointRotWorldValues[dI] = jointRotationsWorldFBX[jI][d];
							jointRotLocalValues[dI] = jointRotationsLocalFBX[jI][d];
						}
					}

					// send joint values
					sendData(std::string("/mocap/") + std::to_string(skel_id) + "/joint/rot_world", jointRotWorldValues);
					sendData(std::string("/mocap/") + std::to_string(skel_id) + "/joint/rot_local", jointRotLocalValues);

				}
				else
				{
					// do not send the data
				}
			}
		}
	}
	catch (dab::Exception& e)
	{
		std::cout << e << "\n";
	}

	mLock.unlock();
}

void
OscSender::sendData(const std::string& pAddress, const std::vector<float>& pValues)
{
	ofxOscMessage oscMessage;

	oscMessage.setAddress(pAddress);

	for (float value : pValues)
	{
		oscMessage.addFloatArg(value);
	}

	mSender.sendMessage(oscMessage, false);
}

void 
OscSender::threadedFunction()
{
	while (isThreadRunning())
	{
		send();

		std::this_thread::sleep_for(std::chrono::microseconds(mSendInterval));
	}
}


# pragma mark OscManager definition


OscManager::OscManager()
{}

OscManager::~OscManager() 
{
	stop();
}

void 
OscManager::addSender(const std::string& pIpAddress, unsigned int pPort, float pSendRate)
{
	std::shared_ptr<OscSender> sender( new OscSender(pIpAddress, pPort, pSendRate) );
	mSenders.push_back(sender);
}

void 
OscManager::start()
{
	for(auto sender : mSenders)
	{ 
		sender->start();
	}
}

void 
OscManager::stop()
{
	for (auto sender : mSenders)
	{
		sender->stop();
	}
}

