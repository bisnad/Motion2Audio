#pragma once

#include "ofxOsc.h"
#include "ofThread.h"
#include "dab_singleton.h"
#include "dab_exception.h"
#include <mutex>

namespace dab
{

namespace xsens
{

# pragma mark OscSender declaration

class OscSender : public ofThread
{
public:
	OscSender(const std::string& pIpAddress, unsigned int pPort, float pSendRate);
	~OscSender();

	void start();
	void stop();
	void send();
	void sendData(const std::string& pAddress, const std::vector<float>& pValues);

	protected:
		std::string mIpAddress;
		unsigned int mPort;
		unsigned int mSendInterval; // in microseconds
		ofxOscSender mSender;

		void threadedFunction();

		std::mutex mLock;
};

# pragma mark OscManager declaration

class OscManager : public Singleton<OscManager>
{
public:
	OscManager();
	~OscManager();

	void addSender(const std::string& pIpAddress, unsigned int pPort, float pSendRate);
	void start();
	void stop();

protected:
	std::vector<std::shared_ptr<OscSender>> mSenders;
};

};

};