#pragma once

#include <vector>
#include <map>
#include <mutex> 
#include "dab_exception.h"
#include "dab_singleton.h"
#include "dab_xsens_stream_manager.h"

namespace dab
{

namespace xsens
{

class MocapSkeleton;

# pragma mark MocapSkeletonManager declaration

class MocapSkeletonManager : public Singleton<MocapSkeletonManager>
{
public:
	friend class StreamManager;

	MocapSkeletonManager();
	~MocapSkeletonManager();

	const std::vector<unsigned int>& skeletonIds() const;
	bool skeletonAvailable(unsigned int pSkeletonId) const;
	std::shared_ptr<MocapSkeleton> skeleton(unsigned int pSkeletonId) const throw (dab::Exception);

protected:
	std::vector<unsigned int> mSkeletonIds;
	std::map<unsigned int, std::shared_ptr<MocapSkeleton> > mSkeletons;

	std::mutex mLock;

	void updateProperty(unsigned int pSkeletonId, const std::string& pPropertyName, const std::vector<float>& pPropertyValues) throw (dab::Exception);
};

# pragma mark MocapSkeletonManager declaration


class MocapSkeleton
{
public:
	friend class StreamManager;
	friend class MocapSkeletonManager;

	MocapSkeleton(unsigned int pId);
	~MocapSkeleton();

	int id() const;
	const std::vector<std::string>& propertyNames() const;
	bool propertyAvailable(const std::string& pPropertyName) const;
	const std::vector<float>&property(const std::string& pPropertyName) const throw (dab::Exception);

protected:
	static unsigned int sJointCount;

	unsigned int mId;
	unsigned int mJointCount;
	std::vector<std::string> mPropertyNames;
	std::map<std::string, std::vector<float> > mProperties;
	std::map<std::string, bool> mPropertiesAvailable;

	std::mutex mLock;

	void updateProperty(const std::string& pPropertyName, const std::vector<float>& pPropertyValues);
};

};

};


/*
Infos concerning segment names and indices (from XSens manual)

0: Pelvis
1: L5
2: L3
3: T12
4: T8
5: Neck
6: Head
7: Right Shoulder
8: Right Upper Arm
9: Right Forearm
10: Right Hand
11: Left Shoulder
12: Left Upper Arm
13: Left Forearm
14: Left Hand
15: Right Upper Leg
16: Right Lower Leg
17: Right Foot
18: Right Toe
19: Left Upper Leg
20: Left Lower Leg
21: Left Foot
22: Left Toe
*/