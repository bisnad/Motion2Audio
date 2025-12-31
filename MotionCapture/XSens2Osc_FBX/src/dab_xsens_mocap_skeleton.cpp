#include "dab_xsens_mocap_skeleton.h"

using namespace dab;
using namespace dab::xsens;


# pragma mark MocapSkeletonManager definition

MocapSkeletonManager::MocapSkeletonManager()
{}

MocapSkeletonManager::~MocapSkeletonManager()
{}

const std::vector<unsigned int>& 
MocapSkeletonManager::skeletonIds() const
{
	return mSkeletonIds;
}

bool 
MocapSkeletonManager::skeletonAvailable(unsigned int pSkeletonId) const
{
	return mSkeletons.find(pSkeletonId) != mSkeletons.end();
}

std::shared_ptr<MocapSkeleton> 
MocapSkeletonManager::skeleton(unsigned int pSkeletonId) const throw (dab::Exception)
{
	if (skeletonAvailable(pSkeletonId) == false) throw dab::Exception("Data Error: skeleton with id " + std::to_string(pSkeletonId) + " not available\n", __FILE__, __FUNCTION__, __LINE__);

	return mSkeletons.at(pSkeletonId);
}

void 
MocapSkeletonManager::updateProperty(unsigned int pSkeletonId, const std::string& pPropertyName, const std::vector<float>& pPropertyValues) throw (dab::Exception)
{
	mLock.lock();

	if (skeletonAvailable(pSkeletonId) == false)
	{
		mSkeletonIds.push_back(pSkeletonId);
		mSkeletons[pSkeletonId] = std::shared_ptr<MocapSkeleton>( new MocapSkeleton(pSkeletonId) );
	}

	try
	{
		mSkeletons[pSkeletonId]->updateProperty(pPropertyName, pPropertyValues);
	}
	catch (dab::Exception& e)
	{
		e += dab::Exception("Data Error: failed to update property " + pPropertyName + " to skeleton with id " + std::to_string(pSkeletonId) + "\n", __FILE__, __FUNCTION__, __LINE__);
		mLock.unlock();
		throw e;
	}

	mLock.unlock();
}

# pragma mark MocapSkeleton definition

unsigned int MocapSkeleton::sJointCount = 23;

MocapSkeleton::MocapSkeleton(unsigned int pId)
	: mId(pId)
	, mJointCount(sJointCount)
{}

MocapSkeleton::~MocapSkeleton()
{}

int
MocapSkeleton::id() const
{
	return mId;
}

const std::vector<std::string>&
MocapSkeleton::propertyNames() const
{
	return mPropertyNames;
}

bool 
MocapSkeleton::propertyAvailable(const std::string& pPropertyName) const
{
	return mProperties.find(pPropertyName) != mProperties.end();
}

const std::vector<float>& 
MocapSkeleton::property(const std::string& pPropertyName) const throw (dab::Exception)
{
	if (propertyAvailable(pPropertyName) == false) throw dab::Exception("Data Error: property " + pPropertyName + " not found\n", __FILE__, __FUNCTION__, __LINE__);

	return mProperties.at(pPropertyName);
}

void 
MocapSkeleton::updateProperty(const std::string& pPropertyName, const std::vector<float>& pPropertyValues)
{


	mLock.lock();

	//std::cout << "skeleton id " << mId << " update property " << pPropertyName << "\n";

	if (propertyAvailable(pPropertyName) == false)
	{
		mPropertyNames.push_back(pPropertyName);
	}

	mProperties[pPropertyName] = pPropertyValues;

	mLock.unlock();
}
