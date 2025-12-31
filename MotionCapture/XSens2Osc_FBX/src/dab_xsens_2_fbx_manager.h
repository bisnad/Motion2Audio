#pragma once

#include <array>
#include <vector>
#include "dab_singleton.h"

namespace dab
{

namespace xsens
{

# pragma mark Xsens2FbxManager declaration

class Xsens2FbxManager : public Singleton<Xsens2FbxManager>
{
public:
	Xsens2FbxManager();
	~Xsens2FbxManager();

	int getJointCount() const;

	std::vector<std::array<float, 3>> remapJointIndices(const std::vector<std::array<float, 3>>& pJointData) const;
	std::vector<std::array<float, 4>> remapJointIndices(const std::vector<std::array<float, 4>>& pJointData) const;

	std::vector<std::array<float, 3>> convertWorld2Local( const std::vector<std::array<float, 3>>& pRotWorld) const;
	std::vector<std::array<float, 4>> convertWorld2Local(const std::vector<std::array<float, 4>>& pRotWorld) const;

	std::vector<std::array<float, 3>> swapCoordinates(const std::vector<std::array<float, 3>>& pJointData) const;
	std::vector<std::array<float, 4>> swapCoordinates(const std::vector<std::array<float, 4>>& pJointData) const;

protected:
	static int sJointCount;
	static std::vector<int> sJointParents;

	// quaternion helper functions
	std::array<float, 4> qconjugate(const std::array<float, 4>& pQ) const; 
	std::array<float, 4> qmul(const std::array<float, 4>& pQ1, const std::array<float, 4>& pQ2) const;

	int mJointCount;
	std::vector<int> mJointParents;

	std::vector<size_t> mJointIndexMap = { 0, 15, 16, 17, 18, 19, 20, 21, 22, 1, 2, 3, 4, 11, 12, 13, 14, 7, 8, 9, 10, 5, 6 };
	std::array<size_t, 4> mPosIndexMap = { 1, 2, 0 };
	std::array<size_t, 4> mRotIndexMap = { 0, 2, 3, 1 };
};

};
};