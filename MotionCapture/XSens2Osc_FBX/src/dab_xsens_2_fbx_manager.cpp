#include "dab_xsens_2_fbx_manager.h"

using namespace dab;
using namespace dab::xsens;

# pragma mark Xsens2FbxManager definition

int Xsens2FbxManager::sJointCount = 23;
std::vector<int> Xsens2FbxManager::sJointParents = { -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 12, 21 };

Xsens2FbxManager::Xsens2FbxManager():
    mJointCount(sJointCount),
    mJointParents(sJointParents)
{}

Xsens2FbxManager::~Xsens2FbxManager()
{
}

int 
Xsens2FbxManager::getJointCount() const
{
    return mJointCount;
}

std::array<float, 4> 
Xsens2FbxManager::qconjugate(const std::array<float, 4>& pQ) const
{
	return { pQ[0], -pQ[1], -pQ[2], -pQ[3] };
}

std::array<float, 4> 
Xsens2FbxManager::qmul(const std::array<float, 4>& pQ1, const std::array<float, 4>& pQ2) const
{
	const float w1 = pQ1[0], x1 = pQ1[1], y1 = pQ1[2], z1 = pQ1[3];
	const float w2 = pQ2[0], x2 = pQ2[1], y2 = pQ2[2], z2 = pQ2[3];

	std::array<float, 4> q;
	q[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2; // w
	q[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2; // x
	q[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2; // y
	q[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2; // z
	return q;
}

std::vector<std::array<float, 3>> 
Xsens2FbxManager::remapJointIndices(const std::vector<std::array<float, 3>>& pJointData) const
{
	std::vector<std::array<float, 3>> remappedJointData(mJointCount);

	for (size_t jI = 0; jI < mJointCount; ++jI)
	{
		for (size_t d = 0; d < 3; ++d)
		{
			remappedJointData[jI][d] = pJointData[mJointIndexMap[jI]][d];
		}
	}

	return remappedJointData;
}

std::vector<std::array<float, 4>> 
Xsens2FbxManager::remapJointIndices(const std::vector<std::array<float, 4>>& pJointData) const
{
	std::vector<std::array<float, 4>> remappedJointData(mJointCount);

	for (size_t jI = 0; jI < mJointCount; ++jI)
	{
		for (size_t d = 0; d < 4; ++d)
		{
			remappedJointData[jI][d] = pJointData[mJointIndexMap[jI]][d];
		}
	}

	return remappedJointData;
}

std::vector<std::array<float, 3>> 
Xsens2FbxManager::convertWorld2Local(const std::vector<std::array<float, 3>>& pPosWorld) const
{
    if (pPosWorld.size() != mJointCount)
        throw std::runtime_error("Incurrent Number of Joint Rotations");

    std::vector<std::array<float, 3>> posLocal(mJointCount);

    // Find root
    int root = -1;
    for (std::size_t j = 0; j < mJointCount; ++j)
    {
        if (mJointParents[j] == -1)
        {
            root = static_cast<int>(j);
            break;
        }
    }
    if (root < 0)
        throw std::runtime_error("No root joint found (parent == -1).");

    // Root: local == world
    posLocal[root] = pPosWorld[root];

    // Other joints
    for (std::size_t j = 0; j < mJointCount; ++j)
    {
        if (static_cast<int>(j) == root)
            continue;

        int p = mJointParents[j];
        if (p < 0)
            continue; // safety, should not happen except root

        const std::array<float, 3>& pos_p_world = pPosWorld[static_cast<std::size_t>(p)];
        const std::array<float, 3>& pos_j_world = pPosWorld[j];

        std::array<float, 3> pos_j_local = { 
            pos_j_world[0] - pos_p_world[0], 
            pos_j_world[1] - pos_p_world[1],
            pos_j_world[2] - pos_p_world[2] 
        };

        posLocal[j] = pos_j_local;
    }

    return posLocal;
}

std::vector<std::array<float, 4>> 
Xsens2FbxManager::convertWorld2Local(const std::vector<std::array<float, 4>>& pRotWorld) const
{
    if(pRotWorld.size() != mJointCount)
        throw std::runtime_error("Incurrent Number of Joint Rotations");

    std::vector<std::array<float, 4>> rotLocal(mJointCount);

    // Find root
    int root = -1;
    for (std::size_t j = 0; j < mJointCount; ++j)
    {
        if (mJointParents[j] == -1)
        {
            root = static_cast<int>(j);
            break;
        }
    }
    if (root < 0)
        throw std::runtime_error("No root joint found (parent == -1).");

    // Root: local == world
    rotLocal[root] = pRotWorld[root];

    // Other joints
    for (std::size_t j = 0; j < mJointCount; ++j)
    {
        if (static_cast<int>(j) == root)
            continue;

        int p = mJointParents[j];
        if (p < 0)
            continue; // safety, should not happen except root

        const std::array<float, 4>& R_p_world = pRotWorld[static_cast<std::size_t>(p)];
        const std::array<float, 4>& R_j_world = pRotWorld[j];

        std::array<float, 4> R_p_inv = qconjugate(R_p_world);
        std::array<float, 4> R_j_local = qmul(R_p_inv, R_j_world);

        rotLocal[j] = R_j_local;
    }

    return rotLocal;
}

std::vector<std::array<float, 3>> 
Xsens2FbxManager::swapCoordinates(const std::vector<std::array<float, 3>>& pJointPos) const
{
    std::vector<std::array<float, 3>> swappedJointPos(mJointCount);

    for (size_t jI = 0; jI < mJointCount; ++jI)
    {
        for (size_t d = 0; d < 4; ++d)
        {
            swappedJointPos[jI][d] = pJointPos[jI][mPosIndexMap[d]];
        }
    }

    return swappedJointPos;
}

std::vector<std::array<float, 4>> 
Xsens2FbxManager::swapCoordinates(const std::vector<std::array<float, 4>>& pJointRot) const
{
    std::vector<std::array<float, 4>> swappedJointRot(mJointCount);

    for (size_t jI = 0; jI < mJointCount; ++jI)
    {
        for (size_t d = 0; d < 4; ++d)
        {
            swappedJointRot[jI][d] = pJointRot[jI][mRotIndexMap[d]];
        }
    }

    return swappedJointRot;
}
