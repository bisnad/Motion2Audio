/** \file dab_json_helper.cpp
*/

#include "dab_json_helper.h"

using namespace dab;

const Json::Value&
JsonHelper::getValue(const Json::Value& pValue, const std::string& pValueName) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw dab::Exception("JSON ERROR: Value does not contain value name: " + pValueName, __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueName];
}

int
JsonHelper::getInt(const Json::Value& pValue, const std::string& pValueName) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw dab::Exception("JSON ERROR: Value does not contain value name: " + pValueName, __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueName].isInt() == false) throw dab::Exception("JSON ERROR: value name " + pValueName + " is not an integer", __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueName].asInt();
}

float
JsonHelper::getFloat(const Json::Value& pValue, const std::string& pValueName) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw dab::Exception("JSON ERROR: Value does not contain value name: " + pValueName, __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueName].isDouble() == false) throw dab::Exception("JSON ERROR: value name " + pValueName + " is not a float", __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueName].asFloat();
}

std::string
JsonHelper::getString(const Json::Value& pValue, const std::string& pValueName) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw dab::Exception("JSON ERROR: Value does not contain value name: " + pValueName, __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueName].isString() == false) throw dab::Exception("JSON ERROR: value name " + pValueName + " is not a float", __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueName].asString();
}

const Json::Value&
JsonHelper::getValue(const Json::Value& pValue, int pValueIndex) throw (dab::Exception)
{
	if (pValue.size() <= pValueIndex) throw dab::Exception("JSON ERROR: value index " + std::to_string(pValueIndex) + " exceeds size " + std::to_string(pValue.size()), __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueIndex];
}

int
JsonHelper::getInt(const Json::Value& pValue, unsigned int pValueIndex) throw (dab::Exception)
{
	if (pValue.size() <= pValueIndex) throw dab::Exception("JSON ERROR: value index " + std::to_string(pValueIndex) + " exceeds size " + std::to_string(pValue.size()), __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueIndex].isInt() == false) throw dab::Exception("JSON ERROR: value at index " + std::to_string(pValueIndex) + " is not an integer", __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueIndex].asInt();
}

float
JsonHelper::getFloat(const Json::Value& pValue, unsigned int pValueIndex) throw (dab::Exception)
{
	if (pValue.size() <= pValueIndex) throw dab::Exception("JSON ERROR: value index " + std::to_string(pValueIndex) + " exceeds size " + std::to_string(pValue.size()), __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueIndex].isDouble() == false) throw dab::Exception("JSON ERROR: value at index " + std::to_string(pValueIndex) + " is not a float", __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueIndex].asFloat();
}

std::string
JsonHelper::getString(const Json::Value& pValue, unsigned int pValueIndex) throw (dab::Exception)
{
	if (pValue.size() <= pValueIndex) throw dab::Exception("JSON ERROR: value index " + std::to_string(pValueIndex) + " exceeds size " + std::to_string(pValue.size()), __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueIndex].isString() == false) throw dab::Exception("JSON ERROR: value at index " + std::to_string(pValueIndex) + " is not a string", __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueIndex].asString();
}


const Json::Value&
JsonHelper::getValue(const Json::Value& pValue, const std::string& pValueName, int pValueIndex) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw dab::Exception("JSON ERROR: Value does not contain value name: " + pValueName, __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueName].size() <= pValueIndex) throw dab::Exception("JSON ERROR: value index " + std::to_string(pValueIndex) + " exceeds size " + std::to_string(pValue[pValueName].size()) + " of value name: " + pValueName + " ", __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueName][pValueIndex];
}

int
JsonHelper::getInt(const Json::Value& pValue, const std::string& pValueName, unsigned int pValueIndex) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw dab::Exception("JSON ERROR: Value does not contain value name: " + pValueName, __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueName].size() <= pValueIndex) throw dab::Exception("JSON ERROR: value index " + std::to_string(pValueIndex) + " exceeds size " + std::to_string(pValue[pValueName].size()) + " of value name: " + pValueName + " ", __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueName][pValueIndex].isInt() == false) throw dab::Exception("JSON ERROR: value name " + pValueName + " at index " + std::to_string(pValueIndex) + " is not an integer", __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueName][pValueIndex].asInt();
}

float
JsonHelper::getFloat(const Json::Value& pValue, const std::string& pValueName, unsigned int pValueIndex) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw dab::Exception("JSON ERROR: Value does not contain value name: " + pValueName, __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueName].size() <= pValueIndex) throw dab::Exception("JSON ERROR: value index " + std::to_string(pValueIndex) + " exceeds size " + std::to_string(pValue[pValueName].size()) + " of value name: " + pValueName + " ", __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueName][pValueIndex].isDouble() == false) throw dab::Exception("JSON ERROR: value name " + pValueName + " at index " + std::to_string(pValueIndex) + " is not a float", __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueName][pValueIndex].asFloat();
}

std::string
JsonHelper::getString(const Json::Value& pValue, const std::string& pValueName, unsigned int pValueIndex) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw dab::Exception("JSON ERROR: Value does not contain value name: " + pValueName, __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueName].size() <= pValueIndex) throw dab::Exception("JSON ERROR: value index " + std::to_string(pValueIndex) + " exceeds size " + std::to_string(pValue[pValueName].size()) + " of value name: " + pValueName + " ", __FILE__, __FUNCTION__, __LINE__);
	if (pValue[pValueName][pValueIndex].isString() == false) throw dab::Exception("JSON ERROR: value name " + pValueName + " at index " + std::to_string(pValueIndex) + " is not a string", __FILE__, __FUNCTION__, __LINE__);

	return pValue[pValueName][pValueIndex].asString();
}

void 
JsonHelper::getValues(const Json::Value& pValue, const std::string& pValueName, std::vector<Json::Value>& pValues) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw Exception("JSON ERROR: no data with name " + pValueName + " found", __FILE__, __FUNCTION__, __LINE__);

	const Json::Value& data = pValue[pValueName];
	int dataSize = data.size();

	pValues.resize(dataSize);

	for (int d = 0; d < dataSize; ++d)
	{
		pValues[d] = data[d];
	}
}

void 
JsonHelper::getInts(const Json::Value& pValue, const std::string& pValueName, std::vector<int>& pValues) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw Exception("JSON ERROR: no data with name " + pValueName + " found", __FILE__, __FUNCTION__, __LINE__);

	const Json::Value& data = pValue[pValueName];
	int dataSize = data.size();

	pValues.resize(dataSize);

	for (int d = 0; d < dataSize; ++d)
	{
		pValues[d] = data[d].asInt();
	}
}

void 
JsonHelper::getFloats(const Json::Value& pValue, const std::string& pValueName, std::vector<float>& pValues) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw Exception("JSON ERROR: no data with name " + pValueName + " found", __FILE__, __FUNCTION__, __LINE__);

	const Json::Value& data = pValue[pValueName];
	int dataSize = data.size();

	pValues.resize(dataSize);

	for (int d = 0; d < dataSize; ++d)
	{
		pValues[d] = data[d].asFloat();
	}
}

void 
JsonHelper::getStrings(const Json::Value& pValue, const std::string& pValueName, std::vector<std::string>& pValues) throw (dab::Exception)
{
	if (pValue.isMember(pValueName) == false) throw Exception("JSON ERROR: no data with name " + pValueName + " found", __FILE__, __FUNCTION__, __LINE__);

	const Json::Value& data = pValue[pValueName];
	int dataSize = data.size();

	pValues.resize(dataSize);

	for (int d = 0; d < dataSize; ++d)
	{
		pValues[d] = data[d].asString();
	}
}

void 
JsonHelper::getInts(const Json::Value& pValue, std::vector<int>& pValues) throw (dab::Exception)
{
	int dataSize = pValue.size();

	pValues.resize(dataSize);

	for (int d = 0; d < dataSize; ++d)
	{
		pValues[d] = pValue[d].asInt();
	}
}

void 
JsonHelper::getFloats(const Json::Value& pValue, std::vector<float>& pValues) throw (dab::Exception)
{
	int dataSize = pValue.size();

	pValues.resize(dataSize);

	for (int d = 0; d < dataSize; ++d)
	{
		pValues[d] = pValue[d].asFloat();
	}
}

void 
JsonHelper::getStrings(const Json::Value& pValue, std::vector<std::string>& pValues) throw (dab::Exception)
{
	int dataSize = pValue.size();

	pValues.resize(dataSize);

	for (int d = 0; d < dataSize; ++d)
	{
		pValues[d] = pValue[d].asString();
	}
}
