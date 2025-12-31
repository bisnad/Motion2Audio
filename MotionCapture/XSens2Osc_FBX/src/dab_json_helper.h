/** \file dab_json_helper.h
*/

#pragma once

#include <iostream>
#include "ofxJSON.h"
#include "dab_singleton.h"
#include "dab_exception.h"

namespace dab
{

class JsonHelper : public Singleton<JsonHelper>
{
public:
	const Json::Value& getValue(const Json::Value& pValue, const std::string& pValueName) throw (dab::Exception);
	int getInt(const Json::Value& pValue, const std::string& pValueName) throw (dab::Exception);
	float getFloat(const Json::Value& pValue, const std::string& pValueName) throw (dab::Exception);
	std::string getString(const Json::Value& pValue, const std::string& pValueName) throw (dab::Exception);

	const Json::Value& getValue(const Json::Value& pValue, int pValueIndex) throw (dab::Exception);
	int getInt(const Json::Value& pValue, unsigned int pValueIndex) throw (dab::Exception);
	float getFloat(const Json::Value& pValue, unsigned int pValueIndex) throw (dab::Exception);
	std::string getString(const Json::Value& pValue, unsigned int pValueIndex) throw (dab::Exception);

	const Json::Value& getValue(const Json::Value& pValue, const std::string& pValueName, int pValueIndex) throw (dab::Exception);
	int getInt(const Json::Value& pValue, const std::string& pValueName, unsigned int pValueIndex) throw (dab::Exception);
	float getFloat(const Json::Value& pValue, const std::string& pValueName, unsigned int pValueIndex) throw (dab::Exception);
	std::string getString(const Json::Value& pValue, const std::string& pValueName, unsigned int pValueIndex) throw (dab::Exception);

	void getValues(const Json::Value& pValue, const std::string& pValueName, std::vector<Json::Value>& pValues) throw (dab::Exception);
	void getInts(const Json::Value& pValue, const std::string& pValueName, std::vector<int>& pValues) throw (dab::Exception);
	void getFloats(const Json::Value& pValue, const std::string& pValueName, std::vector<float>& pValues) throw (dab::Exception);
	void getStrings(const Json::Value& pValue, const std::string& pValueName, std::vector<std::string>& pValues) throw (dab::Exception);

	void getInts(const Json::Value& pValue, std::vector<int>& pValues) throw (dab::Exception);
	void getFloats(const Json::Value& pValue, std::vector<float>& pValues) throw (dab::Exception);
	void getStrings(const Json::Value& pValue, std::vector<std::string>& pValues) throw (dab::Exception);

protected:
};

};