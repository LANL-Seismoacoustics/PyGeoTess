//- ****************************************************************************************
//- Notice: This computer software was prepared by Sandia Corporation, hereinafter
//- the Contractor, under Contract DE-AC04-94AL85000 with the Department of Energy (DOE).
//- All rights in the computer software are reserved by DOE on behalf of the United
//- States Government and the Contractor as provided in the Contract. You are authorized
//- to use this computer software for Governmental purposes but it is not to be released
//- or distributed to the public.
//- NEITHER THE U.S. GOVERNMENT NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED,
//- OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//- This notice including this sentence must appear on any copies of this computer software.
//- ****************************************************************************************

#include "GeoTessModelAmplitudeDeveloper.h"

// **** _BEGIN GEOTESS NAMESPACE_ **********************************************

namespace geotess {

// **** _EXPLICIT TEMPLATE INSTANTIATIONS_ *************************************

// **** _STATIC INITIALIZATIONS_************************************************

// **** _FUNCTION IMPLEMENTATIONS_ *********************************************

/**
 * This class is an example of a class that extends GeoTessModel.
 * It does everything a GeoTessModel can do but adds an extra
 * data item to the model.  In this example, the extra data is
 * just a simple String, but in real models that extend
 * GeoTessModel, it could be anything.
 *
 * <p>Classes that extend GeoTessModel should provide
 * implementations of all the GeoTessModel constructors with
 * the first thing that they do is call the super class
 * constructor.
 * <p>In addition, classes that extend GeoTessModel should
 * override 4 IO functions: loadModelBinary, writeModelBinary,
 * loadModelAscii, writeModelAscii.
 * See examples below.
 * <p>The first thing that these methods do is call the super
 * class implementations to read/write the standard
 * GeoTessModel information.  After that, the methods
 * may read/write the application specific data from/to
 * the end of the standard GeoTessModel file.
 * @author sballar
 *
 */

// Scott:  in every constructor, you have to provide default values for
// all the spreading parameters that are of type string and double.
// It is best to specify them in the order they are declared in
// GeoTessModelAmplitude.h or the compiler will issue warnings.
// I recommend setting default value of all doubles to NaN_DOUBLE
// that way if any value does not get set and is involved in any
// calculations, the result of the calculation will NaN, which will
// be easily detected in tests.

/**
 * Extend basemodel constructor by initializing extraData.
 */
GeoTessModelAmplitudeDeveloper::GeoTessModelAmplitudeDeveloper()
: GeoTessModelAmplitude()
{ }

/**
 * Extend basemodel constructor by initializing extraData.
 */
GeoTessModelAmplitudeDeveloper::GeoTessModelAmplitudeDeveloper(const string& modelInputFile, const string& relativeGridPath)
: GeoTessModelAmplitude(modelInputFile, relativeGridPath)
{ }

/**
 * Extend basemodel constructor by initializing extraData.
 */
GeoTessModelAmplitudeDeveloper::GeoTessModelAmplitudeDeveloper(const string& modelInputFile)
: GeoTessModelAmplitude(modelInputFile)
{ }

/**
 * Extend basemodel constructor by initializing extraData.
 */
GeoTessModelAmplitudeDeveloper::GeoTessModelAmplitudeDeveloper(vector<int>& attributeFilter)
: GeoTessModelAmplitude(attributeFilter)
{ }

/**
 * Extend basemodel constructor by initializing extraData.
 */
GeoTessModelAmplitudeDeveloper::GeoTessModelAmplitudeDeveloper(const string& modelInputFile, const string& relativeGridPath,
		vector<int>& attributeFilter)
: GeoTessModelAmplitude(modelInputFile, relativeGridPath, attributeFilter)
{ }

/**
 * Extend basemodel constructor by initializing extraData.
 */
GeoTessModelAmplitudeDeveloper::GeoTessModelAmplitudeDeveloper(const string& modelInputFile, vector<int>& attributeFilter)
: GeoTessModelAmplitude(modelInputFile, attributeFilter)
{ }

/**
 * Extend basemodel constructor by initializing extraData.
 */
GeoTessModelAmplitudeDeveloper::GeoTessModelAmplitudeDeveloper(const string& gridFileName, GeoTessMetaData* metaData)
: GeoTessModelAmplitude(gridFileName, metaData)
{ }

/**
 * Extend basemodel constructor by initializing extraData.
 */
GeoTessModelAmplitudeDeveloper::GeoTessModelAmplitudeDeveloper(GeoTessGrid* grid, GeoTessMetaData* metaData)
: GeoTessModelAmplitude(grid, metaData)
{ }

/**
 * Construct a new GeoTessModelAmplitude by making a deep copy of an
 * existing GeoTessModel and initializing the extra data with default
 * values.
 * @param baseModel pointer to an existing GeoTessModel.
 */
GeoTessModelAmplitudeDeveloper::GeoTessModelAmplitudeDeveloper(GeoTessModel* baseModel)
: GeoTessModelAmplitude(baseModel)
{ }

/**
 * Destructor.
 */
GeoTessModelAmplitudeDeveloper::~GeoTessModelAmplitudeDeveloper()
{
	// if this derived class allocated any memory,
	// it must be deleted here.
}

map<string, vector<float> >& GeoTessModelAmplitudeDeveloper::refreshFrequencyMap()
{
	frequencyMap.clear();
	string band, attribute;

	// first iterate over all the model attributes and extract frequency bands from the attribute names
	if (getMetaData().getAttributeIndex("Q0") < 0)
		for (int i=0; i<getNAttributes(); ++i)
		{
			attribute = getMetaData().getAttributeName(i);
			if (attribute.find("Q[") == 0 && attribute.find("]") == attribute.length()-1)
			{
				band = attribute.substr(2, attribute.length()-3);
				vector<string> tokens(2);
				CPPUtils::tokenizeString(band, "_", tokens);

				frequencyMap.insert(pair<string, vector<float> >(band, vector<float>(2)));
				frequencyMap[band][0] = CPPUtils::stof(tokens[0]);
				frequencyMap[band][1] = CPPUtils::stof(tokens[1]);
			}
		}

	// now iterate over all the site terms searching for frequency bands.

	map<string, map<string, map<string, float> > >::iterator it1;
	map<string, map<string, float> >::iterator it2;
	map<string, float>::iterator it3;

	for (it1 = siteTrans.begin(); it1 != siteTrans.end(); ++it1)
		for (it2 = (*it1).second.begin(); it2 != (*it1).second.end(); ++it2)
			for (it3 = (*it2).second.begin(); it3 != (*it2).second.end(); ++it3)
				if (frequencyMap.find((*it3).first) == frequencyMap.end())
				{
					band = (*it3).first;
					vector<string> tokens(2);
					CPPUtils::tokenizeString(band, "_", tokens);

					frequencyMap.insert(pair<string, vector<float> >(band, vector<float>(2)));
					frequencyMap[band][0] = CPPUtils::stof(tokens[0]);
					frequencyMap[band][1] = CPPUtils::stof(tokens[1]);
				}

	return frequencyMap;
}

/**
 * Override GeoTessModel::writeModelAscii()
 * Applications don't call this protected method directly.
 * It is call from GeoTessModel.writeModel().
 *
 * @param output the output ascii stream to which model is written.
 * @param gridFileName
 *
 */
void GeoTessModelAmplitudeDeveloper::writeModelAscii(IFStreamAscii& output, const string& gridFileName)
{
	if (fileformat > 1 && !isSpreadingValid())
	{
		ostringstream os;
		os << endl << "ERROR in GeoTessModelAmplitudeDeveloper::writeModelAscii()"<< endl
				<< "invalid spreading parameters:" << endl <<
				toString() << endl;
		throw GeoTessException(os, __FILE__, __LINE__, 12003);
	}

	// write all the base class GeoTessModel information to output.
	GeoTessModel::writeModelAscii(output, gridFileName);

	// output pointer is now positioned just past the end of the base class
	// GeoTessModel information.  The derived class can now write whatever is appropriate.

	// it is good practice, but not required, to store the class
	// name as the first thing added by the extending class.
	output.writeStringNL(class_name());

	// it is good practice, but not required, to store a format
	// version number as the second thing added by the extending class.
	// With this information, if the format of information added by the derived class
	// changes in a future release it may be possible to make the class backward compatible.

	output.writeIntNL(fileformat);

	output.writeStringNL(phase);
	output.writeIntNL(getNSiteTrans());

	string station, channel, band;
	float siteTran;
	map<string, map<string, map<string, float> > >::iterator it1;
	map<string, map<string, float> >::iterator it2;
	map<string, float>::iterator it3;

	ostringstream os;
	os << scientific;
	os.width(10);
	os.precision(8);

	for (it1 = siteTrans.begin(); it1 != siteTrans.end(); ++it1)
	{
		station = (*it1).first;
		for (it2 = (*it1).second.begin(); it2 != (*it1).second.end(); ++it2)
		{
			channel = (*it2).first;
			for (it3 = (*it2).second.begin(); it3 != (*it2).second.end(); ++it3)
			{
				band = (*it3).first;
				siteTran = (*it3).second;
				if (!std::isnan(siteTran))
				{
					os << left
							<< setw(8) << station << " "
							<< setw(8) << channel << " "
							<< setw(12) << band << " "
							<< right << siteTran;

					output.writeStringNL(os.str());
					os.str("");
					os.clear();
				}
			}
		}
	}

	if (fileformat >= 2)
	{
		// start output of spreading attributes.
		// first write output to the os buffer

		os << "interpolatorType = " << interpolatorType << endl;
		os << "dxkm = " << dxkm << endl;

		os << "spreadmode = " << spreadmode << endl;
		os << "rhos = " << rhos << endl;
		os << "alphas = " << alphas << endl;
		os << "betas = " << betas << endl;
		os << "radpatp = " << radpatp << endl;
		os << "radpats = " << radpats << endl;
		os << "m0ref = " << m0ref << endl;
		os << "zeta = " << zeta << endl;
		os << "sigma = " << sigma << endl;
		os << "psi = " << psi << endl;
		os << "xc = " << xc << endl;
		os << "xt = " << xt << endl;
		os << "p1 = " << p1 << endl;
		os << "p2 = " << p2 << endl;
		os << "pathvel = " << pathvel << endl;

		// write the buffer to output and clear the buffer
		output.writeString(os.str());
		os.clear();
	}
}

/**
 * Override GeoTessModel::writeModelBinary()
 * Applications don't call this protected method directly.
 * It is call from GeoTessModel.writeModel().
 *
 * @param output the output ascii stream to which model is written.
 * @param gridFileName
 */
void GeoTessModelAmplitudeDeveloper::writeModelBinary(IFStreamBinary& output, const string& gridFileName)
{
	if (fileformat > 1 && !isSpreadingValid())
	{
		ostringstream os;
		os << endl << "ERROR in GeoTessModelAmplitudeDeveloper::writeModelAscii()"<< endl
				<< "invalid spreading parameters:" << endl <<
				toString() << endl;
		throw GeoTessException(os, __FILE__, __LINE__, 12003);
	}

	// write all the base class GeoTessModel information to output.
	GeoTessModel::writeModelBinary(output, gridFileName);

	// output pointer is now positioned just past the end of the base class
	// GeoTessModel information.  The derived class can now write whatever is appropriate.

	// store the class name as the first thing added by the extending class.
	output.writeString(class_name());

	// store a format version number as the second thing added by the extending class.
	// With this information, if the format of information added by the derived class
	// changes in a future release it may be possible to make the class backward compatible.

	output.writeInt(fileformat);

	output.writeString(phase);

	output.writeInt(getNStations());

	map<string, map<string, map<string, float> > >::iterator it1;
	map<string, map<string, float> >::iterator it2;
	map<string, float>::iterator it3;

	for (it1 = siteTrans.begin(); it1 != siteTrans.end(); ++it1)
	{
		output.writeString((*it1).first); // station
		output.writeInt((*it1).second.size());
		for (it2 = (*it1).second.begin(); it2 != (*it1).second.end(); ++it2)
		{
			output.writeString((*it2).first); // channel
			output.writeInt((*it2).second.size());
			for (it3 = (*it2).second.begin(); it3 != (*it2).second.end(); ++it3)
			{
				output.writeString((*it3).first); // band
				output.writeFloat((*it3).second);
			}
		}
	}

	if (fileformat >= 2)
	{
		// start output of spreading attributes.

		output.writeString(interpolatorType);
		output.writeDouble(dxkm);

		output.writeString(getSpreadMode());
		output.writeDouble(rhos);
		output.writeDouble(alphas);
		output.writeDouble(betas);
		output.writeDouble(radpatp);
		output.writeDouble(radpats);
		output.writeDouble(m0ref);
		output.writeDouble(zeta);
		output.writeDouble(sigma);
		output.writeDouble(psi);
		output.writeDouble(xc);
		output.writeDouble(xt);
		output.writeDouble(p1);
		output.writeDouble(p2);
		output.writeDouble(pathvel);
	}
}

} // end namespace geotess
