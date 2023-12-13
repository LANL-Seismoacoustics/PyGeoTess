/*
 * GeoTessAmplitudeDeveloper.h
 *
 *  Created on: Apr 26, 2022
 *      Author: sballar
 */

#ifndef INCLUDE_GEOTESSMODELAMPLITUDEDEVELOPER_H_
#define INCLUDE_GEOTESSMODELAMPLITUDEDEVELOPER_H_

// **** _SYSTEM INCLUDES_ ******************************************************

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <vector>

// use standard library objects
using namespace std;

// **** _LOCAL INCLUDES_ *******************************************************

#include "IFStreamAscii.h"
#include "IFStreamBinary.h"

#include "GeoTessModel.h"
#include "GeoTessModelAmplitude.h"
#include "GeoTessGreatCircle.h"
#include "GeoTessPointMap.h"
#include "GeoTessMetaData.h"

// **** _BEGIN GEOTESS NAMESPACE_ **********************************************

namespace geotess {

// **** _FORWARD REFERENCES_ ***************************************************

// **** _CLASS DEFINITION_ *****************************************************

/**
 * @author sballar
 *
 */
class GeoTessModelAmplitudeDeveloper  : virtual public GeoTessModelAmplitude
{
private:

	/**
	 * Override GeoTessModel::writeModelAscii()
	 * Applications don't call this protected method directly.
	 * It is call from GeoTessModel.writeModel().
	 *
	 * @param output the output ascii stream to which model is written.
	 * @param gridFileName
	 *
	 */
	void writeModelAscii(IFStreamAscii& output, const string& gridFileName);

	/**
	 * Override GeoTessModel::writeModelBinary()
	 * Applications don't call this protected method directly.
	 * It is call from GeoTessModel.writeModel().
	 *
	 * @param output the output ascii stream to which model is written.
	 * @param gridFileName
	 */
	void writeModelBinary(IFStreamBinary& output, const string& gridFileName);


public:

	/**
	 * Default constructor.
	 *
	 */
	GeoTessModelAmplitudeDeveloper();

	/**
	 * Construct a new GeoTessModel object and populate it with information from
	 * the specified file.
	 *
	 * @param modelInputFile
	 *            name of file containing the model.
	 * @param relativeGridPath
	 *            the relative path from the directory where the model is stored
	 *            to the directory where the grid is stored. Often, the model
	 *            and grid are stored together in the same file in which case
	 *            this parameter is ignored. Sometimes, however, the grid is
	 *            stored in a separate file and only the name of the grid file
	 *            (without path information) is stored in the model file. In
	 *            this case, the code needs to know which directory to search
	 *            for the grid file. The default is "" (empty string), which
	 *            will cause the code to search for the grid file in the same
	 *            directory in which the model file resides. Bottom line is that
	 *            the default value is appropriate when the grid is stored in
	 *            the same file as the model, or the model file is in the same
	 *            directory as the model file.
	 */
	GeoTessModelAmplitudeDeveloper(const string& modelInputFile, const string& relativeGridPath);

	/**
	 * Construct a new GeoTessModel object and populate it with information from
	 * the specified file.
	 *
	 * <p>relativeGridPath is assumed to be "" (empty string), which is appropriate
	 * when the grid information is stored in the same file as the model or when
	 * the grid is stored in a separate file located in the same directory as the
	 * model file.
	 *
	 * @param modelInputFile
	 *            name of file containing the model.
	 */
	GeoTessModelAmplitudeDeveloper(const string& modelInputFile);

	/**
	 * Default constructor.
	 *
	 * @param attributeFilter the indexes of available attributes that should
	 *            be loaded into memory.
	 */
	GeoTessModelAmplitudeDeveloper(vector<int>& attributeFilter);

	/**
	 * Construct a new GeoTessModel object and populate it with information from
	 * the specified file.
	 *
	 * @param modelInputFile
	 *            name of file containing the model.
	 * @param relativeGridPath
	 *            the relative path from the directory where the model is stored
	 *            to the directory where the grid is stored. Often, the model
	 *            and grid are stored together in the same file in which case
	 *            this parameter is ignored. Sometimes, however, the grid is
	 *            stored in a separate file and only the name of the grid file
	 *            (without path information) is stored in the model file. In
	 *            this case, the code needs to know which directory to search
	 *            for the grid file. The default is "" (empty string), which
	 *            will cause the code to search for the grid file in the same
	 *            directory in which the model file resides. Bottom line is that
	 *            the default value is appropriate when the grid is stored in
	 *            the same file as the model, or the model file is in the same
	 *            directory as the model file.
	 * @param attributeFilter the indexes of available attributes that should
	 *            be loaded into memory.
	 */
	GeoTessModelAmplitudeDeveloper(const string& modelInputFile, const string& relativeGridPath,
			vector<int>& attributeFilter);

	/**
	 * Construct a new GeoTessModel object and populate it with information from
	 * the specified file.
	 *
	 * <p>relativeGridPath is assumed to be "" (empty string), which is appropriate
	 * when the grid information is stored in the same file as the model or when
	 * the grid is stored in a separate file located in the same directory as the
	 * model file.
	 *
	 * @param modelInputFile
	 *            name of file containing the model.
	 * @param attributeFilter the indexes of available attributes that should
	 *            be loaded into memory.
	 */
	GeoTessModelAmplitudeDeveloper(const string& modelInputFile, vector<int>& attributeFilter);

	/**
	 * Parameterized constructor, specifying the grid and metadata for the
	 * model. The grid is constructed and the data structures are initialized
	 * based on information supplied in metadata. The data structures are not
	 * populated with any information however (all Profiles are NULL). The
	 * application should populate the new model's Profiles after this
	 * constructor completes.
	 *
	 * <p>
	 * Before calling this constructor, the supplied MetaData object must be
	 * populated with required information by calling the following MetaData
	 * methods:
	 * <ul>
	 * <li>setDescription()
	 * <li>setLayerNames()
	 * <li>setAttributes()
	 * <li>setDataType()
	 * <li>setLayerTessIds() (only required if grid has more than one
	 * multi-level tessellation)
	 * </ul>
	 *
	 * @param gridFileName
	 *            name of file from which to load the grid.
	 * @param metaData
	 *            MetaData the new GeoTessModel instantiates a reference to the
	 *            supplied metaData. No copy is made.
	 * @throws GeoTessException
	 *             if metadata is incomplete.
	 */
	GeoTessModelAmplitudeDeveloper(const string& gridFileName, GeoTessMetaData* metaData);

	/**
	 * Parameterized constructor, specifying the grid and metadata for the
	 * model. The grid is constructed and the data structures are initialized
	 * based on information supplied in metadata. The data structures are not
	 * populated with any information however (all Profiles are NULL). The
	 * application should populate the new model's Profiles after this
	 * constructor completes.
	 *
	 * <p>
	 * Before calling this constructor, the supplied MetaData object must be
	 * populated with required information by calling the following MetaData
	 * methods:
	 * <ul>
	 * <li>setDescription()
	 * <li>setLayerNames()
	 * <li>setAttributes()
	 * <li>setDataType()
	 * <li>setLayerTessIds() (only required if grid has more than one
	 * multi-level tessellation)
	 * <li>setSoftwareVersion()
	 * <li>setGenerationDate()
	 * </ul>
	 *
	 * @param grid
	 *            a pointer to the GeoTessGrid that will support this
	 *            GeoTessModel.  GeoTessModel assumes ownership of the
	 *            supplied grid object and will delete it when it is
	 *            done with it.
	 * @param metaData
	 *            MetaData the new GeoTessModel instantiates a reference to the
	 *            supplied metaData. No copy is made.
	 * @throws GeoTessException
	 *             if metadata is incomplete.
	 */
	GeoTessModelAmplitudeDeveloper(GeoTessGrid* grid, GeoTessMetaData* metaData);

	/**
	 * Construct a new GeoTessModelAmplitude by making a deep copy of an
	 * existing GeoTessModel and initializing the extra data with default
	 * values.
	 * @param baseModel pointer to an existing GeoTessModel.
	 */
	GeoTessModelAmplitudeDeveloper(GeoTessModel* baseModel);

	/**
	 * Destructor.
	 */
	virtual ~GeoTessModelAmplitudeDeveloper();

	/**
	 * Specify the file format for this model, either 1 or 2.
	 * If 1, only sitetrans info is appended.  If 2, sitetrans and spreading
	 * info is appended.
	 * @param _fileformat
	 * @throws GeoTessException if _fileformat is not 1 or 2
	 */
	void setFileformat(const int& _fileformat) {
		if (_fileformat != 1 && _fileformat != 2)
		{
			ostringstream os;
			os << endl << "ERROR in GeoTessModelAmplitudeDeveloper::setFileformat()"<< endl
					<< "fileformat must be either 1 or 2";
			throw GeoTessException(os, __FILE__, __LINE__, 1);
		}
		fileformat = _fileformat;
	}

	/**
	 * Specify the phase supported by this model.
	 * This method sets the values of the phase-dependent coefficients
	 * used by method getSpreadingXY().
	 * @param _phase the phase supported by this model
	 * @throws GeoTessException if phase is not one of [ Pn, Sn, Pg, Lg ]
	 */
	void setPhase(const string& _phase) {
		if (_phase != "Pn" && _phase != "Sn" && _phase != "Pg" && _phase != "Lg")
		{
			ostringstream os;
			os << endl << "ERROR in GeoTessModelAmplitudeDeveloper::setPhase()"<< endl
					<< "phase " << _phase << " is not supported." << endl;
			throw GeoTessException(os, __FILE__, __LINE__, 1);
		}
		phase = _phase;
	}

	/**
	 * Setter specifying a map from station -> channel -> band -> siteTran
	 * @param _siteTrans
	 */
	void setSiteTrans(map<string, map<string, map<string, float> > >& _siteTrans)
	{ siteTrans = _siteTrans; }

	/**
	 * Refresh the frequencyMap by iterating over all the model attribute names and extracting
	 * the frequency band, if possible.  Then iterate over all the site trans and search for
	 * unique frequency bands.  It is only necessary to do this after making any additions to
	 * the set of unique frequency bands by, for example, adding siteTrans items.
	 * @return  a reference to the map from a string representation of a frequency band, to the
	 * same information stored as a 2 element array of doubles.
	 */
	map<string, vector<float> >&  refreshFrequencyMap();

	/**
	 * Set spreadmode string as above
	 * check for valid string
	 */
	void setInterpolatorType(const string& _interpolatorType) {
		if (_interpolatorType != "LINEAR" && _interpolatorType != "NATURAL_NEIGHBOR")
		{
			ostringstream os;
			os << endl << "ERROR in GeoTessModelAmplitude::setInterpolatorType( "<< _interpolatorType << " )"<< endl
					<< _interpolatorType << " is not recognized.  Must specify either LINEAR or NATURAL_NEIGHBOR" << endl;
			throw GeoTessException(os, __FILE__, __LINE__, 12001);
		}
		interpolatorType = _interpolatorType;
	}

	/**
	 * set dxkm = path integration interval (km) through 2D tesselation
	 * @param _dxkm path integration interval in km
	 */
	void setDxKm(const double& _dxkm) { dxkm = _dxkm; }

	/**
	 * Set spreading mode.  Must be one of Standard, XYPn, XYPnSn, XY1seg, XY2seg
	 * @param _spreadmode
	 */
	void setSpreadMode(const string& _spreadmode) {
		if (_spreadmode != "Standard" && _spreadmode != "XYPn" && _spreadmode != "XYPnSn"
				&& _spreadmode != "XY1seg" && _spreadmode != "XY2seg")
		{
			ostringstream os;
			os << endl << "ERROR in GeoTessModelAmplitude::setSpreadMode( "<< _spreadmode << " )"<< endl
					<< _spreadmode << " is not recognized.  Must specify one of: Standard, XYPn, XYPnSn, XY1seg, XY2seg." << endl;
			throw GeoTessException(os, __FILE__, __LINE__, 12001);
		}
		spreadmode = _spreadmode;
	}

	/**
	 * set MDAC sigma =  earthquake apparent stress Pa MKS
	 * @param MDAC sigma
	 */
	void setMDACSigma(const double& _sigma) { sigma = _sigma; }

	/**
	 * set MDAC psi = stress scaling (unitless)
	 * @param MDAC psi
	 */
	void setMDACPsi(const double& _psi) { psi = _psi; }

	/**
	 * set MDAC zeta = earthquake P/S corner frequency ratio (unitless)
	 * check valid range?
	 */
	void setMDACZeta(const double& _zeta) { zeta = _zeta; kfact = NaN_DOUBLE; }

	/**
	 * set MDAC Moref = earthquake reference moment Nm MKS
	 * check valid range? watch for wrong units***
	 */
	void setMDACM0ref(const double& _m0ref) { m0ref = _m0ref; }

	/**
	 * set MDAC radpatp = average/relative earthquake P radiation (unitless?)
	 * check valid range? watch for wrong units***?
	 */
	void setMDACRadpatP(const double& _radpatp) { radpatp = _radpatp; kfact = NaN_DOUBLE; }

	/**
	 * set MDAC radpats = average/relative earthquake S radiation (unitless?)
	 * check valid range? watch for wrong units***?
	 */
	void setMDACRadpatS(const double& _radpats) { radpats = _radpats; kfact = NaN_DOUBLE; }

	/**
	 * set MDAC alphas = source region compressional velocity m/s MKS
	 * check valid range? watch for wrong units***
	 */
	void setMDACAlphaS(const double& _alphas) { alphas = _alphas; kfact = NaN_DOUBLE; }

	/**
	 * set MDAC betas = source region shear velocity m/s MKS
	 * check valid range? watch for wrong units***
	 */
	void setMDACBetaS(const double& _betas) { betas = _betas; kfact = NaN_DOUBLE; }

	/**
	 * set MDAC rhos = source region density kg/m3 MKS
	 * check valid range? watch for wrong units***
	 */
	void setMDACRhoS(const double& _rhos) { rhos = _rhos; }

	/**
	 * set p1 = local distance spreading (unitless)
	 * check valid range?
	 */
	void setSpreadP1(const double& _p1) { p1 = _p1; }

	/**
	 * set p2 = regional distance spreading (unitless)
	 * check valid range?
	 */
	void setSpreadP2(const double& _p2) { p2 = _p2; }

	/**
	 * set xc = critical distance = transition between local and regional spreading (km)
	 * check valid range?
	 */
	void setSpreadXC(const double& _xc) { xc = _xc; }

	/**
	 * set xt = transition factor between local and regional spreading (km)
	 * xt = 1 reduces to Street-Herrman spreading, > 1 smoother transition
	 * used for coda spreading
	 * check valid range? >= 1
	 */
	void setSpreadXT(const double& _xt) { xt = _xt; }

	/**
	 * Set path velocity in furlongs/fortnight
	 * @param _pathvel path velocity in furlongs/fortnight
	 */
	void setPathVelocity(const double& _pathvel) { pathvel = _pathvel; }

};

}// end namespace geotess





#endif /* INCLUDE_GEOTESSMODELAMPLITUDEDEVELOPER_H_ */
