import time

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

"""
Test to compare speed using use_optimal_threadgroups = True with default (False) in custom kernel execution

We use a relatively heavy kernel called 12,000 times. The kernel solves the Bio Heat Thermal Equaion (BHTE)
"""

BHTE_header = """
#include <metal_stdlib>
using namespace metal;
#define pi M_PI_F
#define Tref 43.0

typedef float FloatingType;
"""

BHTE_body = """
#ifdef _METAL
    kernel  void BHTEFDTDKernel(device float *d_output [[ buffer(0) ]], 
                                device float *d_output2 [[ buffer(1) ]],
                                device const float *d_input [[ buffer(2) ]], 
                                device const float *d_input2 [[ buffer(3) ]],
                                device const float *d_bhArr [[ buffer(4) ]],
                                device const float *d_perfArr [[ buffer(5) ]], 
                                device const unsigned int *d_labels [[ buffer(6) ]],
                                device const float *d_Qarr [[ buffer(7) ]],
                                device const unsigned int *d_pointsMonitoring [[ buffer(8) ]],
                                device float *d_MonitorSlice [[ buffer(9) ]],
                                device float *d_Temppoints [[ buffer(10) ]],
                                constant float * floatParams [[ buffer(11) ]],
                                constant unsigned int * intparams [[ buffer(12) ]],
                                uint gid[[thread_position_in_grid]])	
    {
#endif
    #ifdef _MLX
    uint gid = thread_position_in_grid.x;
    #endif

    #define CoreTemp floatParams[0]
    #define dt floatParams[1]
    #define sonication intparams[0]
    #define outerDimx intparams[1]
    #define outerDimy intparams[2]
    #define outerDimz intparams[3]
    #define TotalStepsMonitoring intparams[4]
    #define nFactorMonitoring intparams[5]
    #define n_Step intparams[6]
    #define SelJ intparams[7]
    #define StartIndexQ intparams[8]
    #define TotalSteps intparams[9]
    
    // x,y,z indices for grid
    const int gtidx =  gid/(outerDimy*outerDimz);
    const int gtidy =  (gid - gtidx*outerDimy*outerDimz)/outerDimz;
    const int gtidz =  gid - gtidx*outerDimy*outerDimz - gtidy*outerDimz;

    unsigned int DzDy = outerDimz*outerDimy;
    
    unsigned int coord = gtidx * DzDy + gtidy * outerDimz + gtidz;
    
    float R1,R2,dtp;
    if(gtidx > 0 && gtidx < outerDimx-1 && gtidy > 0 && gtidy < outerDimy-1 && gtidz > 0 && gtidz < outerDimz-1)
    {

        const unsigned int label = d_labels[coord];

        d_output[coord] = d_input[coord] + d_bhArr[label] * 
                          (d_input[coord + 1] + d_input[coord - 1] + d_input[coord + outerDimz] + d_input[coord - outerDimz] +
                           d_input[coord + DzDy] + d_input[coord - DzDy] - 6.0 * d_input[coord]) +
                           d_perfArr[label] * (CoreTemp - d_input[coord]);
        if (sonication)
        {
            d_output[coord] += d_Qarr[coord+StartIndexQ];
        }
        
        R2 = (d_output[coord] >= Tref)?0.5:0.25; 
        R1 = (d_input[coord] >= Tref)?0.5:0.25;

        if(fabs(d_output[coord]-d_input[coord])<0.0001)
        {
            d_output2[coord] = d_input2[coord] + dt * pow((float)R1,(float)(Tref-d_input[coord]));
        }
        else
        {
            if(R1 == R2)
            {
                d_output2[coord] = d_input2[coord] + (pow((float)R2,(float)(Tref-d_output[coord])) - pow((float)R1,(float)(Tref-d_input[coord]))) / 
                                ( -(d_output[coord]-d_input[coord])/ dt * log(R1));
            }
            else
            {
                dtp = dt * (Tref - d_input[coord])/(d_output[coord] - d_input[coord]);

                d_output2[coord] = d_input2[coord] + (1 - pow((float)R1,(float)(Tref-d_input[coord]))) / (- (Tref - d_input[coord])/ dtp * log(R1)) + 
                                (pow((float)R2,(float)(Tref-d_output[coord])) - 1) / (-(d_output[coord] - Tref)/(dt - dtp) * log(R2));
            }
        }

        if (gtidy==SelJ && (n_Step % nFactorMonitoring ==0))
        {
            d_MonitorSlice[gtidx*outerDimz*TotalStepsMonitoring+gtidz*TotalStepsMonitoring+ n_Step/nFactorMonitoring] =d_output[coord];
        }

        if (d_pointsMonitoring[coord]>0)
        {
            d_Temppoints[TotalSteps*(d_pointsMonitoring[coord]-1)+n_Step]=d_output[coord];
        }
    }
    else if(gtidx < outerDimx && gtidy < outerDimy && gtidz < outerDimz)
    {
        // If only grid boundary, just pass previous value
        d_output[coord] = d_input[coord];
        d_output2[coord] = d_input2[coord];
    }
#ifdef _METAL 
}
#endif
"""


def FitSpeedCorticalLong(frequency):
    # from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
    FRef = np.array([270e3, 836e3])
    ClRef = np.array([2448.0, 2516])
    p = np.polyfit(FRef, ClRef, 1)
    return np.round(np.poly1d(p)(frequency))


def FitSpeedTrabecularLong(frequency):
    # from Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
    FRef = np.array([270e3, 836e3])
    ClRef = np.array([2140.0, 2300])
    p = np.polyfit(FRef, ClRef, 1)
    return np.round(np.poly1d(p)(frequency))


def FitSpeedCorticalShear(frequency):
    # from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc
    FRef = np.array([270e3, 836e3])
    Cs270 = np.array([1577.0, 1498.0, 1313.0]).mean()
    Cs836 = np.array([1758.0, 1674.0, 1545.0]).mean()
    CsRef = np.array([Cs270, Cs836])
    p = np.polyfit(FRef, CsRef, 1)
    return np.round(np.poly1d(p)(frequency))


def FitSpeedTrabecularShear(frequency):
    # from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc
    FRef = np.array([270e3, 836e3])
    Cs270 = np.array([1227.0, 1365.0, 1200.0]).mean()
    Cs836 = np.array([1574.0, 1252.0, 1327.0]).mean()
    CsRef = np.array([Cs270, Cs836])
    p = np.polyfit(FRef, CsRef, 1)
    return np.round(np.poly1d(p)(frequency))


def FitAttCorticalLong_Multiple(frequency, bcoeff=1, reductionFactor=0.8):
    # fitting from data obtained from
    # J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    # Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
    # IEEE transactions on ultrasonics, ferroelectrics, and frequency control 68, no. 5 (2020): 1532-1545. doi: 10.1109/TUFFC.2020.3039743

    return np.round(203.25090263 * ((frequency / 1e6) ** bcoeff) * reductionFactor)


def FitAttTrabecularLong_Multiple(frequency, bcoeff=1, reductionFactor=0.8):
    # reduction factor
    # fitting from data obtained from
    # J. Acoust. Soc. Am., Vol. 64, No. 2,  doi: 10.1121/1.382016
    # Phys Med Biol. 2011 Jan 7; 56(1): 219–250. doi :10.1088/0031-9155/56/1/014
    # IEEE transactions on ultrasonics, ferroelectrics, and frequency control 68, no. 5 (2020): 1532-1545. doi: 10.1109/TUFFC.2020.3039743
    return np.round(202.76362433 * ((frequency / 1e6) ** bcoeff) * reductionFactor)


def FitAttBoneShear(frequency, reductionFactor=1.0):
    # from Phys Med Biol. 2017 Aug 7; 62(17): 6938–6962. doi: 10.1088/1361-6560/aa7ccc
    PichardoData = (57.0 / 0.27 + 373 / 0.836) / 2
    return np.round(PichardoData * (frequency / 1e6) * reductionFactor)


def getBHTECoefficient(kappa, rho, c_t, h, t_int, dt=0.1):
    """calculates the Bioheat Transfer Equation coefficient required (time step/density*conductivity*voxelsize"""
    # get the bioheat coefficients for a tissue type -- independent of surrounding tissue types
    # dt = t_int/nt
    # h - voxel resolution - default 1e-3

    bhc_coeff = kappa * dt / (rho * c_t * h**2)
    if bhc_coeff >= (1 / 6):
        best_nt = np.ceil(6 * kappa * t_int) / (rho * c_t * h**2)
        print(
            "The conditions %f,%f,%f does not meet the C-F-L condition and may not be stable. Use nt of %f or greater."
            % (dt, t_int, bhc_coeff, best_nt)
        )
    return bhc_coeff


def getPerfusionCoefficient(w_b, c_t, blood_rho, blood_ct, dt=0.1):
    """Calculates the perfusion coefficient based on the simulation parameters and time step"""
    # get the perfusion coeff for a speicfic tissue type and time period  -- independent of surrounding tissue types
    # wb is in ml/min/kg, needs to be converted to m3/s/kg (1min/60 * 1e-6 m3/ml)

    coeff = w_b / 60 * 1.0e-6 * blood_rho * blood_ct * dt / c_t

    return coeff


def getQCoeff(rho, SoS, alpha, c_t, Absorption, h, dt):
    coeff = (
        dt / (2 * rho**2 * SoS * h * c_t) * Absorption * (1 - np.exp(-2 * h * alpha))
    )
    return coeff


def GetMaterialList(Frequency, BaselineTemperature):
    MatFreq = {}
    Material = {}
    # Density (kg/m3), LongSoS (m/s), ShearSoS (m/s), Long Att (Np/m), Shear Att (Np/m)
    Material["Water"] = np.array([1000.0, 1500.0, 0.0, 0.0, 0.0])
    Material["Cortical"] = np.array(
        [
            1896.5,
            FitSpeedCorticalLong(Frequency),
            FitSpeedCorticalShear(Frequency),
            FitAttCorticalLong_Multiple(Frequency),
            FitAttBoneShear(Frequency),
        ]
    )
    Material["Trabecular"] = np.array(
        [
            1738.0,
            FitSpeedTrabecularLong(Frequency),
            FitSpeedTrabecularShear(Frequency),
            FitAttTrabecularLong_Multiple(Frequency),
            FitAttBoneShear(Frequency),
        ]
    )
    Material["Skin"] = np.array([1116.0, 1537.0, 0.0, 2.3 * Frequency / 500e3, 0])
    Material["Brain"] = np.array([1041.0, 1562.0, 0.0, 3.45 * Frequency / 500e3, 0])

    MatFreq[Frequency] = Material

    Input = {}
    Materials = []
    for k in ["Water", "Skin", "Cortical", "Trabecular", "Brain"]:
        SelM = MatFreq[Frequency][k]
        Materials.append(
            [
                SelM[0],  # Density
                SelM[1],  # Longitudinal SOS
                SelM[2],  # Shear SOS
                SelM[3],  # Long Attenuation
                SelM[4],
            ]
        )  # Shear Attenuation
    Materials = np.array(Materials)
    MaterialList = {}
    MaterialList["Density"] = Materials[:, 0]
    MaterialList["SoS"] = Materials[:, 1]
    MaterialList["Attenuation"] = Materials[:, 3]

    # Water, Skin, Cortical, Trabecular, Brain

    # https://itis.swiss/virtual-population/tissue-properties/database/heat-capacity/
    MaterialList["SpecificHeat"] = [4178, 3391, 1313, 2274, 3630]  # (J/kg/°C)
    # https://itis.swiss/virtual-population/tissue-properties/database/thermal-conductivity/
    MaterialList["Conductivity"] = [0.6, 0.37, 0.32, 0.31, 0.51]  # (W/m/°C)
    # https://itis.swiss/virtual-population/tissue-properties/database/heat-transfer-rate/
    MaterialList["Perfusion"] = np.array([0, 106, 10, 30, 559])

    MaterialList["Absorption"] = np.array([0, 0.85, 0.16, 0.15, 0.85])

    MaterialList["InitTemperature"] = [
        BaselineTemperature,
        BaselineTemperature,
        BaselineTemperature,
        BaselineTemperature,
        BaselineTemperature,
    ]

    return MaterialList


def BHTE_mlx():
    kernel = mx.fast.metal_kernel(
        name="BHTE",
        input_names=[
            "d_input",
            "d_input2",
            "d_bhArr",
            "d_perfArr",
            "d_labels",
            "d_Qarr",
            "d_pointsMonitoring",
            "floatParams",
            "intparams",
        ],
        output_names=["d_output", "d_output2", "d_MonitorSlice", "d_Temppoints"],
        atomic_outputs=False,
        header="#define _MLX" + BHTE_header,
        source=BHTE_body,
    )

    return kernel


def BHTE(
    Pressure,
    MaterialMap,
    MaterialList,
    dx,
    TotalDurationSteps,
    nStepsOn,
    LocationMonitoring,
    nFactorMonitoring=1,
    dt=0.1,
    blood_rho=1050,
    blood_ct=3617,
    stableTemp=37.0,
    DutyCycle=1.0,
    MonitoringPointsMap=None,
    initT0=None,
    initDose=None,
    use_optimal_threadgroups=False,
):

    # Verify valid initT0, initDose values if provided
    for k in [initT0, initDose]:
        if k is not None:
            assert (
                MaterialMap.shape[0] == k.shape[0]
                and MaterialMap.shape[1] == k.shape[1]
                and MaterialMap.shape[2] == k.shape[2]
            )

            assert k.dtype == np.float32

    # Verify valid MonitoringPointsMap (i.e. grid points of interest) if provided
    if MonitoringPointsMap is not None:
        assert (
            MaterialMap.shape[0] == MonitoringPointsMap.shape[0]
            and MaterialMap.shape[1] == MonitoringPointsMap.shape[1]
            and MaterialMap.shape[2] == MonitoringPointsMap.shape[2]
        )

        assert MonitoringPointsMap.dtype == np.uint32

    # Calculate perfusion, bioheat, and Q coefficients for each material in grid
    perfArr = np.zeros(MaterialMap.max() + 1, np.float32)
    bhArr = np.zeros(MaterialMap.max() + 1, np.float32)
    if initT0 is None:
        initTemp = np.zeros(MaterialMap.shape, dtype=np.float32)
    else:
        initTemp = initT0

    Qarr = np.zeros(MaterialMap.shape, dtype=np.float32)

    for n in range(MaterialMap.max() + 1):
        bhArr[n] = getBHTECoefficient(
            MaterialList["Conductivity"][n],
            MaterialList["Density"][n],
            MaterialList["SpecificHeat"][n],
            dx,
            TotalDurationSteps,
            dt=dt,
        )
        perfArr[n] = getPerfusionCoefficient(
            MaterialList["Perfusion"][n],
            MaterialList["SpecificHeat"][n],
            blood_rho,
            blood_ct,
            dt=dt,
        )
        if initT0 is None:
            initTemp[MaterialMap == n] = MaterialList["InitTemperature"][n]
        # print(n,(MaterialMap==n).sum(),Pressure[MaterialMap==n].mean())

        Qarr[MaterialMap == n] = (
            Pressure[MaterialMap == n] ** 2
            * getQCoeff(
                MaterialList["Density"][n],
                MaterialList["SoS"][n],
                MaterialList["Attenuation"][n],
                MaterialList["SpecificHeat"][n],
                MaterialList["Absorption"][n],
                dx,
                dt,
            )
            * DutyCycle
        )

    # Dimensions of grid
    N1 = np.int32(Pressure.shape[0])
    N2 = np.int32(Pressure.shape[1])
    N3 = np.int32(Pressure.shape[2])

    # If InitDose not supplied, create array of zeros
    if initDose is None:
        initDose = np.zeros(MaterialMap.shape, dtype=np.float32)

    # Create array of temperature points to monitor based on MonitoringPointsMap
    if MonitoringPointsMap is not None:
        MonitoringPoints = MonitoringPointsMap
        TotalPointsMonitoring = np.sum((MonitoringPointsMap > 0).astype(int))
        TemperaturePoints = np.zeros(
            (TotalPointsMonitoring, TotalDurationSteps), np.float32
        )
    else:
        MonitoringPoints = np.zeros(MaterialMap.shape, dtype=np.uint32)
        TemperaturePoints = np.zeros((10), np.float32)  # just dummy array

    # Ensure valid number of time points for monitoring
    TotalStepsMonitoring = int(TotalDurationSteps / nFactorMonitoring)
    if TotalStepsMonitoring % nFactorMonitoring != 0:
        TotalStepsMonitoring += 1
    MonitorSlice = np.zeros(
        (MaterialMap.shape[0], MaterialMap.shape[2], TotalStepsMonitoring), np.float32
    )

    # Inital temperature and thermal dose
    T1 = np.zeros(initTemp.shape, dtype=np.float32)
    Dose0 = initDose
    Dose1 = np.zeros(MaterialMap.shape, dtype=np.float32)

    nFraction = int(TotalDurationSteps / 10)
    if nFraction == 0:
        nFraction = 1

    # Create mlx arrays
    d_perfArr = mx.array(perfArr)
    d_bhArr = mx.array(bhArr)
    d_Qarr = mx.array(Qarr)
    d_MaterialMap = mx.array(MaterialMap)
    d_T0 = mx.array(initTemp)
    d_Dose0 = mx.array(Dose0)
    d_MonitoringPoints = mx.array(MonitoringPoints)

    # Build program from source code
    knl = BHTE_mlx()
    sel_device = mx.default_device()

    # Float variables to be passed to kernel
    floatparams = np.array([stableTemp, dt], dtype=np.float32)
    d_floatparams = mx.array(floatparams)

    # Calculate BHTE for each time point
    t0 = time.time()
    print("N1*N2*N3", N1 * N2 * N3)
    for n in range(TotalDurationSteps):
        if n < nStepsOn:
            dUS = 1  # Ultrasound on
        else:
            dUS = 0  # Ultrasound off

        # Int variables to be passed to kernel
        intparams = np.array(
            [
                dUS,
                N1,
                N2,
                N3,
                TotalStepsMonitoring,
                nFactorMonitoring,
                n,
                LocationMonitoring,
                0,
                TotalDurationSteps,
            ],
            dtype=np.uint32,
        )
        d_intparams = mx.array(intparams)

        # At each time point, the previous output is used as the current input (e.g. d_T0 and d_T1 alternate, same with Dose0 and Dose1)
        if n % 2 == 0:
            if n == 0:
                extraakargs = {"init_value": 0}
            else:
                extraakargs = {}
            d_T1, d_Dose1, d_MonitorSlice, d_TemperaturePoints = knl(
                inputs=[
                    d_T0,
                    d_Dose0,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_Qarr,
                    d_MonitoringPoints,
                    d_floatparams,
                    d_intparams,
                ],
                output_shapes=[
                    T1.shape,
                    Dose1.shape,
                    MonitorSlice.shape,
                    TemperaturePoints.shape,
                ],
                output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
                grid=(N1 * N2 * N3, 1, 1),
                threadgroup=(256, 1, 1),
                verbose=False,
                stream=sel_device,
                use_optimal_threadgroups=use_optimal_threadgroups,
                **extraakargs
            )

        else:
            if n == 1:
                extraakargs = {"init_value": 0}
            else:
                extraakargs = {}
            d_T0, d_Dose0, d_MonitorSlice, d_TemperaturePoints = knl(
                inputs=[
                    d_T1,
                    d_Dose1,
                    d_bhArr,
                    d_perfArr,
                    d_MaterialMap,
                    d_Qarr,
                    d_MonitoringPoints,
                    d_floatparams,
                    d_intparams,
                ],
                output_shapes=[
                    initTemp.shape,
                    Dose0.shape,
                    MonitorSlice.shape,
                    TemperaturePoints.shape,
                ],
                output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
                grid=(N1 * N2 * N3, 1, 1),
                threadgroup=(256, 1, 1),
                verbose=False,
                use_optimal_threadgroups=use_optimal_threadgroups,
                **extraakargs
            )
        if n % 10 == 0:
            mx.eval(d_T1)

        # Track progress of BHTE calculation
        if n % nFraction == 0:
            print(n, TotalDurationSteps)

    # Grab final output depending on time point number
    if n % 2 == 0:
        ResTemp = d_T1
        ResDose = d_Dose1
    else:
        ResTemp = d_T0
        ResDose = d_Dose0
    print("Time to complete ", TotalDurationSteps, " loops", time.time() - t0)
    print("Done BHTE")

    # Transfer back numpy array
    t1 = time.time()
    T1 = np.asarray(ResTemp)
    Dose1 = np.asarray(ResDose)
    MonitorSlice = np.asarray(d_MonitorSlice)
    TemperaturePoints = np.asarray(d_TemperaturePoints)
    print("Time to recover results", time.time() - t1)
    print("Total time ", time.time() - t0)


if __name__ == "__main__":

    # Simulation settings to run FDTD kernel
    BaseIsppa = 5  # W/cm2
    BaselineTemperature = 37.0
    dt = 0.01  # Time step
    DutyCycle = 0.1
    DurationUS = 120  # Time that ultrasound is on
    DurationOff = 120  # Time that ultrasound is off
    Repetitions = 1  # Number of cycles of DurationUS + DurationOff
    Frequency = 1000e3  # Ultrasound frequency
    PPW = 6

    # Domain properties
    MediumSOS = 1500  # m/s - water
    MediumDensity = 1000  # kg/m3
    ShortestWavelength = MediumSOS / Frequency
    SpatialStep = ShortestWavelength / PPW

    # Limits of domain, in mm
    xfmin = -3.5e-2
    xfmax = 3.5e-2
    yfmin = -3.5e-2
    yfmax = 3.5e-2
    zfmin = 0.0
    zfmax = 12e-2

    # x,y,z points
    xfield = np.linspace(xfmin, xfmax, int(np.ceil((xfmax - xfmin) / SpatialStep) + 1))
    yfield = np.linspace(yfmin, yfmax, int(np.ceil((yfmax - yfmin) / SpatialStep) + 1))
    zfield = np.linspace(zfmin, zfmax, int(np.ceil((zfmax - zfmin) / SpatialStep) + 1))

    # Number of points in each axis
    nxf = len(xfield)
    nyf = len(yfield)
    nzf = len(zfield)
    print(nxf, nyf, nzf)

    # Grid formation
    xp, yp, zp = np.meshgrid(xfield, yfield, zfield, indexing="ij")

    # Generate pressure map for grid shaped as an ellipsoid
    radius_x, radius_y, radius_z = 0.01, 0.01, 0.02
    x_offset, y_offset, z_offset = 0.0 * xfmax, 0.0 * yfmax, 0.6 * zfmax
    ellipsoid = (
        (xp / (radius_x - x_offset)) ** 2
        + (yp / (radius_y - y_offset)) ** 2
        + ((zp - z_offset) / radius_z) ** 2
    )
    pressure = np.exp(-ellipsoid) * 500000  # Use an exponential decay for smoothness

    # Middles slice indexes
    cx = nxf // 2
    cy = nyf // 2
    cz = nzf // 2

    # Create Material Map
    MaterialMap = 4 * np.ones_like(pressure, dtype=np.uint32)  # All Brain

    # Determine material properties for specified frequency
    MaterialList = GetMaterialList(Frequency, BaselineTemperature)

    # Create masks
    SelSkin = MaterialMap == 1
    SelSkull = (MaterialMap > 1) & (MaterialMap < 4)
    SelBrain = MaterialMap == 4

    # Other BHTE parameters
    nFactorMonitoring = int(50e-3 / dt)  # we just track every 50 ms
    TotalDurationSteps = int((DurationUS + 0.001) / dt)
    nStepsOn = int(DurationUS / dt)
    TotalDurationStepsOff = int((DurationOff + 0.001) / dt)

    print("*** Running with use_optimal_threadgroups = False (default)***")

    BHTE(
        pressure,
        MaterialMap,
        MaterialList,
        (xfield[1] - xfield[0]),
        TotalDurationSteps,
        nStepsOn,
        cy,
        nFactorMonitoring=nFactorMonitoring,
        dt=dt,
        DutyCycle=DutyCycle,
        stableTemp=BaselineTemperature,
        use_optimal_threadgroups=False,
    )

    print("*** Running with use_optimal_threadgroups = True ***")

    BHTE(
        pressure,
        MaterialMap,
        MaterialList,
        (xfield[1] - xfield[0]),
        TotalDurationSteps,
        nStepsOn,
        cy,
        nFactorMonitoring=nFactorMonitoring,
        dt=dt,
        DutyCycle=DutyCycle,
        stableTemp=BaselineTemperature,
        use_optimal_threadgroups=True,
    )
