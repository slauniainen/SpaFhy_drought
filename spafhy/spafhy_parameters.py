# -*- coding: utf-8 -*-
"""
PARAMETERS OF SPAFHY

Created on Mon Jun 25 18:34:12 2018

@author: slauniai
"""
#pathlib.Path converts '/' to '\' on windows,
#i.e. one can always use '/' in file path
import pathlib
#SpaFHy  v. 250618 parameter files for different catchments

def parameters():
    pgen = {'catchment_id': 3,
            'gis_folder': str(pathlib.Path('C:/Projects/TRAM\data/SpaFHyData/Vihti/')),
            'forcing_file': str(pathlib.Path('C:/Projects/TRAM\data/SpaFHyData/Vihti/vihti_FMI_10x10.csv')),
            'runoff_file': None, #str(pathlib.Path('../SpaFHyData/Runoff/Runoffs_SVEcatchments_mmd.csv')),
            #'soil_file': '../Spathy/Spathy/Spathy_hyde_example/soiltypes.ini',
            'ncf_file': str(pathlib.Path('c:/tempdata/spafhy/test.nc')),
            'start_date': '2013-01-01',
            'end_date': '2014-12-31',
            'spinup_end': '2013-12-31',
            'dt': 86400.0,
            'spatial_cpy': True,
            'spatial_soil': True     
           }
    
    # canopygrid
    pcpy = {'loc': {'lat': 61.4, 'lon': 23.7},
            'flow' : { # flow field
                     'zmeas': 2.0,
                     'zground': 0.5,
                     'zo_ground': 0.01
                     },
            'interc': { # interception
                        'wmax': 1.5, # storage capacity for rain (mm/LAI)
                        'wmaxsnow': 4.5, # storage capacity for snow (mm/LAI),
                        },
            'snow': {
                    # degree-day snow model
                    'kmelt': 2.8934e-05, # melt coefficient in open (mm/s)
                    'kfreeze': 5.79e-6, # freezing coefficient (mm/s)
                    'r': 0.05 # maximum fraction of liquid in snow (-)
                    },
                    
            'physpara': {
                        # canopy conductance
                        'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                        'g1_conif': 2.1, # stomatal parameter, conifers
                        'g1_decid': 3.5, # stomatal parameter, deciduous
                        'q50': 50.0, # light response parameter (Wm-2)
                        'kp': 0.6, # light attenuation parameter (-)
                        'rw': 0.20, # critical value for REW (-),
                        'rwmin': 0.02, # minimum relative conductance (-)
                        # soil evaporation
                        'gsoil': 1e-2 # soil surface conductance if soil is fully wet (m/s)
                        },
            'phenopara': {
                        #seasonal cycle of physiology: smax [degC], tau[d], xo[degC],fmin[-](residual photocapasity)
                        'smax': 18.5, # degC
                        'tau': 13.0, # days
                        'xo': -4.0, # degC
                        'fmin': 0.05, # minimum photosynthetic capacity in winter (-)
                        # deciduos phenology
                        'lai_decid_min': 0.1, # minimum relative LAI (-)
                        'ddo': 45.0, # degree-days for bud-burst (5degC threshold)
                        'ddur': 23.0, # duration of leaf development (days)
                        'sdl': 9.0, # daylength for senescence start (h)
                        'sdur': 30.0, # duration of leaf senescence (days),
                         },
            'state': {
                       'lai_conif': 3.5, # conifer 1-sided LAI (m2 m-2)
                       'lai_decid_max': 0.5, # maximum annual deciduous 1-sided LAI (m2 m-2): 
                       'hc': 16.0, # canopy height (m)
                       'cf': 0.6, # canopy closure fraction (-)
                       #initial state of canopy storage [mm] and snow water equivalent [mm]
                       'w': 0.0, # canopy storage mm
                       'swe': 0.0, # snow water equivalent mm
                       }
            }

        
    # BUCKET
    pbu = {'depth': 0.4,  # root zone depth (m)
           # following soil properties are used if spatial_soil = False
           'poros': 0.43, # porosity (-)
           'fc': 0.33, # field capacity (-)
           'wp': 0.13,	 # wilting point (-)
           'ksat': 2.0e-6, 
           'beta': 4.7,
           #organic (moss) layer
           'org_depth': 0.04, # depth of organic top layer (m)
           'org_poros': 0.9, # porosity (-)
           'org_fc': 0.3, # field capacity (-)
           'org_rw': 0.24, # critical vol. moisture content (-) for decreasing phase in Ef
           'maxpond': 0.0, # max ponding allowed (m)
           #initial states: rootzone and toplayer soil saturation ratio [-] and pond storage [m]
           'rootzone_sat': 0.6, # root zone saturation ratio (-)
           'org_sat': 1.0, # organic top layer saturation ratio (-)
           'pond_sto': 0.0 # pond storage
           }
    
    # TOPMODEL
    ptop = {'dt': 86400.0, # timestep (s)
            'm': 0.01, # scaling depth (m)
            'ko': 0.001, # transmissivity parameter (ms-1)
            'twi_cutoff': 99.5,  # cutoff of cumulative twi distribution (%)
            'so': 0.05 # initial saturation deficit (m)
           }
    
    return pgen, pcpy, pbu, ptop

def parameters_vihti():
    # Vihti MEOLO / EFFORTE test site.
    
    pgen = {'catchment_id': 'vihti',
            'gis_folder': str(pathlib.Path('C:/Projects/TRAM\data/SpaFHyData/Vihti/')),
            'forcing_file': str(pathlib.Path('C:/Projects/TRAM\data/SpaFHyData/Vihti/vihti_FMI_10x10.csv')),
            'runoff_file': None,
            #'soil_file': r'c:\Repositories\Spathy\ini\soiltypes.ini',
            #Results are in netCDF4 file
            'ncf_file': str(pathlib.Path('results/Vihti.nc')),
            #SpaFHy model itsels, state(s), in pickle file
            'pickle_file':str(pathlib.Path('results/SpaFHyVihti.pk')),
            #SpaFHy topgraphic background map, pickle file
            'topographic_map_file':str(pathlib.Path('../SpaFHyData/Vihti/rasters/vihti_rasters.pk')),
            'start_date': '2015-01-01',
            'end_date': '2016-11-30',
            'spinup_end': '2015-12-31',
            'dt': 86400.0,
            'spatial_cpy': True,
            'spatial_soil': True     
           }
    
    # canopygrid
    pcpy = {'loc': {'lat': 61.41, 'lon': 24.35},
            'flow' : { # flow field
                     'zmeas': 2.0,
                     'zground': 0.5,
                     'zo_ground': 0.01
                     },
            'interc': { # interception
                        'wmax': 1.5, # storage capacity for rain (mm/LAI)
                        'wmaxsnow': 4.5, # storage capacity for snow (mm/LAI),
                        },
            'snow': {
                    # degree-day snow model
                    'kmelt': 2.8934e-05, # melt coefficient in open (mm/s)
                    'kfreeze': 5.79e-6, # freezing coefficient (mm/s)
                    'r': 0.05 # maximum fraction of liquid in snow (-)
                    },
                    
            'physpara': {
                        # canopy conductance
                        'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                        'g1_conif': 2.1, # stomatal parameter, conifers
                        'g1_decid': 3.5, # stomatal parameter, deciduous
                        'q50': 50.0, # light response parameter (Wm-2)
                        'kp': 0.6, # light attenuation parameter (-)
                        'rw': 0.20, # critical value for REW (-),
                        'rwmin': 0.02, # minimum relative conductance (-)
                        # soil evaporation
                        'gsoil': 1e-2 # soil surface conductance if soil is fully wet (m/s)
                        },
            'phenopara': {
                        #seasonal cycle of physiology: smax [degC], tau[d], xo[degC],fmin[-](residual photocapasity)
                        'smax': 18.5, # degC
                        'tau': 13.0, # days
                        'xo': -4.0, # degC
                        'fmin': 0.05, # minimum photosynthetic capacity in winter (-)
                        # deciduos phenology
                        'lai_decid_min': 0.1, # minimum relative LAI (-)
                        'ddo': 45.0, # degree-days for bud-burst (5degC threshold)
                        'ddur': 23.0, # duration of leaf development (days)
                        'sdl': 9.0, # daylength for senescence start (h)
                        'sdur': 30.0, # duration of leaf senescence (days),
                         },
            'state': {
                       'lai_conif': 3.5, # conifer 1-sided LAI (m2 m-2)
                       'lai_decid_max': 0.5, # maximum annual deciduous 1-sided LAI (m2 m-2): 
                       'hc': 16.0, # canopy height (m)
                       'cf': 0.6, # canopy closure fraction (-)
                       #initial state of canopy storage [mm] and snow water equivalent [mm]
                       'w': 0.0, # canopy storage mm
                       'swe': 0.0, # snow water equivalent mm
                       }
            }
            
    # BUCKET
    pbu = {'depth': 0.4,  # root zone depth (m)
           # following soil properties are used if spatial_soil = False
           'poros': 0.43, # porosity (-)
           'fc': 0.33, # field capacity (-)
           'wp': 0.13,	 # wilting point (-)
           'ksat': 2.0e-6, 
           'beta': 4.7,
           #organic (moss) layer
           'org_depth': 0.04, # depth of organic top layer (m)
           'org_poros': 0.9, # porosity (-)
           'org_fc': 0.3, # field capacity (-)
           'org_rw': 0.24, # critical vol. moisture content (-) for decreasing phase in Ef
           'maxpond': 0.0, # max ponding allowed (m)
           #initial states: rootzone and toplayer soil saturation ratio [-] and pond storage [m]
           'rootzone_sat': 0.6, # root zone saturation ratio (-)
           'org_sat': 1.0, # organic top layer saturation ratio (-)
           'pond_sto': 0.0 # pond storage
           }
    
    # TOPMODEL
    ptop = {'dt': 86400.0, # timestep (s)
            'm': 0.01, # scaling depth (m)
            'ko': 0.001, # transmissivity parameter (ms-1)
            'twi_cutoff': 99.5,  # cutoff of cumulative twi distribution (%)
            'so': 0.05 # initial saturation deficit (m)
           }
    
    return pgen, pcpy, pbu, ptop

def parameters_kuru():
    # Kuru MEOLO / EFFORTE test site.
    
    pgen = {'catchment_id': 'Kuru',
            'gis_folder': str(pathlib.Path('../SpaFHyData/Kuru/')),
            'forcing_file': str(pathlib.Path('../SpaFHyData/Kuru/Kuru_FMI.csv')),
            'runoff_file': None,
            #'soil_file': r'c:\Repositories\Spathy\ini\soiltypes.ini',
            'ncf_file': str(pathlib.Path('results/Kuru.nc')),
            'start_date': '2013-01-01',
            'end_date': '2017-12-31',
            'spinup_end': '2013-12-31',
            'dt': 86400.0,
            'spatial_cpy': True,
            'spatial_soil': True     
           }
    
    # canopygrid
    pcpy = {'loc': {'lat': 61.95, 'lon': 23.85},
            'flow' : { # flow field
                     'zmeas': 2.0,
                     'zground': 0.5,
                     'zo_ground': 0.01
                     },
            'interc': { # interception
                        'wmax': 1.5, # storage capacity for rain (mm/LAI)
                        'wmaxsnow': 4.5, # storage capacity for snow (mm/LAI),
                        },
            'snow': {
                    # degree-day snow model
                    'kmelt': 2.8934e-05, # melt coefficient in open (mm/s)
                    'kfreeze': 5.79e-6, # freezing coefficient (mm/s)
                    'r': 0.05 # maximum fraction of liquid in snow (-)
                    },
                    
            'physpara': {
                        # canopy conductance
                        'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                        'g1_conif': 2.1, # stomatal parameter, conifers
                        'g1_decid': 3.5, # stomatal parameter, deciduous
                        'q50': 50.0, # light response parameter (Wm-2)
                        'kp': 0.6, # light attenuation parameter (-)
                        'rw': 0.20, # critical value for REW (-),
                        'rwmin': 0.02, # minimum relative conductance (-)
                        # soil evaporation
                        'gsoil': 1e-2 # soil surface conductance if soil is fully wet (m/s)
                        },
            'phenopara': {
                        #seasonal cycle of physiology: smax [degC], tau[d], xo[degC],fmin[-](residual photocapasity)
                        'smax': 18.5, # degC
                        'tau': 13.0, # days
                        'xo': -4.0, # degC
                        'fmin': 0.05, # minimum photosynthetic capacity in winter (-)
                        # deciduos phenology
                        'lai_decid_min': 0.1, # minimum relative LAI (-)
                        'ddo': 45.0, # degree-days for bud-burst (5degC threshold)
                        'ddur': 23.0, # duration of leaf development (days)
                        'sdl': 9.0, # daylength for senescence start (h)
                        'sdur': 30.0, # duration of leaf senescence (days),
                         },
            'state': {
                       'lai_conif': 3.5, # conifer 1-sided LAI (m2 m-2)
                       'lai_decid_max': 0.5, # maximum annual deciduous 1-sided LAI (m2 m-2): 
                       'hc': 16.0, # canopy height (m)
                       'cf': 0.6, # canopy closure fraction (-)
                       #initial state of canopy storage [mm] and snow water equivalent [mm]
                       'w': 0.0, # canopy storage mm
                       'swe': 0.0, # snow water equivalent mm
                       }
            }
            
            
        
    # BUCKET
    pbu = {'depth': 0.4,  # root zone depth (m)
           # following soil properties are used if spatial_soil = False
           'poros': 0.43, # porosity (-)
           'fc': 0.33, # field capacity (-)
           'wp': 0.13,	 # wilting point (-)
           'ksat': 2.0e-6, 
           'beta': 4.7,
           #organic (moss) layer
           'org_depth': 0.04, # depth of organic top layer (m)
           'org_poros': 0.9, # porosity (-)
           'org_fc': 0.3, # field capacity (-)
           'org_rw': 0.24, # critical vol. moisture content (-) for decreasing phase in Ef
           'maxpond': 0.0, # max ponding allowed (m)
           #initial states: rootzone and toplayer soil saturation ratio [-] and pond storage [m]
           'rootzone_sat': 0.6, # root zone saturation ratio (-)
           'org_sat': 1.0, # organic top layer saturation ratio (-)
           'pond_sto': 0.0 # pond storage
           }
    
    # TOPMODEL
    ptop = {'dt': 86400.0, # timestep (s)
            'm': 0.01, # scaling depth (m)
            'ko': 0.001, # transmissivity parameter (ms-1)
            'twi_cutoff': 99.5,  # cutoff of cumulative twi distribution (%)
            'so': 0.05 # initial saturation deficit (m)
           }
    
    return pgen, pcpy, pbu, ptop

def parameters_hyde():
    # parameter file for running SpaFHy_point at Hyytiälä
    
    pgen = {'catchment_id': 'Hyde',
            'gis_folder': None,
            'forcing_file': None,
            'runoff_file':None,
            'output_folder': str(pathlib.Path('results/Vihti.nc')),
            'start_date': '2013-01-01',
            'end_date': '2016-12-31',
            'spinup_end': '2013-12-31',
            'dt': 86400.0,
            'spatial_cpy': False,
            'spatial_soil': False     
           }
    
    pcpy = {'lat': 61.4,
            'lon': 23.7,
            'lai_conif': 3.5, # conifer 1-sided LAI (m2 m-2)
            'lai_decid': 0.5, # deciduous 1-sided LAI (m2 m-2)
            'hc': 16.0, # canopy height (m)
            'cf': 0.6, # canopy closure fraction (-)
            'zmeas': 10.0, # forcing data (U) height above canopy top (m)
            # interception
            'wmax': 1.5, # storage capacity for rain (mm/LAI)
            'wmaxsnow': 4.5, # storage capacity for snow (mm/LAI)
            # degree-day snow model
            'kmelt': 2.8934e-05, # melt coefficient in open (mm/s)
            'kfreeze': 5.79e-6, # freezing coefficient (mm/s)
            # canopy conductance
            'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
            'g1_conif': 2.1, # stomatal parameter, conifers
            'g1_decid': 3.5, # stomatal parameter, deciduous
            'q50': 50.0, # light response parameter (Wm-2)
            'kp': 0.6, # light attenuation parameter (-)
            'rw': 0.20, # critical value for REW (-),
            'rwmin': 0.02, # minimum relative conductance (-)
            # soil evaporation
            'gsoil': 1e-2, # soil surface conductance if soil is fully wet (m/s)
            #seasonal cycle of physiology: smax [degC], tau[d], xo[degC],fmin[-](residual photocapasity)
            'smax': 18.5, # degC
            'tau': 13.0, # days
            'xo': -4.0, # degC
            'fmin': 0.05, # minimum photosynthetic capacity in winter (-)
            # deciduos phenology
            'lai_min': 0.1, # minimum relative LAI (-)
            'ddo': 45.0, # degree-days for bud-burst (5degC threshold)
            'ddur': 23.0, # duration of leaf development (days)
            'sdl': 9.0, # daylength for senescence start (h)
            'sdur': 30.0, # duration of leaf senescence (days)
    
            #initial state of canopy storage [mm] and snow water equivalent [mm]
            'w': 0.0,
            'swe': 0.0
           }
    
    # BUCKET
    pbu = {'depth': 0.4,  # root zone depth (m)
           # following soil properties are used if spatial_soil = False
           'poros': 0.43, # porosity (-)
           'fc': 0.33, # field capacity (-)
           'wp': 0.13,	 # wilting point (-)
           'ksat': 2.0e-6, 
           'beta': 4.7,
           #organic (moss) layer
           'org_depth': 0.04, # depth of organic top layer (m)
           'org_poros': 0.9, # porosity (-)
           'org_fc': 0.3, # field capacity (-)
           'org_rw': 0.24, # critical vol. moisture content (-) for decreasing phase in Ef
           'maxpond': 0.0, # max ponding allowed (m)
           #initial states: rootzone and toplayer soil saturation ratio [-] and pond storage [m]
           'rootzone_sat': 0.6, # root zone saturation ratio (-)
           'org_sat': 1.0, # organic top layer saturation ratio (-)
           'pond_sto': 0.0 # pond storage
           }

    
    return pgen, pcpy, pbu

def soil_properties():
    psoil = {
             'FineTextured': 
                 {'airentry': 34.2,
                  'alpha': 0.018,
                  'beta': 7.9,
                  'fc': 0.34,
                  'ksat': 1e-06,
                  'n': 1.16,
                  'poros': 0.5,
                  'soil_id': 3.0,
                  'wp': 0.25,
                  'wr': 0.07,
                 },

             'MediumTextured': 
                 {'airentry': 20.8,
                  'alpha': 0.024,
                  'beta': 4.7,
                  'fc': 0.33,
                  'ksat': 1e-05,
                  'n': 1.2,
                  'poros': 0.43,
                  'soil_id': 2.0,
                  'wp': 0.13,
                  'wr': 0.05,
                 },

            'CoarseTextured':
                 {'airentry': 14.7,
                  'alpha': 0.039,
                  'beta': 3.1,
                  'fc': 0.21,
                  'ksat': 0.0001,
                  'n': 1.4,
                  'poros': 0.41,
                  'soil_id': 1.0,
                  'wp': 0.1,
                  'wr': 0.05,
                 },

             'Peat':
                 {'airentry': 29.2,
                  'alpha': 0.123,
                  'beta': 6.0,
                  'fc': 0.414,
                  'ksat': 5e-05,
                  'n': 1.28,
                  'poros': 0.9,
                  'soil_id': 4.0,
                  'wp': 0.11,
                  'wr': 0.0,
                 },
              'Humus':
                 {'airentry': 29.2,
                  'alpha': 0.123,
                  'beta': 6.0,
                  'fc': 0.35,
                  'ksat': 8e-06,
                  'n': 1.28,
                  'poros': 0.85,
                  'soil_id': 5.0,
                  'wp': 0.15,
                  'wr': 0.01,
                 },
            }
    return psoil