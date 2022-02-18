# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:21:14 2018

@author: slauniai
"""

def parameters_vihti():
    # Vihti MEOLO / EFFORTE test site.
    
    pgen = {'catchment_id': 'Vihti',
            'gis_folder': r'c:\Projects\TRAM\data\SpaFHyData\Vihti',
            'forcing_file': r'c:\Projects\TRAM\data\SpaFHyData\Vihti\Vihti_FMI_10x10.csv',
            'runoff_file': None,
            #'soil_file': r'c:\Repositories\Spathy\ini\soiltypes.ini',
            'ncf_file': r'c:\tempdata\spafhy\Vihti.nc',
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