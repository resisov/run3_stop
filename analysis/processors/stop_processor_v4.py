#!/usr/bin/env python
import logging
import numpy as np
import awkward as ak
import json
import copy
from collections import defaultdict
from coffea import processor
import cachetools
import hist
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from coffea.util import load, save
from optparse import OptionParser
from coffea.nanoevents.methods import vector
import gzip

class AnalysisProcessor(processor.ProcessorABC):
    lumis = {
        '2022pre': 7.9804,
        '2022post': 26.6717,
        '2023pre': 17.794,
        '2023post': 9.451,
        '2024':  109.82,
        '2025': 110.58
    }
    lumiMasks = {
        '2022pre': LumiMask('data/lumiMask/Cert_Collisions2022_355100_362760_Golden.json'),
        '2022post': LumiMask('data/lumiMask/Cert_Collisions2022_355100_362760_Golden.json'),
        '2023pre': LumiMask('data/lumiMask/Cert_Collisions2023_366442_370790_Golden.json'),
        '2023post': LumiMask('data/lumiMask/Cert_Collisions2023_366442_370790_Golden.json'),
        '2024': LumiMask('data/lumiMask/Cert_Collisions2024_378981_386951_Golden.json'),
        '2025': LumiMask('data/lumiMask/Cert_Collisions2025_391658_398903_Golden.json'),
    }
    metFilters = {
        '2022pre': [
                'goodVertices', 
                'globalSuperTightHalo2016Filter', 
                'HBHENoiseFilter', 
                'HBHENoiseIsoFilter', 
                'EcalDeadCellTriggerPrimitiveFilter', 
                'BadPFMuonFilter', 
                'BadPFMuonDzFilter', 
                'eeBadScFilter', 
                'ecalBadCalibFilter'
        ],
        '2022post': [
                'goodVertices', 
                'globalSuperTightHalo2016Filter', 
                'HBHENoiseFilter', 
                'HBHENoiseIsoFilter', 
                'EcalDeadCellTriggerPrimitiveFilter', 
                'BadPFMuonFilter', 
                'BadPFMuonDzFilter', 
                'eeBadScFilter', 
                'ecalBadCalibFilter'
        ],
        '2023pre': [
                'goodVertices', 
                'globalSuperTightHalo2016Filter', 
                'HBHENoiseFilter', 
                'HBHENoiseIsoFilter', 
                'EcalDeadCellTriggerPrimitiveFilter', 
                'BadPFMuonFilter', 
                'BadPFMuonDzFilter', 
                'eeBadScFilter', 
                'ecalBadCalibFilter'
        ],
        '2023post': [
                'goodVertices', 
                'globalSuperTightHalo2016Filter', 
                'HBHENoiseFilter', 
                'HBHENoiseIsoFilter', 
                'EcalDeadCellTriggerPrimitiveFilter', 
                'BadPFMuonFilter', 
                'BadPFMuonDzFilter', 
                'eeBadScFilter', 
                'ecalBadCalibFilter'
        ],
        '2024': [
                'goodVertices', 
                'globalSuperTightHalo2016Filter', 
                'HBHENoiseFilter', 
                'HBHENoiseIsoFilter', 
                'EcalDeadCellTriggerPrimitiveFilter', 
                'BadPFMuonFilter', 
                'BadPFMuonDzFilter', 
                'eeBadScFilter', 
                'ecalBadCalibFilter'
        ],
        '2025': [
                'goodVertices', 
                'globalSuperTightHalo2016Filter', 
                'HBHENoiseFilter', 
                'HBHENoiseIsoFilter', 
                'EcalDeadCellTriggerPrimitiveFilter', 
                'BadPFMuonFilter', 
                'BadPFMuonDzFilter', 
                'eeBadScFilter', 
                'ecalBadCalibFilter'
        ]
    }

    def __init__(self, year, xsec, corrections, ids, common, shift_name=None):
        self._year = year
        self._lumi = 1000 * float(self.lumis[year])
        self._xsec = xsec

        self._corrections = corrections
        self._ids = ids
        self._common = common
        self._shift_name = shift_name
        
        self._samples = {
            'cat1_preselection': ('TT','QCD','Zto2Nu','WtoLNu','ST','GJ','DY','JetMET','VV', 'SMS'),
            'cat2_LLCR_highDeltaM': ('TT','QCD','Zto2Nu','WtoLNu','ST','GJ','DY','JetMET','VV', 'SMS'),
            'cat3_QCDCR_highDeltaM': ('TT','QCD','Zto2Nu','WtoLNu','ST','GJ','DY','JetMET','VV', 'SMS'),
            'cat4_GCR_highDeltaM': ('TT','QCD','Zto2Nu','WtoLNu','ST','GJ','DY','EGamma','VV', 'SMS'),
            'cat5_DY2E_highDeltaM': ('TT','QCD','Zto2Nu','WtoLNu','ST','GJ','DY','EGamma','VV', 'SMS'),
            'cat6_DY2M_highDeltaM': ('TT','QCD','Zto2Nu','WtoLNu','ST','GJ','DY','Muon','VV', 'SMS'),
            'cat7_SR_highDeltaM': ('TT','QCD','Zto2Nu','WtoLNu','ST','GJ','DY','JetMET','VV', 'SMS'),
        }
        self._signal_triggers ={
            '2022pre': [
                'PFMET120_PFMHT120_IDTight',
                'PFMET130_PFMHT130_IDTight',
                'PFMET140_PFMHT140_IDTight',
                'PFMETNoMu120_PFMHTNoMu120_IDTight',
                'PFMETNoMu130_PFMHTNoMu130_IDTight',
                'PFMETNoMu140_PFMHTNoMu140_IDTight'
            ],
            '2022post': [
                'PFMET120_PFMHT120_IDTight',
                'PFMET130_PFMHT130_IDTight',
                'PFMET140_PFMHT140_IDTight',
                'PFMETNoMu120_PFMHTNoMu120_IDTight',
                'PFMETNoMu130_PFMHTNoMu130_IDTight',
                'PFMETNoMu140_PFMHTNoMu140_IDTight'
            ],
            '2023pre': [
                'PFMET120_PFMHT120_IDTight',
                'PFMET130_PFMHT130_IDTight',
                'PFMET140_PFMHT140_IDTight',
                'PFMETNoMu120_PFMHTNoMu120_IDTight',
                'PFMETNoMu130_PFMHTNoMu130_IDTight',
                'PFMETNoMu140_PFMHTNoMu140_IDTight'
            ],
            '2023post': [
                'PFMET120_PFMHT120_IDTight',
                'PFMET130_PFMHT130_IDTight',
                'PFMET140_PFMHT140_IDTight',
                'PFMETNoMu120_PFMHTNoMu120_IDTight',
                'PFMETNoMu130_PFMHTNoMu130_IDTight',
                'PFMETNoMu140_PFMHTNoMu140_IDTight'
            ],
            '2024': [
                'PFMET120_PFMHT120_IDTight',
                'PFMET130_PFMHT130_IDTight',
                'PFMET140_PFMHT140_IDTight',
                'PFMETNoMu120_PFMHTNoMu120_IDTight',
                'PFMETNoMu130_PFMHTNoMu130_IDTight',
                'PFMETNoMu140_PFMHTNoMu140_IDTight'
            ],
            '2025': [
                'PFMET120_PFMHT120_IDTight',
                'PFMET130_PFMHT130_IDTight',
                'PFMET140_PFMHT140_IDTight',
                'PFMETNoMu120_PFMHTNoMu120_IDTight',
                'PFMETNoMu130_PFMHTNoMu130_IDTight',
                'PFMETNoMu140_PFMHTNoMu140_IDTight'
            ]
        }
        self._reference_triggers = {
            '2022pre': [
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf'
            ],
            '2022post': [
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf'
            ],
            '2023pre': [
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf'
            ],
            '2023post': [
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf'
            ],
            '2024': [
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf'
            ],
            '2025': [
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf'
            ]
        }
        self._electron_triggers = {
            '2022pre': [
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Ele135_CaloIdVT_GsfTrkIdT',
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                'DoubleEle25_CaloIdL_MW',
                'DoubleEle27_CaloIdL_MW',
                'DoubleEle33_CaloIdL_MW'
            ],
            '2022post': [
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Ele135_CaloIdVT_GsfTrkIdT',
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                'DoubleEle25_CaloIdL_MW',
                'DoubleEle27_CaloIdL_MW',
                'DoubleEle33_CaloIdL_MW'
            ],
            '2023pre': [
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Ele135_CaloIdVT_GsfTrkIdT',
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                'DoubleEle25_CaloIdL_MW',
                'DoubleEle27_CaloIdL_MW',
                'DoubleEle33_CaloIdL_MW'
            ],
            '2023post': [
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Ele135_CaloIdVT_GsfTrkIdT',
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                'DoubleEle25_CaloIdL_MW',
                'DoubleEle27_CaloIdL_MW',
                'DoubleEle33_CaloIdL_MW'
            ],
            '2024': [
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Ele135_CaloIdVT_GsfTrkIdT',
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                'DoubleEle25_CaloIdL_MW',
                'DoubleEle27_CaloIdL_MW',
                'DoubleEle33_CaloIdL_MW'
            ],
            '2025': [
                'Ele115_CaloIdVT_GsfTrkIdT',
                'Ele135_CaloIdVT_GsfTrkIdT',
                'Ele30_WPTight_Gsf',
                'Ele32_WPTight_Gsf',
                'Ele35_WPTight_Gsf',
                'Ele38_WPTight_Gsf',
                'Ele40_WPTight_Gsf',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                'DoubleEle25_CaloIdL_MW',
                'DoubleEle27_CaloIdL_MW',
                'DoubleEle33_CaloIdL_MW'
            ]
        }
        self._muon_triggers = {
            '2022pre': [
                'IsoMu20',
                'IsoMu24',
                'IsoMu27',
                'IsoMu24_eta2p1',
                'Mu50',
                'Mu55'
            ],
            '2022post': [
                'IsoMu20',
                'IsoMu24',
                'IsoMu27',
                'IsoMu24_eta2p1',
                'Mu50',
                'Mu55'
            ],
            '2023pre': [
                'IsoMu20',
                'IsoMu24',
                'IsoMu27',
                'IsoMu24_eta2p1',
                'Mu50',
                'Mu55'
            ],
            '2023post': [
                'IsoMu20',
                'IsoMu24',
                'IsoMu27',
                'IsoMu24_eta2p1',
                'Mu50',
                'Mu55'
            ],
            '2024': [
                'IsoMu20',
                'IsoMu24',
                'IsoMu27',
                'IsoMu24_eta2p1',
                'Mu50',
                'Mu55'
            ],
            '2025': [
                'IsoMu20',
                'IsoMu24',
                'IsoMu27',
                'IsoMu24_eta2p1',
                'Mu50',
                'Mu55'
            ]
        }
        self._photon_triggers = {
            '2022pre': [
                'Photon175',
                'Photon200'
            ],
            '2022post': [
                'Photon175',
                'Photon200'
            ],
            '2023pre': [
                'Photon175',
                'Photon200'
            ],
            '2023post': [
                'Photon175',
                'Photon200'
            ],
            '2024': [
                'Photon175',
                'Photon200'
            ],
            '2025': [
                'Photon175',
                'Photon200'
            ]
        }
        self._ht_triggers = {
            '2022pre': [
                'PFHT180',
                'PFHT250',
                'PFHT350',
                'PFHT430',
                'PFHT510',
                'PFHT590',
                'PFHT680',
                'PFHT780',
                'PFHT890',
                'PFHT1050',
            ],
            '2022post': [
                'PFHT180',
                'PFHT250',
                'PFHT350',
                'PFHT430',
                'PFHT510',
                'PFHT590',
                'PFHT680',
                'PFHT780',
                'PFHT890',
                'PFHT1050',
            ],
            '2023pre': [
                'PFHT180',
                'PFHT250',
                'PFHT350',
                'PFHT430',
                'PFHT510',
                'PFHT590',
                'PFHT680',
                'PFHT780',
                'PFHT890',
                'PFHT1050',
            ],
            '2023post': [
                'PFHT180',
                'PFHT250',
                'PFHT350',
                'PFHT430',
                'PFHT510',
                'PFHT590',
                'PFHT680',
                'PFHT780',
                'PFHT890',
                'PFHT1050',
            ],
            '2024': [
                'PFHT180',
                'PFHT250',
                'PFHT350',
                'PFHT430',
                'PFHT510',
                'PFHT590',
                'PFHT680',
                'PFHT780',
                'PFHT890',
                'PFHT1050',
            ],
            '2025': [
                'PFHT180',
                'PFHT250',
                'PFHT350',
                'PFHT430',
                'PFHT510',
                'PFHT590',
                'PFHT680',
                'PFHT780',
                'PFHT890',
                'PFHT1050',
            ]
        }

        self.make_output = lambda: {
            'sumw': 0.0,
            'template': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable([250,260,270,280,290,300,350,400,500,800], name='met', label=r'$E_{T}^{miss}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'metpt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.IntCategory([0, 1], name='signal_trigger', label='Signal Trigger'),
                hist.axis.Variable([250,300,350,400,500,800,1500], name='met', label=r'$E_{T}^{miss}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'metpt_10GeVbins': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(150, 0, 1500, name='metpt_10GeVbins', label=r'$E_{T}^{miss}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'metphi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -np.pi, np.pi, name='metphi', label=r'$\phi_{E_{T}^{miss}}$'),
                storage=hist.storage.Weight(),
            ),
            'recoilpt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable([250,300,350,400,500,800,1500], name='recoilpt', label=r'$E_{T}^{miss}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'recoilphi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -np.pi, np.pi, name='recoilphi', label=r'$\phi_{E_{T}^{miss}}$'),
                storage=hist.storage.Weight(),
            ),
            'nElectron': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(5, 0, 5, name='nElectron', label='Number of Electrons'),
                storage=hist.storage.Weight(),
            ),
            'nMuon': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(5, 0, 5, name='nMuon', label='Number of Muons'),
                storage=hist.storage.Weight(),
            ),
            'nJet': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(20, 0, 20, name='nJet', label='Number of Jets'),
                storage=hist.storage.Weight(),
            ),
            'j1pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable(np.arange(0,1001,10), name='j1pt', label=r'Leading Jet $p_{T}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'j1eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -5.0, 5.0, name='j1eta', label=r'Leading Jet $\eta$'),
                storage=hist.storage.Weight(),
            ),
            'j1phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -np.pi, np.pi, name='j1phi', label=r'Leading Jet $\phi$'),
                storage=hist.storage.Weight(),
            ),
            'j2pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable(np.arange(0,1001,10), name='j2pt', label=r'Sub-leading Jet $p_{T}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'j2eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -5.0, 5.0, name='j2eta', label=r'Sub-leading Jet $\eta$'),
                storage=hist.storage.Weight(),
            ),
            'j2phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -np.pi, np.pi, name='j2phi', label=r'Sub-leading Jet $\phi$'),
                storage=hist.storage.Weight(),
            ),
            'nb_loose': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(5, 0, 5, name='nb_loose', label='Number of b-tagged Jets'),
                storage=hist.storage.Weight(),
            ),
            'nb_medium': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(5, 0, 5, name='nb_medium', label='Number of b-tagged Jets'),
                storage=hist.storage.Weight(),
            ),
            'nb_tight': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(5, 0, 5, name='nb_tight', label='Number of b-tagged Jets'),
                storage=hist.storage.Weight(),
            ),
            'b1_medium_pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable(np.arange(0,1001,10), name='b1_medium_pt', label=r'Leading b-tagged Jet $p_{T}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'b1_medium_eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -5.0, 5.0, name='b1_medium_eta', label=r'Leading b-tagged Jet $\eta$'),
                storage=hist.storage.Weight(),
            ),
            'b1_medium_phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -np.pi, np.pi, name='b1_medium_phi', label=r'Leading b-tagged Jet $\phi$'),
                storage=hist.storage.Weight(),
            ),
            'b1_tight_pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable(np.arange(0,601,10), name='b1_tight_pt', label=r'Leading Tight b-tagged Jet $p_{T}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'b1_tight_eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -5.0, 5.0, name='b1_tight_eta', label=r'Leading Tight b-tagged Jet $\eta$'),
                storage=hist.storage.Weight(),
            ),
            'b1_tight_phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -np.pi, np.pi, name='b1_tight_phi', label=r'Leading Tight b-tagged Jet $\phi$'),
                storage=hist.storage.Weight(),
            ),
            'b1_loose_pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable(np.arange(0,601,10), name='b1_loose_pt', label=r'Leading Loose b-tagged Jet $p_{T}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'b1_loose_eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -5.0, 5.0, name='b1_loose_eta', label=r'Leading Loose b-tagged Jet $\eta$'),
                storage=hist.storage.Weight(),
            ),
            'b1_loose_phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -np.pi, np.pi, name='b1_loose_phi', label=r'Leading Loose b-tagged Jet $\phi$'),
                storage=hist.storage.Weight(),
            ),
            'ht': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable([300,350,400,500,600,700,800,900,1000,1200,1500,2000,3000], name='ht', label=r'$H_{T}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'mll': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(100,50,150, name='mll', label='Dilepton Invariant Mass (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'pll': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(10,0,1000, name='pll', label='Dilepton System $p_{T}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'nPV': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(100, 0, 100, name='nPV', label='Number of Primary Vertices'),
                storage=hist.storage.Weight(),
            ),
            'nfj': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(10, 0, 10, name='nfj', label='Number of Fat Jets'),
                storage=hist.storage.Weight(),
            ),
            'fj1pt': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable(np.arange(0,1001,10), name='fj1pt', label=r'Leading Fat Jet $p_{T}$ (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'fj1mass': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable(np.arange(0,501,10), name='fj1mass', label='Leading Fat Jet Mass (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'fj1msd': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Variable(np.arange(0,501,10), name='fj1msd', label='Leading Fat Jet Soft Drop Mass (GeV)'),
                storage=hist.storage.Weight(),
            ),
            'fj1phi': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -np.pi, np.pi, name='fj1phi', label=r'Leading Fat Jet $\phi$'),
                storage=hist.storage.Weight(),
            ),
            'fj1eta': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(64, -5.0, 5.0, name='fj1eta', label=r'Leading Fat Jet $\eta$'),
                storage=hist.storage.Weight(),
            ),
            'fj1TvsQCD': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(500, 0.0, 1.0, name='fj1TvsQCD', label='Leading Fat Jet TvsQCD'),
                storage=hist.storage.Weight(),
            ),
            'fj1WvsQCD': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(50, 0.0, 1.0, name='fj1WvsQCD', label='Leading Fat Jet WvsQCD'),
                storage=hist.storage.Weight(),
            ),
            'fj1QCD': hist.Hist(
                hist.axis.StrCategory([], name='region', growth=True),
                hist.axis.StrCategory([], name='systematic', growth=True),
                hist.axis.Regular(50, 0.0, 1.0, name='fj1QCD', label='Leading Fat Jet QCD'),
                storage=hist.storage.Weight(),
            ),
        }
    
    def process(self, events):
        isData = not hasattr(events, 'genWeight')
        selection = PackedSelection(dtype="uint64")
        weights = Weights(len(events), storeIndividual=True)
        output = self.make_output()
        if not isData:
            output['sumw'] = ak.sum(events.genWeight)
        
        dataset = events.metadata['dataset']
        group = dataset
        try:
            caller = dataset.split("____")[0]
        except:
            caller = dataset
        ## Grouping Single top samples as ST
        if 'TWminus' in dataset or 'TbarWplus' in dataset or 'TBbar' in dataset or 'TbarB' in dataset:
            group = 'ST-' + dataset
        ## Grouping WW, WZ, ZZ samples as VV
        if 'WW' in dataset or 'WZ' in dataset or 'ZZ' in dataset:
            group = 'VV-' + dataset
        selected_regions = []
        for region, samples in self._samples.items():
            for sample in samples:
                if sample in group:
                    selected_regions.append(region)
                    break
                    
        """ Initialize the physics objects """
        ### read the ids
        isTrackElectron = self._ids['isTrackElectron']
        isTrackMuon = self._ids['isTrackMuon']
        isTrackPion = self._ids['isTrackPion']
        isVetoElectron = self._ids['isVetoElectron']
        isMediumElectron = self._ids['isMediumElectron']
        isLooseMuon = self._ids['isLooseMuon']
        isMediumMuon = self._ids['isMediumMuon']
        isMediumPhoton = self._ids['isMediumPhoton']
        isMediumTau = self._ids['isMediumTau']
        isGoodJet = self._ids['isGoodJet']
        isGoodFatJet = self._ids['isGoodFatJet']
        isJetVeto = self._ids['isJetVeto']

        ### read the corrections
        get_pu_weight = self._corrections['get_pu_weight']
        get_jec_correction = self._corrections['get_jec_correction']
        get_jec_uncertainty = self._corrections['get_jec_uncertainty']
        get_fjec_correction = self._corrections['get_fjec_correction']
        get_ele_veto_id_sf = self._corrections['get_ele_veto_id_sf']
        get_ele_medium_id_sf = self._corrections['get_ele_medium_id_sf']
        get_ele_hlt_sf = self._corrections['get_ele_hlt_sf']
        get_mu_loose_id_sf = self._corrections['get_mu_loose_id_sf']
        get_mu_medium_id_sf = self._corrections['get_mu_medium_id_sf']
        get_mu_hlt_sf = self._corrections['get_mu_hlt_sf']
        get_photon_id_sf = self._corrections['get_photon_id_sf']
        get_top_pt_reweight = self._corrections['get_top_pt_reweight']
        get_btag_weight = self._corrections['get_btag_weight']

        ### Initialize global quantities
        npv = events.PV.npvsGood
        run = events.run
        if '2022' in self._year or '2023' in self._year:
            met = events.MET
        else:
            met = events.PuppiMET
        if (not isData) and self._shift_name in ["metUnclusteredUp", "metUnclusteredDown"]:
            direction = "Up" if self._shift_name.endswith("Up") else "Down"
            met_fields = set(getattr(met, "fields", []))
            pt_field = f"ptUnclustered{direction}"
            phi_field = f"phiUnclustered{direction}"

            if pt_field not in met_fields or phi_field not in met_fields:
                raise ValueError(
                    f"Could not find MET pt and phi variables for unclustered energy shift. "
                    f"Need {pt_field} and {phi_field}. Available fields: {met_fields}"
                )

            met = ak.with_field(met, met[pt_field], "pt")
            met = ak.with_field(met, met[phi_field], "phi")
        calo_met = events.CaloMET

        ### Tracks
        trk = events.IsoTrack
        trk['isTrackElectron'] = isTrackElectron(trk, met.pt, met.phi, self._year)
        trk['isTrackMuon'] = isTrackMuon(trk, met.pt, met.phi, self._year)
        trk['isTrackPion'] = isTrackPion(trk, met.pt, met.phi, self._year)
        trk_e = trk[trk.isTrackElectron]
        trk_m = trk[trk.isTrackMuon]
        trk_pi = trk[trk.isTrackPion]

        ### Electrons
        e = events.Electron
        e_sf_veto, e_sf_veto_up, e_sf_veto_down = get_ele_veto_id_sf(self._year, e.eta+e.deltaEtaSC, e.pt, e.phi)
        e_sf_medium, e_sf_medium_up, e_sf_medium_down = get_ele_medium_id_sf(self._year, e.eta+e.deltaEtaSC, e.pt, e.phi)
        e_sf_hlt, e_sf_hlt_up, e_sf_hlt_down = get_ele_hlt_sf(self._year, e.eta, e.pt, e.phi)
        e['isveto'] = isVetoElectron(e, self._year)
        e['veto_id_sf'] = ak.where(e.isveto, e_sf_veto, ak.ones_like(e.pt))
        e['veto_id_sf_up'] = ak.where(e.isveto, e_sf_veto_up, ak.ones_like(e.pt))
        e['veto_id_sf_down'] = ak.where(e.isveto, e_sf_veto_down, ak.ones_like(e.pt))
        e['ismedium'] = isMediumElectron(e, self._year)
        e['medium_id_sf'] = ak.where(e.ismedium, e_sf_medium, ak.ones_like(e.pt))
        e['medium_id_sf_up'] = ak.where(e.ismedium, e_sf_medium_up, ak.ones_like(e.pt))
        e['medium_id_sf_down'] = ak.where(e.ismedium, e_sf_medium_down, ak.ones_like(e.pt))
        e['hlt_sf'] = ak.where(e.ismedium, e_sf_hlt, ak.ones_like(e.pt))
        e['hlt_sf_up'] = ak.where(e.ismedium, e_sf_hlt_up, ak.ones_like(e.pt))
        e['hlt_sf_down'] = ak.where(e.ismedium, e_sf_hlt_down, ak.ones_like(e.pt))
        e['T'] = ak.zip({
            'r': e.pt,
            'phi': e.phi,
        }, with_name='PolarTwoVector', behavior=vector.behavior)
        e_veto = e[e.isveto]
        e_medium = e[e.ismedium]
        #print(e_medium.pt)
        mT_e = np.sqrt(2 * e_veto.pt * met.pt * (1 - np.cos(met.delta_phi(e_veto.T))))
        leading_e = ak.firsts(e_medium)
        second_e = ak.pad_none(e_medium, target=2)[:,1]
        mee = (leading_e + second_e).mass
        pee = (leading_e + second_e).pt
        
        ### Muons
        m = events.Muon
        m_sf_loose, m_sf_loose_up, m_sf_loose_down = get_mu_loose_id_sf(self._year, m.eta, m.pt)
        m_sf_medium, m_sf_medium_up, m_sf_medium_down = get_mu_medium_id_sf(self._year, m.eta, m.pt)
        m_sf_hlt, m_sf_hlt_up, m_sf_hlt_down = get_mu_hlt_sf(self._year, m.eta, m.pt)
        m['isloose'] = isLooseMuon(m, self._year)
        m['loose_id_sf'] = ak.where(m.isloose, m_sf_loose, ak.ones_like(m.pt))
        m['loose_id_sf_up'] = ak.where(m.isloose, m_sf_loose_up, ak.ones_like(m.pt))
        m['loose_id_sf_down'] = ak.where(m.isloose, m_sf_loose_down, ak.ones_like(m.pt))
        m['ismedium'] = isMediumMuon(m, self._year)
        m['medium_id_sf'] = ak.where(m.ismedium, m_sf_medium, ak.ones_like(m.pt))
        m['medium_id_sf_up'] = ak.where(m.ismedium, m_sf_medium_up, ak.ones_like(m.pt))
        m['medium_id_sf_down'] = ak.where(m.ismedium, m_sf_medium_down, ak.ones_like(m.pt))
        m['hlt_sf'] = ak.where(m.ismedium, m_sf_hlt, ak.ones_like(m.pt))
        m['hlt_sf_up'] = ak.where(m.ismedium, m_sf_hlt_up, ak.ones_like(m.pt))
        m['hlt_sf_down'] = ak.where(m.ismedium, m_sf_hlt_down, ak.ones_like(m.pt))
        
        m['T'] = ak.zip({
            'r': m.pt,
            'phi': m.phi,
        }, with_name='PolarTwoVector', behavior=vector.behavior)
        m_loose = m[m.isloose]
        m_medium = m[m.ismedium]
        mT_m = np.sqrt(2 * m_loose.pt * met.pt * (1 - np.cos(met.delta_phi(m_loose.T))))
        leading_m = ak.firsts(m_medium)
        second_m = ak.pad_none(m_medium, target=2)[:,1]
        mmm = (leading_m + second_m).mass
        pmm = (leading_m + second_m).pt

        ### Photons
        p = events.Photon
        p['ismedium'] = isMediumPhoton(p, self._year)
        p_sf_id, p_sf_id_up, p_sf_id_down = get_photon_id_sf(self._year, "Medium", p.eta, p.pt, p.phi)
        p['id_sf'] = ak.where(p.ismedium, p_sf_id, ak.ones_like(p.pt))
        p['id_sf_up'] = ak.where(p.ismedium, p_sf_id_up, ak.ones_like(p.pt))
        p['id_sf_down'] = ak.where(p.ismedium, p_sf_id_down, ak.ones_like(p.pt))

        p['T'] = ak.zip({
            'r': p.pt,
            'phi': p.phi,
        }, with_name='PolarTwoVector', behavior=vector.behavior)
        p_medium = p[p.ismedium]
        leading_p = ak.firsts(p_medium)

        ### Taus
        t = events.Tau
        t['ismedium'] = isMediumTau(t, met.pt, met.phi, self._year)
        t['T'] = ak.zip({
            'r': t.pt,
            'phi': t.phi,
        }, with_name='PolarTwoVector', behavior=vector.behavior)
        t_medium = t[t.ismedium]
        #print(t_medium.pt)

        ### Jets
        j = events.Jet # Events/Jet_*
        ### Appling JECs
        rho_density = events.Rho.fixedGridRhoFastjetAll
        j_raw_pt = j.pt * (1.0 - j.rawFactor)
        j_raw_mass = j.mass * (1.0 - j.rawFactor)
        jec_corr = get_jec_correction(self._year, j_raw_pt, j.eta, j.phi, rho_density, j.area, run, isData)
        j['pt'] = j_raw_pt * jec_corr
        j['mass'] = j_raw_mass * jec_corr

        ## JEC Shift variation for systematics
        if (not isData) and self._shift_name in ["jesTotalUp", "jesTotalDown"]:
            jec_unc = get_jec_uncertainty(self._year, j.pt, j.eta)
            if self._shift_name == "jesTotalUp":
                j['pt'] = j.pt * (1 + jec_unc)
                j['mass'] = j.mass * (1 + jec_unc)
            elif self._shift_name == "jesTotalDown":
                j['pt'] = j.pt * (1 - jec_unc)
                j['mass'] = j.mass * (1 - jec_unc)

        ### Appling JetID
        j['isgood'] = isGoodJet(j, self._year)
        j['T'] = ak.zip({
            'r': j.pt,
            'phi': j.phi,
        }, with_name='PolarTwoVector', behavior=vector.behavior)
        ## b-tagging
        j['isupartL'] = (j.btagUParTAK4B>0.0246)
        j['isupartM'] = (j.btagUParTAK4B>0.1272)
        j['isupartT'] = (j.btagUParTAK4B>0.4648)

        ## Jet Veto Map
        j['isveto'] = isJetVeto(j, self._year)

        ## Jet Cleaning
        j['isclean'] = (
            ak.all(j.metric_table(p_medium) > 0.2, axis=2)
        )
        j['isclean_l'] = (
            ak.all(j.metric_table(e_medium) > 0.2, axis=2) &
            ak.all(j.metric_table(m_medium) > 0.2, axis=2)
        )

        j_good = j[j.isgood]
        j_veto = j[j.isveto]
        j1 = ak.firsts(j_good)
        j2 = ak.pad_none(j_good, target=2)[:,1]
        b = j_good[j_good.isupartM]
        b1 = ak.firsts(b)
        b_tight = j_good[j_good.isupartT]
        b1_tight = ak.firsts(b_tight)
        b_loose = j_good[j_good.isupartL]
        b1_loose = ak.firsts(b_loose)
        nb_loose = ak.num(b_loose, axis=1)
        nb_medium = ak.num(b, axis=1)
        nb_tight = ak.num(b_tight, axis=1)
        b_clean_p = b[b.isclean]
        b_clean_l = b[b.isclean_l]

        ### FatJets
        fj = events.FatJet
        ### Appling JECs
        fj_raw_pt = fj.pt * (1.0 - fj.rawFactor)
        fj_raw_mass = fj.mass * (1.0 - fj.rawFactor)
        fjec_corr = get_fjec_correction(self._year, fj.pt, fj.eta, fj.phi, rho_density, fj.area, run, isData)

        fj['pt'] = fj_raw_pt * fjec_corr
        fj['mass'] = fj_raw_mass * fjec_corr
        ### Appling FatJetID
        fj['isgood'] = isGoodFatJet(fj, self._year)
        fj['T'] = ak.zip({
            'r': fj.pt,
            'phi': fj.phi,
        }, with_name='PolarTwoVector', behavior=vector.behavior)

        ## FatJet Cleaning
        fj['isclean_p'] = (
            ak.all(fj.metric_table(p_medium) > 0.4, axis=2)
        )
        fj['isclean_l'] = (
            ak.all(fj.metric_table(e_medium) > 0.4, axis=2) &
            ak.all(fj.metric_table(m_medium) > 0.4, axis=2)
        )

        fj_good = fj[fj.isgood]
        nfj_good = ak.num(fj_good, axis=1)

        j_clean_p = j_good[j_good.isclean]
        fj_clean_p = fj_good[fj_good.isclean_p]
        j_clean_l = j_good[j_good.isclean_l]
        fj_clean_l = fj_good[fj_good.isclean_l]

        ### Scalar HT
        scalarHT = {
            'cat1_preselection': ak.sum(j_good.pt, axis=1),
            'cat2_LLCR_highDeltaM': ak.sum(j_good.pt, axis=1),
            'cat3_QCDCR_highDeltaM': ak.sum(j_good.pt, axis=1),
            'cat4_GCR_highDeltaM': ak.sum(j_clean_p.pt, axis=1),
            'cat5_DY2E_highDeltaM': ak.sum(j_clean_l.pt, axis=1),
            'cat6_DY2M_highDeltaM': ak.sum(j_clean_l.pt, axis=1),
            'cat7_SR_highDeltaM': ak.sum(j_good.pt, axis=1),
        }

        ### Hadronic Recoil
        uT = {
            'cat1_preselection': met,
            'cat2_LLCR_highDeltaM': met,
            'cat3_QCDCR_highDeltaM': met,
            'cat4_GCR_highDeltaM': met+leading_p.T,
            'cat5_DY2E_highDeltaM': met+leading_e.T+second_e.T,
            'cat6_DY2M_highDeltaM': met+leading_m.T+second_m.T,
            'cat7_SR_highDeltaM': met,
        }

        mll = {
            'cat5_DY2E_highDeltaM': mee,
            'cat6_DY2M_highDeltaM': mmm,
        }

        pll = {
            'cat5_DY2E_highDeltaM': pee,
            'cat6_DY2M_highDeltaM': pmm,
        }

        """ Variables for selection """
        ### lumimask
        lumimask = np.ones(len(events), dtype=bool)
        if isData:
            lumimask = AnalysisProcessor.lumiMasks[self._year](events.run, events.luminosityBlock)
        selection.add('lumimask', lumimask)

        ### met filters
        met_filters = np.ones(len(events), dtype=bool)
        for flag in AnalysisProcessor.metFilters[self._year]:
            met_filters = met_filters & events.Flag[flag]
        selection.add('met_filters', met_filters)

        ### triggers
        signal_triggers = np.zeros(len(events), dtype=bool)
        for path in self._signal_triggers[self._year]:
            if not hasattr(events.HLT, path):
                continue
            signal_triggers = signal_triggers | events.HLT[path]
        selection.add('signal_trigger', signal_triggers)

        photon_triggers = np.zeros(len(events), dtype=bool)
        for path in self._photon_triggers[self._year]:
            if not hasattr(events.HLT, path):
                continue
            photon_triggers = photon_triggers | events.HLT[path]
        selection.add('photon_trigger', photon_triggers)

        reference_triggers = np.zeros(len(events), dtype=bool)
        for path in self._reference_triggers[self._year]:
            if not hasattr(events.HLT, path):
                continue
            reference_triggers = reference_triggers | events.HLT[path]
        selection.add('reference_trigger', reference_triggers)

        electron_triggers = np.zeros(len(events), dtype=bool)
        for path in self._electron_triggers[self._year]:
            if not hasattr(events.HLT, path):
                continue
            electron_triggers = electron_triggers | events.HLT[path]
        selection.add('electron_trigger', electron_triggers)

        muon_triggers = np.zeros(len(events), dtype=bool)
        for path in self._muon_triggers[self._year]:
            if not hasattr(events.HLT, path):
                continue
            muon_triggers = muon_triggers | events.HLT[path]
        selection.add('muon_trigger', muon_triggers)

        ### Number of objects
        n_trk_e = ak.num(trk_e, axis=1)
        n_trk_m = ak.num(trk_m, axis=1)
        n_trk_pi = ak.num(trk_pi, axis=1)
        n_e_veto = ak.num(e_veto, axis=1)
        n_e_medium = ak.num(e_medium, axis=1)
        n_m_loose = ak.num(m_loose, axis=1)
        n_m_medium = ak.num(m_medium, axis=1)
        n_p_medium = ak.num(p_medium, axis=1)
        n_t_medium = ak.num(t_medium, axis=1)
        n_j_good = ak.num(j_good, axis=1)
        n_j_veto = ak.num(j_veto, axis=1)
        n_j_clean_p = ak.num(j_clean_p, axis=1)
        n_j_clean_l = ak.num(j_clean_l, axis=1)
        n_b = ak.num(b, axis=1)

        ### Opening angle between jets and MET
        j1_met_dphi = np.abs(j1.delta_phi(met))
        j2_met_dphi = np.abs(j2.delta_phi(met))

        ### Opening angle for third, fourth and fifth jets if present
        j3 = ak.pad_none(j_good, target=3)[:,2]
        j4 = ak.pad_none(j_good, target=4)[:,3]

        j3_met_dphi = np.abs(j3.delta_phi(met))
        j4_met_dphi = np.abs(j4.delta_phi(met))

        ### opening angle for cleaned jets
        j1_clean_p = ak.firsts(j_clean_p)
        j2_clean_p = ak.pad_none(j_clean_p, target=2)[:, 1]
        j3_clean_p = ak.pad_none(j_clean_p, target=3)[:, 2]
        j4_clean_p = ak.pad_none(j_clean_p, target=4)[:, 3]

        j1_clean_l = ak.firsts(j_clean_l)
        j2_clean_l = ak.pad_none(j_clean_l, target=2)[:, 1]
        j3_clean_l = ak.pad_none(j_clean_l, target=3)[:, 2]
        j4_clean_l = ak.pad_none(j_clean_l, target=4)[:, 3]

        j1_clean_p_dphi = np.abs(j1_clean_p.delta_phi(met))
        j2_clean_p_dphi = np.abs(j2_clean_p.delta_phi(met))
        j3_clean_p_dphi = np.abs(j3_clean_p.delta_phi(met))
        j4_clean_p_dphi = np.abs(j4_clean_p.delta_phi(met))

        j1_clean_e_dphi = np.abs(j1_clean_l.delta_phi(met))
        j2_clean_e_dphi = np.abs(j2_clean_l.delta_phi(met))
        j3_clean_e_dphi = np.abs(j3_clean_l.delta_phi(met))
        j4_clean_e_dphi = np.abs(j4_clean_l.delta_phi(met))

        j1_clean_m_dphi = np.abs(j1_clean_l.delta_phi(met))
        j2_clean_m_dphi = np.abs(j2_clean_l.delta_phi(met))
        j3_clean_m_dphi = np.abs(j3_clean_l.delta_phi(met))
        j4_clean_m_dphi = np.abs(j4_clean_l.delta_phi(met))
        
        """ Define the selections """

        selection.add('zero_trk_e', n_trk_e == 0)
        selection.add('zero_trk_m', n_trk_m == 0)
        selection.add('zero_trk_pi', n_trk_pi == 0)
        selection.add('zero_veto_j', n_j_veto == 0)
        selection.add('zero_e', n_e_veto == 0)
        selection.add('zero_m', n_m_loose == 0)
        selection.add('zero_t', n_t_medium == 0)
        selection.add('one_veto_lepton', ((n_e_veto == 1) & (n_m_loose == 0)) | ((n_e_veto == 0) & (n_m_loose == 1)))
        selection.add('one_e', n_e_medium == 1)
        selection.add('one_m', n_m_medium == 1)
        selection.add('one_p', n_p_medium == 1)
        selection.add('two_e', n_e_medium == 2)
        selection.add('two_m', n_m_medium == 2)
        selection.add('leading_e_pt40', leading_e.pt > 40)
        selection.add('second_e_pt20', second_e.pt > 20)
        selection.add('ossf_ee', (leading_e.charge != second_e.charge))
        selection.add('leading_m_pt50', leading_m.pt > 50)
        selection.add('second_m_pt20', second_m.pt > 20)
        selection.add('ossf_mm', (leading_m.charge != second_m.charge))
        selection.add('mee_50', mee > 50)
        selection.add('pee_200', pee > 200)
        selection.add('mmm_50', mmm > 50)
        selection.add('pmm_200', pmm > 200)
        selection.add('mee_Z_window', (mee > 81) & (mee < 101))
        selection.add('mmm_Z_window', (mmm > 81) & (mmm < 101))
        selection.add('zero_b', n_b == 0)
        selection.add('one_b', n_b >= 1)
        selection.add('one_b_clean_p', ak.num(b_clean_p, axis=1) >= 1)
        selection.add('one_b_clean_l', ak.num(b_clean_l, axis=1) >= 1)
        selection.add('one_b_tight', ak.num(b_tight, axis=1) >=1)
        selection.add('one_b_loose', ak.num(b_loose, axis=1) >=1)
        selection.add('exact_one_b', n_b == 1)
        selection.add('two_b', n_b >= 2)
        selection.add('two_j', n_j_good >= 2)
        selection.add('five_j', n_j_good >= 5)
        selection.add('five_j_clean_p', n_j_clean_p >= 5)
        selection.add('five_j_clean_l', n_j_clean_l >= 5)
        #selection.add('ht_300', scalarHT > 300)
        selection.add('met_250', met.pt > 250)
        selection.add('met_250_reverse', met.pt < 250)
        selection.add('uT_250_GCR', uT['cat4_GCR_highDeltaM'].r > 250)
        selection.add('mT_100', (ak.all(mT_m < 100, axis=1) & ak.all(mT_e < 100, axis=1)))
        selection.add('puppi/calo', met.pt / calo_met.pt < 5)
        selection.add('opening_angles_preselection', (j1_met_dphi > 0.5) & (j2_met_dphi > 0.15) & ak.fill_none(j3_met_dphi > 0.15, True))
        selection.add('opening_angles_highDeltaM', (j1_met_dphi > 0.5) & (j2_met_dphi > 0.5) & (j3_met_dphi > 0.5) & (j4_met_dphi > 0.5))
        selection.add('opening_angles_QCDCR', (j1_met_dphi < 0.5) | (j2_met_dphi < 0.15) | ak.fill_none(j3_met_dphi < 0.15, False))
        selection.add('opening_angles_QCDCR_highDeltaM', (j1_met_dphi < 0.5) | (j2_met_dphi < 0.5) | (j3_met_dphi < 0.5) | (j4_met_dphi < 0.5))
        selection.add('opening_angles_GCR_highDeltaM', (j1_clean_p_dphi > 0.5) & (j2_clean_p_dphi > 0.5) & (j3_clean_p_dphi > 0.5) & (j4_clean_p_dphi > 0.5))
        selection.add('opening_angles_DY2E_highDeltaM', (j1_clean_e_dphi > 0.5) & (j2_clean_e_dphi > 0.5) & (j3_clean_e_dphi > 0.5) & (j4_clean_e_dphi > 0.5))
        selection.add('opening_angles_DY2M_highDeltaM', (j1_clean_m_dphi > 0.5) & (j2_clean_m_dphi > 0.5) & (j3_clean_m_dphi > 0.5) & (j4_clean_m_dphi > 0.5))
        selection.add('dphi123_0p1', (j1_met_dphi < 0.1) | (j2_met_dphi < 0.1) | (ak.fill_none(j3_met_dphi < 0.1, False)))

        regions = {
            'cat1_preselection': [
                'lumimask', 'met_filters', 'zero_veto_j',
                'signal_trigger',
                'zero_trk_e', 'zero_trk_m', 'zero_trk_pi',
                'zero_m', 'zero_t', 'zero_e', 'two_j',
                'met_250', 'puppi/calo',
                'opening_angles_preselection'
            ],
            'cat2_LLCR_highDeltaM': [
                'lumimask', 'met_filters', 'zero_veto_j',
                'signal_trigger',
                'zero_trk_e', 'zero_trk_m', 'zero_trk_pi',
                'zero_t', 'five_j', 'one_b',
                'one_veto_lepton', 'mT_100',
                'met_250', 'puppi/calo',
                'opening_angles_highDeltaM'
            ],
            'cat3_QCDCR_highDeltaM': [
                'lumimask', 'met_filters', 'zero_veto_j',
                'signal_trigger',
                'zero_trk_e', 'zero_trk_m', 'zero_trk_pi',
                'zero_m', 'zero_t', 'zero_e', 'five_j', 'one_b',
                'met_250', 'puppi/calo',
                'opening_angles_QCDCR_highDeltaM', 'dphi123_0p1'
            ],
            'cat4_GCR_highDeltaM': [
                'lumimask', 'met_filters', 'zero_veto_j',
                'photon_trigger',
                'zero_trk_e', 'zero_trk_m', 'zero_trk_pi', 'one_p',
                'zero_m', 'zero_t', 'zero_e', 'five_j_clean_p', 'one_b_clean_p',
                'met_250_reverse', 'puppi/calo', 'uT_250_GCR',
                'opening_angles_GCR_highDeltaM'
            ],
            'cat5_DY2E_highDeltaM': [
                'lumimask', 'met_filters', 'zero_veto_j',
                'electron_trigger',
                'zero_trk_e', 'zero_trk_m', 'zero_trk_pi',
                'zero_t','five_j_clean_l', 'one_b_clean_l', 'zero_m', 'two_e',
                'leading_e_pt40', 'second_e_pt20', 'mee_50', 'ossf_ee', 'pee_200', 'mee_Z_window',
                'puppi/calo',
                'opening_angles_DY2E_highDeltaM'
            ],
            'cat6_DY2M_highDeltaM': [
                'lumimask', 'met_filters', 'zero_veto_j',
                'muon_trigger',
                'zero_trk_e', 'zero_trk_m', 'zero_trk_pi',
                'zero_t','five_j_clean_l', 'one_b_clean_l', 'zero_e', 'two_m',
                'leading_m_pt50', 'second_m_pt20', 'mmm_50', 'ossf_mm', 'pmm_200', 'mmm_Z_window',
                'puppi/calo',
                'opening_angles_DY2M_highDeltaM'
            ],
            'cat7_SR_highDeltaM': [
                'lumimask', 'met_filters', 'zero_veto_j',
                'signal_trigger',
                'zero_trk_e', 'zero_trk_m', 'zero_trk_pi',
                'zero_m', 'zero_t', 'zero_e', 'five_j', 'one_b',
                'met_250', 'puppi/calo',
                'opening_angles_highDeltaM'
            ],
        }
        if not isData:
            weights.add('genweight', events.genWeight)
            # PU reweighting and uncertainties
            pu_nom, pu_up, pu_down = get_pu_weight(self._year, events.Pileup.nTrueInt)
            weights.add('pileup', pu_nom, pu_up, pu_down)
            # Top pT reweighting
            gen = events.GenPart
            gen['isTop'] = (abs(gen.pdgId) == 6) & gen.hasFlags(['fromHardProcess', 'isLastCopy'])
            genTops = gen[gen.isTop]
            top_pt_sf = np.ones(len(events), dtype='float')
            if ('TTto' in dataset):
                top_pt_sf = ak.to_numpy(np.sqrt(get_top_pt_reweight(genTops.pt[:,0]) * get_top_pt_reweight(genTops.pt[:,1])))
            weights.add('top_pt_reweight', top_pt_sf)
            # btagging sf and uncertainties
            btagSF, btagSF_bc_correlatedUp, btagSF_bc_correlatedDown, btagSF_bc_uncorrelatedUp, btagSF_bc_uncorrelatedDown, \
            btagSF_light_correlatedUp, btagSF_light_correlatedDown, btagSF_light_uncorrelatedUp, btagSF_light_uncorrelatedDown = get_btag_weight('UParTAK4', self._year, 'medium', caller).btag_weight(j_good.pt, j_good.eta, j_good.hadronFlavour, j_good.isupartM)
            weights.add('btagSF', btagSF)
            weights.add('btagSF_bc_correlated', np.ones(len(events), dtype='float'), btagSF_bc_correlatedUp/btagSF, btagSF_bc_correlatedDown/btagSF)
            weights.add('btagSF_bc_uncorrelated', np.ones(len(events), dtype='float'), btagSF_bc_uncorrelatedUp/btagSF, btagSF_bc_uncorrelatedDown/btagSF)
            weights.add('btagSF_light_correlated', np.ones(len(events), dtype='float'), btagSF_light_correlatedUp/btagSF, btagSF_light_correlatedDown/btagSF)
            weights.add('btagSF_light_uncorrelated', np.ones(len(events), dtype='float'), btagSF_light_uncorrelatedUp/btagSF, btagSF_light_uncorrelatedDown/btagSF)

            # electron id sf
            ele_id_sf = np.ones(len(events), dtype='float')
            ele_id_sf_up = np.ones(len(events), dtype='float')
            ele_id_sf_down = np.ones(len(events), dtype='float')

            # Exactly one veto electron
            mask_one_veto_e = ak.to_numpy(n_e_veto == 1)
            ele_veto_evt_sf = ak.to_numpy(ak.fill_none(ak.prod(e_veto.veto_id_sf, axis=1), 1.0))
            ele_veto_evt_sf_up = ak.to_numpy(ak.fill_none(ak.prod(e_veto.veto_id_sf_up, axis=1), 1.0))
            ele_veto_evt_sf_down = ak.to_numpy(ak.fill_none(ak.prod(e_veto.veto_id_sf_down, axis=1), 1.0))
            ele_id_sf[mask_one_veto_e] = ele_veto_evt_sf[mask_one_veto_e]
            ele_id_sf_up[mask_one_veto_e] = ele_veto_evt_sf_up[mask_one_veto_e]
            ele_id_sf_down[mask_one_veto_e] = ele_veto_evt_sf_down[mask_one_veto_e]

            # Exactly two medium electrons
            mask_two_medium_e = ak.to_numpy(n_e_medium == 2)
            ele_medium_evt_sf = ak.to_numpy(ak.fill_none(ak.prod(e_medium.medium_id_sf, axis=1), 1.0))
            ele_medium_evt_sf_up = ak.to_numpy(ak.fill_none(ak.prod(e_medium.medium_id_sf_up, axis=1), 1.0))
            ele_medium_evt_sf_down = ak.to_numpy(ak.fill_none(ak.prod(e_medium.medium_id_sf_down, axis=1), 1.0))
            ele_id_sf[mask_two_medium_e] = ele_medium_evt_sf[mask_two_medium_e]
            ele_id_sf_up[mask_two_medium_e] = ele_medium_evt_sf_up[mask_two_medium_e]
            ele_id_sf_down[mask_two_medium_e] = ele_medium_evt_sf_down[mask_two_medium_e]
            
            weights.add('electron_id', ele_id_sf, ele_id_sf_up, ele_id_sf_down)

            # electron hlt sf
            ele_hlt_sf = np.ones(len(events), dtype='float')
            ele_hlt_sf_up = np.ones(len(events), dtype='float')
            ele_hlt_sf_down = np.ones(len(events), dtype='float')

            ele_hlt_evt_sf = ak.to_numpy(ak.fill_none(ak.prod(e_medium.hlt_sf, axis=1), 1.0))
            ele_hlt_evt_sf_up = ak.to_numpy(ak.fill_none(ak.prod(e_medium.hlt_sf_up, axis=1), 1.0))
            ele_hlt_evt_sf_down = ak.to_numpy(ak.fill_none(ak.prod(e_medium.hlt_sf_down, axis=1), 1.0))

            ele_hlt_sf[mask_two_medium_e] = ele_hlt_evt_sf[mask_two_medium_e]
            ele_hlt_sf_up[mask_two_medium_e] = ele_hlt_evt_sf_up[mask_two_medium_e]
            ele_hlt_sf_down[mask_two_medium_e] = ele_hlt_evt_sf_down[mask_two_medium_e]

            weights.add('electron_hlt', ele_hlt_sf, ele_hlt_sf_up, ele_hlt_sf_down)

            # muon id sf
            mu_id_sf = np.ones(len(events), dtype='float')
            mu_id_sf_up = np.ones(len(events), dtype='float')
            mu_id_sf_down = np.ones(len(events), dtype='float')

            # Exactly one loose muon
            mask_one_loose_m = ak.to_numpy(n_m_loose == 1)
            mu_loose_evt_sf = ak.to_numpy(ak.fill_none(ak.prod(m_loose.loose_id_sf, axis=1), 1.0))
            mu_loose_evt_sf_up = ak.to_numpy(ak.fill_none(ak.prod(m_loose.loose_id_sf_up, axis=1), 1.0))
            mu_loose_evt_sf_down = ak.to_numpy(ak.fill_none(ak.prod(m_loose.loose_id_sf_down, axis=1), 1.0))

            mu_id_sf[mask_one_loose_m] = mu_loose_evt_sf[mask_one_loose_m]
            mu_id_sf_up[mask_one_loose_m] = mu_loose_evt_sf_up[mask_one_loose_m]
            mu_id_sf_down[mask_one_loose_m] = mu_loose_evt_sf_down[mask_one_loose_m]

            # Exactly two medium muons
            mask_two_medium_m = ak.to_numpy(n_m_medium == 2)
            mu_medium_evt_sf = ak.to_numpy(ak.fill_none(ak.prod(m_medium.medium_id_sf, axis=1), 1.0))
            mu_medium_evt_sf_up = ak.to_numpy(ak.fill_none(ak.prod(m_medium.medium_id_sf_up, axis=1), 1.0))
            mu_medium_evt_sf_down = ak.to_numpy(ak.fill_none(ak.prod(m_medium.medium_id_sf_down, axis=1), 1.0))

            mu_id_sf[mask_two_medium_m] = mu_medium_evt_sf[mask_two_medium_m]
            mu_id_sf_up[mask_two_medium_m] = mu_medium_evt_sf_up[mask_two_medium_m]
            mu_id_sf_down[mask_two_medium_m] = mu_medium_evt_sf_down[mask_two_medium_m]

            weights.add('muon_id', mu_id_sf, mu_id_sf_up, mu_id_sf_down)

            # muon hlt sf
            mu_hlt_sf = np.ones(len(events), dtype='float')
            mu_hlt_sf_up = np.ones(len(events), dtype='float')
            mu_hlt_sf_down = np.ones(len(events), dtype='float')

            mu_hlt_evt_sf = ak.to_numpy(ak.fill_none(ak.prod(m_medium.hlt_sf, axis=1), 1.0))
            mu_hlt_evt_sf_up = ak.to_numpy(ak.fill_none(ak.prod(m_medium.hlt_sf_up, axis=1), 1.0))
            mu_hlt_evt_sf_down = ak.to_numpy(ak.fill_none(ak.prod(m_medium.hlt_sf_down, axis=1), 1.0))

            mu_hlt_sf[mask_two_medium_m] = mu_hlt_evt_sf[mask_two_medium_m]
            mu_hlt_sf_up[mask_two_medium_m] = mu_hlt_evt_sf_up[mask_two_medium_m]
            mu_hlt_sf_down[mask_two_medium_m] = mu_hlt_evt_sf_down[mask_two_medium_m]

            weights.add('muon_hlt', mu_hlt_sf, mu_hlt_sf_up, mu_hlt_sf_down)

            # photon id sf
            photon_id_sf = np.ones(len(events), dtype='float')
            photon_id_sf_up = np.ones(len(events), dtype='float')
            photon_id_sf_down = np.ones(len(events), dtype='float')
            mask_gcr_photon = ak.to_numpy(selection.all(*regions['cat4_GCR_highDeltaM']))

            photon_evt_sf = ak.to_numpy(ak.fill_none(ak.prod(p_medium.id_sf, axis=1), 1.0))
            photon_evt_sf_up = ak.to_numpy(ak.fill_none(ak.prod(p_medium.id_sf_up, axis=1), 1.0))
            photon_evt_sf_down = ak.to_numpy(ak.fill_none(ak.prod(p_medium.id_sf_down, axis=1), 1.0))

            photon_id_sf[mask_gcr_photon] = photon_evt_sf[mask_gcr_photon]
            photon_id_sf_up[mask_gcr_photon] = photon_evt_sf_up[mask_gcr_photon]
            photon_id_sf_down[mask_gcr_photon] = photon_evt_sf_down[mask_gcr_photon]

            weights.add('photon_id', photon_id_sf, photon_id_sf_up, photon_id_sf_down)

        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar

        def fill(region, systematic):
            cut = selection.all(*regions[region])
            if self._shift_name is not None:
                sys_name = self._shift_name
            else:
                sys_name = 'nominal' if systematic is None else systematic
            if systematic in weights.variations:
                weight = weights.weight(modifier=systematic)[cut]
            else:
                weight = weights.weight()[cut]
            output['template'].fill(
                region=region,
                systematic=sys_name,
                met=met.pt[cut],
                weight=weight
            )
            output['metpt'].fill(
                region=region,
                systematic=sys_name,
                signal_trigger= signal_triggers[cut],
                met=met.pt[cut],
                weight=weight
            )
            output['recoilpt'].fill(
                region=region,
                systematic=sys_name,
                recoilpt=uT[region].r[cut],
                weight=weight
            )
            if systematic is None:
                variables = {
                    'metpt_10GeVbins': met.pt,
                    'metphi': met.phi,
                    'recoilphi': uT[region].phi,
                    'nElectron': n_e_medium,
                    'nMuon': n_m_medium,
                    'nJet': n_j_good,
                    'j1pt': j1.pt,
                    'j1eta': j1.eta,
                    'j1phi': j1.phi,
                    'j2pt': j2.pt,
                    'j2eta': j2.eta,
                    'j2phi': j2.phi,
                    'nb_loose': nb_loose,
                    'nb_medium': nb_medium,
                    'nb_tight': nb_tight,
                    'b1_medium_pt': b1.pt,
                    'b1_medium_eta': b1.eta,
                    'b1_medium_phi': b1.phi,
                    'b1_tight_pt': b1_tight.pt,
                    'b1_tight_eta': b1_tight.eta,
                    'b1_tight_phi': b1_tight.phi,
                    'b1_loose_pt': b1_loose.pt,
                    'b1_loose_eta': b1_loose.eta,
                    'b1_loose_phi': b1_loose.phi,
                    'ht': scalarHT[region],
                    'nPV': npv,
                    'nfj': nfj_good,
                    'fj1pt': ak.fill_none(ak.firsts(fj_good).pt, -99),
                    'fj1mass': ak.fill_none(ak.firsts(fj_good).mass, -99),
                    'fj1msd': ak.fill_none(ak.firsts(fj_good).msoftdrop, -99),
                    'fj1phi': ak.fill_none(ak.firsts(fj_good).phi, -99),
                    'fj1eta': ak.fill_none(ak.firsts(fj_good).eta, -99),
                    'fj1TvsQCD': ak.fill_none(ak.firsts(fj_good).particleNetWithMass_TvsQCD, -99),
                    'fj1WvsQCD': ak.fill_none(ak.firsts(fj_good).particleNetWithMass_WvsQCD, -99),
                    'fj1QCD': ak.fill_none(ak.firsts(fj_good).particleNetWithMass_QCD, -99),
                }
                if region in mll:
                    variables['mll'] = mll[region]
                if region in pll:
                    variables['pll'] = pll[region]
                for variable in output:
                    if variable not in variables:
                        continue
                    normalized_variable = {variable: normalize(variables[variable], cut)}
                    output[variable].fill(
                        region=region,
                        systematic=sys_name,
                        **normalized_variable,
                        weight=weight
                    )
        if self._shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [None]
        for region in regions:
            if region not in selected_regions:
                continue
            ### Adding HT > 300 cut
            selection.add('ht_300_'+region, scalarHT[region] > 300)
            regions[region].append('ht_300_'+region)

            for systematic in systematics:
                if isData and systematic is not None:
                    continue
                fill(region, systematic)
        scale = 1
        if self._xsec[dataset] > 0:
            scale = self._lumi * self._xsec[dataset]
        
        for key in output:
            if key == 'sumw':
                continue
            output[key] *= scale

        return output
    
    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    parser.add_option('-m', '--metadata', help='metadata', dest='metadata')
    parser.add_option('-n', '--name', help='name', dest='name')
    parser.add_option('--shift', help='systematic shift name', dest='shift', default=None)    
    (options, args) = parser.parse_args()

    with gzip.open('metadata/'+options.metadata+'.json.gz') as fin:
        samplefiles = json.load(fin)
        xsec = {k: v['xs'] for k,v in samplefiles.items()}

    corrections = load('data/corrections.coffea')
    ids         = load('data/ids.coffea')
    common      = load('data/common.coffea')

    processor_instance=AnalysisProcessor(year=options.year,
                                         xsec=xsec,
                                         corrections=corrections,
                                         ids=ids,
                                         common=common,
                                         shift_name=options.shift)

    save(processor_instance, 'data/stop_'+options.name+'.processor')