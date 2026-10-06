"""Unchanged physical-variable bins and labels from the legacy histogrammer."""

LOWDM_VARIABLE_SPECS = {
    "met": {
        "branch": "met",
        "bins": [
            0,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            500,
            650,
            800,
            1000,
            1500
        ],
        "overflow_policy": "fold",
        "xlabel": "$p_{T}^{miss}$ (GeV)"
    },
    "ht": {
        "branch": "ht",
        "bins": [
            0,
            300,
            500,
            700,
            1000,
            1500,
            2000,
            3000
        ],
        "xlabel": "$H_{T}$ (GeV)"
    },
    "njet": {
        "branch": "njet",
        "bins": [
            -0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            5.5,
            6.5,
            8.5,
            12.5
        ],
        "xlabel": "$N_{j}$"
    },
    "nb_medium_lowdm": {
        "branch": "nb_medium_lowdm",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            6.5
        ],
        "xlabel": "$N_{b}$ medium"
    },
    "nb_loose_lowdm": {
        "branch": "nb_loose_lowdm",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            6.5
        ],
        "xlabel": "$N_{b}$ loose"
    },
    "n_e_veto": {
        "branch": "n_e_veto",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5
        ],
        "xlabel": "$N_{e}$ veto"
    },
    "n_m_loose": {
        "branch": "n_m_loose",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5
        ],
        "xlabel": "$N_{\\mu}$ loose"
    },
    "lowdm_mtb": {
        "branch": "lowdm_mtb",
        "bins": [
            0,
            50,
            100,
            150,
            200,
            300,
            500,
            800,
            1200
        ],
        "xlabel": "low-$\\Delta m$ $m_{T}^{b}$ (GeV)"
    },
    "lowdm_met_sqrt_ht": {
        "branch": "lowdm_met_sqrt_ht",
        "bins": [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            40,
            60
        ],
        "xlabel": "$p_{T}^{miss}/\\sqrt{H_{T}}$"
    },
    "lowdm_isr_pt": {
        "branch": "lowdm_isr_pt",
        "bins": [
            0,
            200,
            250,
            300,
            350,
            400,
            500,
            650,
            800,
            1000,
            1500
        ],
        "xlabel": "ISR $p_{T}$ (GeV)"
    },
    "lowdm_isr_dphi": {
        "branch": "lowdm_isr_dphi",
        "bins": [
            0,
            0.5,
            1,
            1.5,
            2,
            2.5,
            3.1416
        ],
        "xlabel": "$\\Delta\\phi$(ISR, $p_{T}^{miss}$)"
    },
    "lowdm_ptb": {
        "branch": "lowdm_ptb",
        "bins": [
            0,
            30,
            60,
            100,
            150,
            200,
            300,
            500,
            800
        ],
        "xlabel": "low-$\\Delta m$ $p_{T}^{b}$ (GeV)"
    },
    "n_lowdm_isr": {
        "branch": "n_lowdm_isr",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            5.5
        ],
        "xlabel": "$N_{ISR}$"
    },
    "recoil_gcr": {
        "branch": "recoil_gcr",
        "bins": [
            0,
            200,
            250,
            300,
            350,
            400,
            500,
            650,
            800,
            1000,
            1500
        ],
        "overflow_policy": "fold",
        "xlabel": "Photon recoil $p_{T}$ (GeV)"
    },
    "recoil_dy2e": {
        "branch": "recoil_dy2e",
        "bins": [
            0,
            200,
            250,
            300,
            350,
            400,
            500,
            650,
            800,
            1000,
            1500
        ],
        "overflow_policy": "fold",
        "xlabel": "Dielectron recoil $p_{T}$ (GeV)"
    },
    "recoil_dy2m": {
        "branch": "recoil_dy2m",
        "bins": [
            0,
            200,
            250,
            300,
            350,
            400,
            500,
            650,
            800,
            1000,
            1500
        ],
        "overflow_policy": "fold",
        "xlabel": "Dimuon recoil $p_{T}$ (GeV)"
    },
    "mee": {
        "branch": "mee",
        "bins": [
            50,
            70,
            81,
            86,
            91,
            96,
            101,
            120,
            150
        ],
        "xlabel": "$m_{ee}$ (GeV)"
    },
    "mmm": {
        "branch": "mmm",
        "bins": [
            50,
            70,
            81,
            86,
            91,
            96,
            101,
            120,
            150
        ],
        "xlabel": "$m_{\\mu\\mu}$ (GeV)"
    },
    "n_photon_medium": {
        "branch": "n_photon_medium",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5
        ],
        "xlabel": "$N_{\\gamma}$ medium"
    },
    "njet_photon_clean": {
        "branch": "njet_photon_clean",
        "bins": [
            -0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            5.5,
            6.5,
            8.5,
            12.5
        ],
        "xlabel": "$N_{j}$ photon-cleaned"
    },
    "nb_photon_clean": {
        "branch": "nb_photon_clean",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            6.5
        ],
        "xlabel": "$N_{b}$ photon-cleaned"
    },
    "ht_photon_clean": {
        "branch": "ht_photon_clean",
        "bins": [
            0,
            300,
            500,
            700,
            1000,
            1500,
            2000,
            3000
        ],
        "xlabel": "Photon-cleaned $H_{T}$ (GeV)"
    },
    "njet_lepton_clean": {
        "branch": "njet_lepton_clean",
        "bins": [
            -0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            5.5,
            6.5,
            8.5,
            12.5
        ],
        "xlabel": "$N_{j}$ lepton-cleaned"
    },
    "nb_lepton_clean": {
        "branch": "nb_lepton_clean",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            6.5
        ],
        "xlabel": "$N_{b}$ lepton-cleaned"
    },
    "ht_lepton_clean": {
        "branch": "ht_lepton_clean",
        "bins": [
            0,
            300,
            500,
            700,
            1000,
            1500,
            2000,
            3000
        ],
        "xlabel": "Lepton-cleaned $H_{T}$ (GeV)"
    },
    "leading_lowdm_fatjet_pt": {
        "branch": "lowdm_fatjet_pt",
        "source": "first",
        "bins": [
            0,
            200,
            250,
            300,
            350,
            400,
            500,
            650,
            800,
            1000,
            1500
        ],
        "xlabel": "Leading low-$\\Delta m$ AK8 $p_{T}$ (GeV)"
    },
    "leading_lowdm_fatjet_msd": {
        "branch": "lowdm_fatjet_msd",
        "source": "first",
        "bins": [
            0,
            30,
            50,
            70,
            90,
            110,
            140,
            200,
            300
        ],
        "xlabel": "Leading low-$\\Delta m$ AK8 $m_{SD}$ (GeV)"
    }
}

HIGHDM_DISTRIBUTION_VARIABLE_SPECS = {
    "nb": {
        "branch": "nb_medium",
        "branch_by_region": {
            "GCR": "nb_photon_clean",
            "DY2E": "nb_lepton_clean",
            "DY2M": "nb_lepton_clean"
        },
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            6.5
        ],
        "overflow_policy": "exclude",
        "xlabel": "$N_{b}$"
    },
    "njet": {
        "branch": "njet",
        "branch_by_region": {
            "GCR": "njet_photon_clean",
            "DY2E": "njet_lepton_clean",
            "DY2M": "njet_lepton_clean"
        },
        "bins": [
            -0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            5.5,
            6.5,
            8.5,
            10.5,
            12.5,
            16.5
        ],
        "overflow_policy": "exclude",
        "xlabel": "$N_{j}$"
    },
    "nfatjet": {
        "branch": "nfj",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            6.5
        ],
        "xlabel": "$N_{fj}$"
    },
    "ntop": {
        "branch": "nboosted_top",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            5.5
        ],
        "xlabel": "$N_{top}$"
    },
    "nw": {
        "branch": "nboosted_w",
        "bins": [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            5.5
        ],
        "xlabel": "$N_{W}$"
    },
    "ht": {
        "branch": "ht",
        "branch_by_region": {
            "GCR": "ht_photon_clean",
            "DY2E": "ht_lepton_clean",
            "DY2M": "ht_lepton_clean"
        },
        "bins": [
            0,
            300,
            500,
            700,
            900,
            1200,
            1500,
            2000,
            2500,
            3000
        ],
        "overflow_policy": "exclude",
        "xlabel": "$H_{T}$ (GeV)"
    },
    "ut": {
        "branch": "met",
        "branch_by_region": {
            "GCR": "recoil_gcr",
            "DY2E": "recoil_dy2e",
            "DY2M": "recoil_dy2m"
        },
        "regions": [
            "LLCR",
            "QCDCR",
            "GCR",
            "DY2E",
            "DY2M"
        ],
        "bins": [
            250,
            300,
            350,
            400,
            500,
            650,
            800,
            1000,
            1500
        ],
        "overflow_policy": "fold",
        "xlabel": "$U_{T}$ (GeV)"
    },
    "ptll": {
        "branch": "pee",
        "branch_by_region": {
            "DY2E": "pee",
            "DY2M": "pmm"
        },
        "regions": [
            "DY2E",
            "DY2M"
        ],
        "bins": [
            200,
            210,
            220,
            240,
            260,
            300,
            350,
            400,
            500,
            650,
            800,
            1000,
            1500
        ],
        "overflow_policy": "exclude",
        "xlabel": "$p_{T}(\\ell\\ell)$ (GeV)"
    },
    "met": {
        "branch": "met",
        "bins": [
            250,
            300,
            350,
            400,
            500,
            650,
            800,
            1000,
            1500
        ],
        "overflow_policy": "fold",
        "xlabel": "$p_{T}^{miss}$ (GeV)"
    },
    "jet_pt": {
        "branch": "j1pt",
        "bins": [
            20,
            30,
            40,
            50,
            70,
            100,
            150,
            200,
            300,
            500,
            800,
            1200,
            1600
        ],
        "xlabel": "Leading Jet $p_{T}$ (GeV)"
    },
    "fatjet_pt": {
        "branch": "fj1pt",
        "bins": [
            200,
            250,
            300,
            350,
            400,
            500,
            650,
            800,
            1000,
            1500
        ],
        "xlabel": "Leading FatJet $p_{T}$ (GeV)"
    },
    "bjet_pt": {
        "branch": "good_jet_pt",
        "mask_branch": "good_jet_b_medium",
        "source": "masked_first",
        "bins": [
            20,
            30,
            40,
            50,
            70,
            100,
            150,
            200,
            300,
            500,
            800,
            1200
        ],
        "xlabel": "Leading b-jet $p_{T}$ (GeV)"
    }
}

