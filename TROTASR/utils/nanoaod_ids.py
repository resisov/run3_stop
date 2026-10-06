from __future__ import annotations
import math
import json
import os
import contextlib
from pathlib import Path
from typing import Any, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import awkward as ak
import correctionlib
from .paths import payload_path as resolve_payload


def isTrackElectron(track, met_pt, met_phi, year):
    pt = track.pt
    eta = track.eta
    pdgId = abs(track.pdgId)
    relIso = track.pfRelIso03_all
    mT = np.sqrt(
        2 * pt * met_pt * (1 - np.cos(track.phi - met_phi))
    )
    mask = (
            (pt > 5)
            & (abs(eta) < 2.5)
            & (pdgId == 11)
            & (relIso < 0.2)
            & (mT < 100)
        )
    return mask


def isTrackMuon(track, met_pt, met_phi, year):
    pt = track.pt
    eta = track.eta
    pdgId = abs(track.pdgId)
    relIso = track.pfRelIso03_all
    mT = np.sqrt(
        2 * pt * met_pt * (1 - np.cos(track.phi - met_phi))
    )
    mask = (
            (pt > 5)
            & (abs(eta) < 2.5)
            & (pdgId == 13)
            & (relIso < 0.2)
            & (mT < 100)
        )
    return mask


def isTrackPion(track, met_pt, met_phi, year):
    pt = track.pt
    eta = track.eta
    pdgId = abs(track.pdgId)
    relIso = track.pfRelIso03_all
    mT = np.sqrt(
        2 * pt * met_pt * (1 - np.cos(track.phi - met_phi))
    )
    mask = (
            (pt > 10)
            & (abs(eta) < 2.5)
            & (pdgId == 211)
            & (relIso < 0.1)
            & (mT < 100)
        )
    return mask


def isVetoElectron(electron, year):
    pt = electron.pt
    eta = electron.eta #+ electron.deltaEtaSC
    cutBased = electron.cutBased
    miniIso = electron.miniPFRelIso_all
    mask = (
            (pt > 10)
            & (abs(eta) < 1.4442)
            & (cutBased >= 1)
            & (miniIso < 0.1)
        ) | (
            (pt > 10)
            & (abs(eta) > 1.5660)
            & (abs(eta) < 2.5)
            & (cutBased >= 1)
            & (miniIso < 0.1)
        )
    return mask


def isMediumElectron(electron, year):
    pt = electron.pt
    eta = electron.eta #+ electron.deltaEtaSC
    cutBased = electron.cutBased
    miniIso = electron.miniPFRelIso_all
    mask = (
            (pt > 10)
            & (abs(eta) < 1.4442)
            & (cutBased >= 3)
            & (miniIso < 0.1)
        ) | (
            (pt > 10)
            & (abs(eta) > 1.5660)
            & (abs(eta) < 2.5)
            & (cutBased >= 3)
            & (miniIso < 0.1)
        )
    return mask


def isLooseMuon(muon, year):
    pt = muon.pt
    eta = muon.eta
    miniIso = muon.miniPFRelIso_all
    looseId = muon.looseId
    mask = (
            (pt > 10)
            & (abs(eta) < 2.4)
            & (looseId)
            & (miniIso < 0.2)
        )
    return mask


def isMediumMuon(muon, year):
    pt = muon.pt
    eta = muon.eta
    miniIso = muon.miniPFRelIso_all
    mediumId = muon.mediumId
    mask = (
            (pt > 10)
            & (abs(eta) < 2.4)
            & (mediumId)
            & (miniIso < 0.2)
        )
    return mask


def isMediumTau(tau, met_pt, met_phi, year):
    pt = tau.pt
    eta = tau.eta
    phi = tau.phi
    dz = tau.dz
    decayMode = tau.decayMode
    idj = tau.idDeepTau2018v2p5VSjet
    mT = np.sqrt(
        2 * pt * met_pt * (1 - np.cos(phi - met_phi))
    )
    mask = (
            (pt > 20)
            & (abs(eta) < 2.5)
            & (abs(dz) < 0.2)
            & ~(decayMode == 5)
            & ~(decayMode == 6)
            & (idj >= 5)
            & (mT < 100)
        )
    return mask


def isGoodJet(jet, year):
    pt = jet.pt
    eta = jet.eta
    phi = jet.phi
    chHEF = jet.chHEF
    neHEF = jet.neHEF
    chEmEF = jet.chEmEF
    neEmEF = jet.neEmEF
    muEF = jet.muEF
    chMultiplicity = jet.chMultiplicity
    neMultiplicity = jet.neMultiplicity
    multiplicity = chMultiplicity + neMultiplicity

    def getJetID(eta, chHEF, neHEF, chEmEF, neEmEF, muEF, chMultiplicity, neMultiplicity, multiplicity):
        if year == '2024':
            evaluator = correctionlib.CorrectionSet.from_file(str(resolve_payload('data/JMESF/'+year+'/jetid.json.gz')))
        elif year == '2025':
            evaluator = correctionlib.CorrectionSet.from_file(str(resolve_payload('data/JMESF/2024/jetid.json.gz')))
        #evaluator = correctionlib.CorrectionSet.from_file('data/JMESF/2022pre/jetid.json.gz')
        corr = evaluator["AK4PUPPI_TightLeptonVeto"]
        counts = ak.num(eta)
        eta, chHEF, neHEF, chEmEF, neEmEF, muEF = ak.flatten(eta), ak.flatten(chHEF), ak.flatten(neHEF), ak.flatten(chEmEF), ak.flatten(neEmEF), ak.flatten(muEF)
        chMultiplicity, neMultiplicity, multiplicity = ak.flatten(chMultiplicity), ak.flatten(neMultiplicity), ak.flatten(multiplicity)
        args = (
            eta,
            chHEF, neHEF, chEmEF, neEmEF, muEF,
            chMultiplicity, neMultiplicity, multiplicity,
        )
        out = corr.evaluate(*args)
        return ak.unflatten(out, counts)
        
    jetId = getJetID(eta, chHEF, neHEF, chEmEF, neEmEF, muEF, chMultiplicity, neMultiplicity, multiplicity)
    mask = (pt > 30) & (abs(eta) < 2.4) & (jetId == 1)
    return mask


def isJetVeto(jet, year):
    pt = jet.pt
    eta = jet.eta
    phi = jet.phi
    evaluator = correctionlib.CorrectionSet.from_file(str(resolve_payload('data/JMESF/'+year+'/jetvetomaps.json.gz')))
    if year == '2024':
        corr = evaluator["Summer24Prompt24_RunBCDEFGHI_V1"]
    elif year == '2025':
        corr = evaluator["Summer24Prompt25_RunCDEFG_V1"]
    counts = ak.num(eta)
    pt = ak.flatten(pt)
    eta, phi = ak.flatten(eta), ak.flatten(phi)
    out = corr.evaluate('jetvetomap',eta,phi)
    mask = ((pt > 30) & (out != 0))
    return ak.unflatten(mask, counts)


def isGoodFatJet(jet, year):
    pt = jet.pt
    eta = jet.eta
    phi = jet.phi
    msd = jet.msoftdrop
    chHEF = jet.chHEF
    neHEF = jet.neHEF
    chEmEF = jet.chEmEF
    neEmEF = jet.neEmEF
    muEF = jet.muEF
    chMultiplicity = jet.chMultiplicity
    neMultiplicity = jet.neMultiplicity
    multiplicity = chMultiplicity + neMultiplicity

    def getJetID(eta, chHEF, neHEF, chEmEF, neEmEF, muEF, chMultiplicity, neMultiplicity, multiplicity):
        if year == '2024':
            evaluator = correctionlib.CorrectionSet.from_file(str(resolve_payload('data/JMESF/'+year+'/jetid.json.gz')))
        elif year == '2025':
            evaluator = correctionlib.CorrectionSet.from_file(str(resolve_payload('data/JMESF/2024/jetid.json.gz'))) ### Future update needed
        corr = evaluator["AK8PUPPI_TightLeptonVeto"]
        counts = ak.num(eta)
        eta, chHEF, neHEF, chEmEF, neEmEF, muEF = ak.flatten(eta), ak.flatten(chHEF), ak.flatten(neHEF), ak.flatten(chEmEF), ak.flatten(neEmEF), ak.flatten(muEF)
        chMultiplicity, neMultiplicity, multiplicity = ak.flatten(chMultiplicity), ak.flatten(neMultiplicity), ak.flatten(multiplicity)
        args = (
            eta,
            chHEF, neHEF, chEmEF, neEmEF, muEF,
            chMultiplicity, neMultiplicity, multiplicity,
        )
        out = corr.evaluate(*args)
        return ak.unflatten(out, counts)
    jetId = getJetID(eta, chHEF, neHEF, chEmEF, neEmEF, muEF, chMultiplicity, neMultiplicity, multiplicity)
    mask = (pt > 200) & (abs(eta) < 2.0) & (jetId == 1) & (msd > 60)
    return mask
