"""Recover the adopted AK8 DeepCSV input from exact MiniAOD parents.

Only new intermediate ROOTs are writable. Source NanoAOD and MiniAOD stay
read-only. CMSSW owns retagging and the retained-event/jet identity match.
"""
V12_ROOT_CODE = r'''
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TObjString.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
class TopMixedV12InputWriter {
 TFile file; TTree* tree;
 public:
 TopMixedV12InputWriter(const char* path):file(path,"UPDATE") {
  if(file.IsZombie()) throw std::runtime_error("Cannot open transaction ROOT");
  tree=dynamic_cast<TTree*>(file.Get("Events"));
  if(!tree) throw std::runtime_error("Missing Events");
 }
 template<typename T> void appendTyped(const char* target, Long64_t n, const Long64_t* offsets,
                                      const T* data, const char* type) {
  if(tree->GetEntries()!=n) throw std::runtime_error("Events count mismatch");
  std::string name(target), cn="n"+name;
  if(tree->GetBranch(name.c_str()) || tree->GetBranch(cn.c_str()))
    throw std::runtime_error("Refusing to overwrite existing input branch");
  Long64_t maxlen=1;
  for(Long64_t i=0;i<n;++i) maxlen=std::max(maxlen,offsets[i+1]-offsets[i]);
  std::vector<T> values(maxlen); int count=0;
  auto cb=tree->Branch(cn.c_str(),&count,(cn+"/I").c_str());
  auto vb=tree->Branch(name.c_str(),values.data(),(name+"["+cn+"]/"+type).c_str());
  for(Long64_t i=0;i<n;++i) {
   count=offsets[i+1]-offsets[i];
   std::copy(data+offsets[i],data+offsets[i+1],values.begin());
   if(cb->Fill()<0 || vb->Fill()<0) throw std::runtime_error("Branch write failed");
  }
  cb->FlushBaskets(); vb->FlushBaskets(); cb->ResetAddress(); vb->ResetAddress();
  if(vb->GetEntries()!=n || cb->GetEntries()!=n) throw std::runtime_error("Incomplete branch");
 }
 void append(const char* target, Long64_t n, const Long64_t* offsets, const float* data) {
  appendTyped(target,n,offsets,data,"F");
 }
 void append_int(const char* target, Long64_t n, const Long64_t* offsets, const int* data) {
  appendTyped(target,n,offsets,data,"I");
 }
 void finish(const char* marker) {
  file.cd();
  if(tree->Write("Events",TObject::kWriteDelete)<=0) throw std::runtime_error("Tree commit failed");
  TObjString info(marker);
  if(info.Write("TopMixedInputs_metadata")<=0) throw std::runtime_error("Marker write failed");
  file.Flush(); file.Close();
 }
};
'''

def deepcsv_cmssw_config(inputs, parents, output):
    """Retag only retained events; the EDM file is temporary worker scratch."""
    import uproot
    selected = set()
    for path in inputs:
        with uproot.open(path) as root:
            rows = root['Events'].arrays(['run', 'luminosityBlock', 'event', 'nfatjet_nanoaod_pt'], library='np')
            selected.update((int(r), int(l), int(e)) for r, l, e, n in zip(
                rows['run'], rows['luminosityBlock'], rows['event'], rows['nfatjet_nanoaod_pt']) if n)
    if not parents or any(not p.startswith('/store/mc/RunIII2024Summer24MiniAODv6/') for p in parents):
        raise ValueError('Expected exact DAS parents from the approved MiniAODv6 campaign')
    events = [f'{r}:{l}:{e}' for r,l,e in sorted(selected)]
    config = '''import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from Configuration.Eras.Modifier_fastSim_cff import fastSim
process = cms.Process('DEEPINPUT', Run3_2024, fastSim)
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '150X_mcRun3_2024_realistic_v2', '')
process.options = cms.untracked.PSet(numberOfThreads=cms.untracked.uint32(1),numberOfStreams=cms.untracked.uint32(1))
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))
process.source = cms.Source('PoolSource', fileNames=cms.untracked.vstring(*PARENTS),
    eventsToProcess=cms.untracked.VEventRange(*EVENTS))
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
updateJetCollection(process, jetSource=cms.InputTag('slimmedJetsAK8', '', '@skipCurrentProcess'),
    pvSource=cms.InputTag('offlineSlimmedPrimaryVertices'), svSource=cms.InputTag('slimmedSecondaryVertices'),
    rParam=0.8, jetCorrections=('AK8PFPuppi', ['L1FastJet','L2Relative','L3Absolute','L2L3Residual'], 'None'),
    btagDiscriminators=['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb'], postfix='AK8DeepCSV')
process.out = cms.OutputModule('PoolOutputModule', fileName=cms.untracked.string(OUTPUT),
    outputCommands=cms.untracked.vstring('drop *','keep *_selectedUpdatedPatJetsAK8DeepCSV_*_*'))
process.end = cms.EndPath(process.out)
process.end.associate(process.patAlgosToolsTask)
'''
    return ('PARENTS='+repr(['root://cms-xrd-global.cern.ch/'+p for p in parents])+'\n'
            +'EVENTS='+repr(events)+'\nOUTPUT='+repr(str(output))+'\n'+config), len(events)

def append_parent_deepcsv(inputs, edm, parents):
    """CMSSW/FWLite stage: append only to new intermediate ROOTs, not NanoAOD."""
    import ROOT
    import json
    import math
    import os
    from array import array
    from pathlib import Path
    from DataFormats.FWLite import Events, Handle
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.DisableImplicitMT()
    tables, needed = {}, set()
    for filename in inputs:
        path = Path(filename)
        from .paths import ROOT as PACKAGE_ROOT
        if PACKAGE_ROOT / 'production' not in path.resolve().parents or path.suffix != '.root':
            raise ValueError('Only new canonical production intermediates are writable')
        f = ROOT.TFile.Open(str(path), 'READ')
        t = f.Get('Events')
        if not t or t.GetBranch('FatJet_btagDeepB'):
            raise ValueError('Refusing a NanoAOD target')
        if t.GetBranch('fatjet_btagdeepb_all'):
            f.Close()
            continue
        rows = []
        for row in t:
            key = (int(row.run), int(row.luminosityBlock), int(row.event))
            jets = list(zip(row.fatjet_nanoaod_pt, row.fatjet_eta_all, row.fatjet_phi_all))
            rows.append((key, jets))
            if jets:
                needed.add(key)
        tables[str(path)] = rows
        f.Close()
    scores = {}
    if needed:
        handle = Handle('std::vector<pat::Jet>')
        for ev in Events(str(edm)):
            aux = ev.eventAuxiliary()
            key = (int(aux.run()), int(aux.luminosityBlock()), int(aux.event()))
            if key not in needed:
                continue
            if key in scores:
                raise ValueError('Duplicate parent event identity: '+str(key))
            ev.getByLabel('selectedUpdatedPatJetsAK8DeepCSV', handle)
            if not handle.isValid():
                raise ValueError('Retagged MiniAOD collection missing')
            scores[key] = [(float(j.pt()), float(j.eta()), float(j.phi()),
                float(j.bDiscriminator('pfDeepCSVJetTags:probb')+j.bDiscriminator('pfDeepCSVJetTags:probbb')))
                for j in handle.product()]
        if set(scores) != needed:
            raise ValueError('Parent event coverage incomplete: '+str(len(needed-set(scores))))
    if not hasattr(ROOT, 'TopMixedV12InputWriter'):
        if not ROOT.gInterpreter.Declare(V12_ROOT_CODE):
            raise RuntimeError('Intermediate branch writer compilation failed')
    for filename, rows in tables.items():
        offsets, values = array('q', [0]), array('f')
        max_dr = max_relpt = 0.0
        for key, jets in rows:
            used = set()
            for pt, eta, phi in jets:
                matches = []
                for idx, (p, e, f, value) in enumerate(scores.get(key, [])):
                    dr = math.hypot(eta-e, math.atan2(math.sin(phi-f), math.cos(phi-f)))
                    rel = abs(pt-p)/max(pt, 1.)
                    if dr < .001 and rel < .002:
                        matches.append((idx, value, dr, rel))
                if len(matches) != 1 or matches[0][0] in used:
                    raise ValueError('Ambiguous NanoAOD/MiniAOD AK8 association: '+str(key))
                idx, value, dr, rel = matches[0]
                # Preserve raw finite discriminator values, including negative sentinels.
                if not math.isfinite(value):
                    raise ValueError('Nonfinite recovered AK8 DeepCSV: '+str(value))
                used.add(idx); values.append(value)
                max_dr, max_relpt = max(max_dr, dr), max(max_relpt, rel)
            offsets.append(len(values))
        marker = dict(schema='topmixed_v15_parent_deepcsv_v1', status='complete', entries=len(rows),
            parents=parents, cmssw=os.environ.get('CMSSW_VERSION'), global_tag='150X_mcRun3_2024_realistic_v2',
            discriminator='pfDeepCSVJetTags:probb + pfDeepCSVJetTags:probbb',
            target='fatjet_btagdeepb_all', source_nanoaod_modified=False,
            association='run/luminosityBlock/event and unique original AK8 p4',
            max_delta_r=max_dr, max_relative_pt=max_relpt)
        writer = ROOT.TopMixedV12InputWriter(filename)
        writer.append('fatjet_btagdeepb_all', len(rows), offsets, values or array('f',[0.]))
        writer.finish(json.dumps(marker, sort_keys=True)); del writer
        f = ROOT.TFile.Open(filename, 'READ'); t = f.Get('Events')
        if t.GetBranch('fatjet_btagdeepb_all').GetEntries() != len(rows):
            raise ValueError('Incomplete DeepCSV branch write')
        f.Close()
        path = Path(filename); meta = json.loads(path.with_suffix('.json').read_text())
        meta['topmixed_inputs'] = marker
        meta['branch_schema']['vector_float32'].append('fatjet_btagdeepb_all')
        path.with_suffix('.json').write_text(json.dumps(meta, indent=2, sort_keys=True)+'\n')
