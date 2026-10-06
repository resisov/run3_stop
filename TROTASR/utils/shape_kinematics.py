"""In-memory retained-event JME propagation using the adopted reconstruction.

The formulas and thresholds are the existing retained_jme_shapes/rebuild_kinematics
ones, internalized here. No external analysis module or event cache is used.
"""
import awkward as ak
import numpy as np
from .ids import object_masks, clean_by_delta_r
from .region_kinematics import jet_kinematics, update_veto_leptons, delta_phi


def first(values, default=-99.):
    return np.asarray(ak.fill_none(ak.firsts(values), default))


def nth(values, i, default=999.):
    return np.asarray(ak.fill_none(ak.pad_none(values, i+1)[:, i], default))


def jet_block(pt, eta, phi, good, medium, met_phi):
    p, e, f = pt[good], eta[good], phi[good]
    d = delta_phi(f, met_phi)
    angles = [nth(d, i) for i in range(4)]
    return dict(njet=np.asarray(ak.sum(good, axis=1)), nb=np.asarray(ak.sum(medium, axis=1)),
                ht=np.asarray(ak.sum(p, axis=1)), j1pt=first(p), j1eta=first(e),
                j1phi=first(f), j2pt=nth(p, 1, -99.), angles=angles,
                min_dphi4=np.minimum.reduce(angles),
                open_pre=(angles[0]>.5)&(angles[1]>.15)&(angles[2]>.15),
                open_high=np.logical_and.reduce([x>.5 for x in angles]),
                qcd_open=np.logical_or.reduce([x<.5 for x in angles]),
                dphi123_0p1=np.logical_or.reduce([x<.1 for x in angles[:3]]))


def met_contribution(a, pt):
    mask = ((pt>15) & (abs(a['jet_eta_all'])<5.2)
            & ((a['jet_ch_em_ef']+a['jet_ne_em_ef'])<.9))
    delta = (a['jet_nanoaod_pt']-pt)*(1-a['jet_muon_subtr_factor'])
    return [np.asarray(ak.sum(ak.where(mask, delta*fn(a['jet_phi_all']), 0), axis=1))
            for fn in (np.cos, np.sin)]


def varied_view(a, momenta):
    """Propagate the same AK4 change to type-1 MET, then all consumed fields."""
    out = {k: a[k] for k in ak.fields(a)}
    before, after = met_contribution(a, a['jet_corrected_pt']), met_contribution(a, momenta['jet_corrected_pt'])
    x = np.asarray(a['met'])*np.cos(a['met_phi']) + after[0]-before[0]
    y = np.asarray(a['met'])*np.sin(a['met_phi']) + after[1]-before[1]
    out.update(momenta)
    out.update(met=np.hypot(x, y), met_phi=np.arctan2(y, x))
    out['puppi_met_corrected'], out['puppi_met_corrected_phi'] = out['met'], out['met_phi']
    return rebuild(out)


def rebuild(a, objects_only=False):
    out = dict(a)
    pt, eta, phi, tag = [out[k] for k in ('jet_corrected_pt','jet_eta_all','jet_phi_all','jet_btag_upart_all')]
    good = (pt>30)&(abs(eta)<2.4)&(out['jet_id_all']!=0)
    medium, loose = good&(tag>.1272), good&(tag>.0246)
    jets = jet_block(pt, eta, phi, good, medium, out['met_phi'])
    for name in ('njet','ht','j1pt','j1eta','j1phi','j2pt','min_dphi4'):
        out[name] = jets[name]
    out['nb_medium'] = out['nb_medium_lowdm'] = jets['nb']
    out['nb_loose'] = out['nb_loose_lowdm'] = np.asarray(ak.sum(loose, axis=1))
    out['pass_ht_300'] = jets['ht']>300
    out['pass_met_250'] = out['met']>250
    for name in ('open_pre','open_high','qcd_open','dphi123_0p1'):
        out['pass_'+name] = jets[name]
    for i, angle in enumerate(jets['angles'], 1):
        out['j%s_met_dphi'%i] = angle
    for name, value in dict(pt=pt, eta=eta, phi=phi, btag_upart=tag,
            hadron_flavour=out['jet_hadron_flavour_all'], b_loose=loose, b_medium=medium).items():
        out['good_jet_'+name] = value[good]
    if not objects_only:
        fpt, feta, fphi, mass, msd = [out[k] for k in ('fatjet_corrected_pt','fatjet_eta_all',
            'fatjet_phi_all','fatjet_corrected_mass','fatjet_msoftdrop_all')]
        fid = out['fatjet_id_all']!=0
        top, w = out['fatjet_top_score_all'], out['fatjet_w_score_all']
        tp = fid&(fpt>400)&(abs(feta)<2)&(msd>105)&(top>.5078)
        wp = fid&(fpt>200)&(abs(feta)<2)&(msd>60)&(msd<105)&(w>.9385)
        out['fatjet_boosted_top_pass_all'], out['fatjet_boosted_w_pass_all'] = tp, wp
        out['nboosted_top'], out['nboosted_w'] = [np.asarray(ak.sum(v,axis=1)) for v in (tp,wp)]
        out['nboosted_total'] = out['nboosted_top']+out['nboosted_w']
        fg = fid&(fpt>200)&(abs(feta)<2)&(msd>60)
        out['nfj'] = np.asarray(ak.sum(fg, axis=1))
        for name, value in dict(pt=fpt,eta=feta,phi=fphi,mass=mass,msd=msd,top_score=top,w_score=w).items():
            out['fj1'+('_' if 'score' in name else '')+name] = first(value[fg])
        isr = fid&(fpt>200)&(abs(feta)<2.4)
        out['n_lowdm_isr'] = np.asarray(ak.sum(isr,axis=1))
        old_eta, old_phi = out['lowdm_isr_eta'], out['lowdm_isr_phi']
        for name, value in dict(pt=fpt,eta=feta,phi=fphi,msd=msd,
                subjet_idx1=out['fatjet_subjet_index1_all'],subjet_idx2=out['fatjet_subjet_index2_all']).items():
            out['lowdm_fatjet_'+name] = value[isr]
            if name in ('pt','eta','phi'):
                out['lowdm_isr_'+name] = first(value[isr])
        # Only the original leading ISR subjet discriminator is retained. Do not
        # attach it to another AK8 when eligibility changes; use the trained sentinel.
        same_isr = (np.abs(out['lowdm_isr_eta']-old_eta)<1e-6)&(delta_phi(out['lowdm_isr_phi'],old_phi)<1e-6)
        out['lowdm_isr_subjet_btag_max'] = np.where(same_isr, out['lowdm_isr_subjet_btag_max'], -1.)
        out['pass_lowdm_topology_veto'] = (out['nboosted_top']==0)&(out['nboosted_w']==0)
    masks = object_masks(out)
    counts = dict(electron_veto='n_e_veto',electron_medium='n_e_medium',muon_loose='n_m_loose',
                  muon_medium='n_m_medium',photon_medium='n_photon_medium')
    for name, mask in masks.items():
        flavor = name.split('_')[0]
        for field in ('pt','eta','phi','eta_sc'):
            if flavor+'_'+field+'_all' in out:
                out[name+'_'+field] = out[flavor+'_'+field+'_all'][mask]
        out[counts[name]] = np.asarray(ak.sum(mask,axis=1))
    met, mp = np.asarray(out['met']), np.asarray(out['met_phi'])
    if objects_only:
        for flavor, mass_name, pt_name in (('electron','mee','pee'),('muon','mmm','pmm')):
            mask = masks[flavor+'_medium']
            pairs = [[nth(out[flavor+'_'+field+'_all'][mask], i, -99. if field=='pt' else 0.)
                      for field in ('pt','eta','phi','mass')] for i in (0,1)]
            pt1, eta1, phi1, m1 = pairs[0]
            pt2, eta2, phi2, m2 = pairs[1]
            x1, y1, z1 = pt1*np.cos(phi1), pt1*np.sin(phi1), pt1*np.sinh(eta1)
            x2, y2, z2 = pt2*np.cos(phi2), pt2*np.sin(phi2), pt2*np.sinh(eta2)
            e1 = np.sqrt(np.maximum(0.,m1*m1+x1*x1+y1*y1+z1*z1))
            e2 = np.sqrt(np.maximum(0.,m2*m2+x2*x2+y2*y2+z2*z2))
            out[mass_name] = np.sqrt(np.maximum(0.,(e1+e2)**2-(x1+x2)**2-(y1+y2)**2-(z1+z2)**2))
            out[pt_name] = np.sqrt(np.maximum(0.,pt1**2+pt2**2+2*pt1*pt2*np.cos(phi1-phi2)))
    # Leptons/photons do not change in JME endpoints; retain their mass/ptll
    # definitions and rebuild just recoil with the existing first-two convention.
    for region, flavor in (('dy2e','electron'),('dy2m','muon'),('gcr','photon')):
        nobj = 1 if flavor=='photon' else 2
        x, y = met*np.cos(mp), met*np.sin(mp)
        for i in range(nobj):
            p = nth(out[flavor+'_medium_pt'],i,0. if flavor=='photon' else -99.)
            f = nth(out[flavor+'_medium_phi'],i,0.)
            x, y = x+p*np.cos(f), y+p*np.sin(f)
        out['recoil_'+region], out['recoil_'+region+'_phi'] = np.hypot(x,y),np.arctan2(y,x)
    tau_mt = np.sqrt(2*out['tau_pt_all']*met*(1-np.cos(out['tau_phi_all']-mp)))
    tau = ((out['tau_pt_all']>20)&(abs(out['tau_eta_all'])<2.5)&(abs(out['tau_dz_all'])<.2)
           &(out['tau_decay_mode_all']!=5)&(out['tau_decay_mode_all']!=6)
           &(out['tau_deeptau_vsjet_all']>=5)&(tau_mt<100))
    out['pass_zero_tau'] = np.asarray(ak.sum(tau,axis=1))==0
    clean_masks = {}
    for region, flavor in (('gcr','photon'),('dy2e','electron'),('dy2m','muon')):
        clean = clean_by_delta_r(eta,phi,out[flavor+'_medium_eta'],out[flavor+'_medium_phi'],.2)
        clean_masks[flavor] = clean
        b = jet_block(pt,eta,phi,good&clean,medium&clean,out['recoil_'+region+'_phi'])
        out['pass_'+region+'_open_high'] = b['open_high']
        if region=='gcr':
            for field in ('ht','njet','nb'):
                out[field+'_photon_clean'] = b[field]
            out['pass_ht_photon_300'] = b['ht']>300
            for i, angle in enumerate(b['angles'],1):
                out['gcr_j%s_recoil_dphi'%i] = angle
            out['gcr_min_recoil_dphi4'] = b['min_dphi4']
        else:
            out['pass_'+region+'_ut_250'] = out['recoil_'+region]>250
    clean = clean_masks['electron']&clean_masks['muon']
    b = jet_block(pt,eta,phi,good&clean,medium&clean,mp)
    for field in ('ht','njet','nb'):
        out[field+'_lepton_clean'] = b[field]
    out['pass_ht_lepton_300'] = b['ht']>300
    b = jet_kinematics(pt,eta,phi,tag,good,met,mp)
    for field in ('mtb','ptb','met_sqrt_ht'):
        out['lowdm_'+field] = b[field]
    out['lowdm_isr_dphi'] = np.where(out['lowdm_isr_pt']>0,delta_phi(out['lowdm_isr_phi'],mp),-99.)
    view = ak.Array(out)
    update_veto_leptons(view)
    return view
