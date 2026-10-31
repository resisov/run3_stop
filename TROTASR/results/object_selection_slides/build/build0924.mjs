import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const source='/Users/taiwoomac/Desktop/Meeting/stop_meeting_0924.pptx';
const root='/Users/taiwoomac/Documents/All Hadronic Stop Analysis/TROTASR';
const workspaceDir=root+'/results/object_selection_slides';
const build=workspaceDir+'/build';
const skill='/Users/taiwoomac/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const python='/Users/taiwoomac/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const [cover,s2,s3,s4,s5,end]=p.slides.items;
const s6=s5.duplicate();s6.moveTo(5);
const s7=s5.duplicate();s7.moveTo(6);
end.moveTo(7);
const limitSlides=[s5,s6,s7];
const navy='#002060',blue='#4472C4',red='#C00000',black='#000000',white='#FFFFFF';
const font='Helvetica Neue Light',medium='Helvetica Neue Medium';

function text(slide,name,value,x,y,w,h,size=28,color=black,bold=false) {
 const s=slide.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;
 s.text.style={typeface:bold?medium:font,fontSize:size,color,bold,autoFit:'none',wrap:'square',verticalAlignment:'middle',insets:{left:0,right:0,top:0,bottom:0}};
 return s;
}
function box(slide,name,value,x,y,w,h,size=27,fill=blue) {
 const s=slide.shapes.add({name,geometry:'roundRect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:navy,width:1},borderRadius:7});
 s.text=value;
 s.text.style={typeface:font,fontSize:size,color:white,alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:{left:8,right:8,top:4,bottom:4}};
 return s;
}
function connect(slide,a,b) {
 return slide.shapes.connect(a,b,{kind:'straight',fromSide:'bottom',toSide:'top',line:{fill:navy,width:2},tail:{type:'triangle',width:'sm',length:'sm'}});
}
for(const [i,id] of ['sh/cnupgny5','sh/ja54na9g','sh/g36tgryd','sh/sb6xsvu9'].entries()) {
 const f=p.resolve(id);
 f.text=`${i+1}     2026. 9. 24.     Taiwoo KIM – Weekly Stop Meeting`;
 f.text.style={typeface:font,fontSize:18.6667,color:black,autoFit:'none',verticalAlignment:'middle',insets:{left:9.6,right:9.6,top:0,bottom:0}};
}
for(const [i,slide] of limitSlides.entries()){
 const title=slide.shapes.items.find(s=>String(s.text)==='Limit comparison');
 if(!title)throw new Error('Comparison title missing');
 title.text='Limit comparison: '+['T2tt','T2bW','T2tb'][i];
 const footer=slide.shapes.items.find(s=>String(s.text).includes('Weekly Stop Meeting'));
 footer.text=`${i+4}     2026. 9. 24.     Taiwoo KIM – Weekly Stop Meeting`;
 footer.text.style={typeface:font,fontSize:18.6667,color:black,autoFit:'none',verticalAlignment:'middle',insets:{left:9.6,right:9.6,top:0,bottom:0}};
}

// Use the four empty body slides of the supplied deck, preserving their frames.
const values=[
 ['Object','Inputs','Jet selection','Candidate selection','Tag requirement'],
 ['Boosted top\nNbst','1 AK8','pT > 400 GeV\n|η| < 2','mSD > 105 GeV','Top score > 0.5078'],
 ['Mixed top\nNmix','2 or 3 AK4\n+ 1 AK8','AK4: pT > 25 GeV, |η| < 2.5\nAK8: pT > 200 GeV, |η| < 2.5','No additional top\nmass or η window','TT / (TT + QCD)\n≥ 0.9027690887451172'],
 ['Resolved top\nNres','3 distinct AK4','Each AK4:\npT > 25 GeV, |η| < 2.5','100 ≤ m(3 AK4) ≤ 250 GeV\n|η(3 AK4)| < 2','TT / (TT + QCD)\n≥ 0.9433798789978027'],
 ['W\nNw','1 AK8','pT > 200 GeV\n|η| < 2','60 < mSD < 105 GeV','W score > 0.9385']
];
const table=s2.tables.add({rows:5,columns:5,left:53,top:151,width:1174,height:384,columnWidths:[155,145,318,268,288],values});
table.styleOptions={headerRow:true,bandedRows:false};
table.borders.assign({fill:'#A5A5A5',width:.7,style:'solid'});
[48,78,96,88,74].forEach((h,i)=>{table.rows[i].height=h;});
for(let r=0;r<5;r++) for(let c=0;c<5;c++) {
 const cell=table.getCell(r,c);cell.fill=r===0?blue:white;
 cell.text.style={typeface:r===0?medium:font,fontSize:r===0?24:21.5,color:r===0?white:black,autoFit:'none',wrap:'square',verticalAlignment:'middle',insets:{left:10,right:8,top:6,bottom:6}};
}
text(s2,'common-id','Jet ID and region-dependent photon/lepton cleaning apply before arbitration.',65,557,1150,34,25);
text(s2,'independent-collections','AK4, AK8 and SubJet are separate jet collections.',65,600,1120,34,27,navy);
text(s2,'mass-definition','mSD: AK8 soft-drop mass. TT and QCD: TROTA class probabilities.',65,643,1000,25,21);

// Native editable priority flow and explicit geometrical overlap conditions.
const ys=[148,233,318,403];
const nodes=[
 box(s3,'boosted','1  Boosted\nNbst',63,ys[0],210,66),
 box(s3,'mixed','2  Mixed\nNmix',63,ys[1],210,66),
 box(s3,'resolved','3  Resolved\nNres',63,ys[2],210,66),
 box(s3,'w','4  W\nNw',63,ys[3],210,66,27,navy)
];
for(let i=0;i<3;i++)connect(s3,nodes[i],nodes[i+1]);
[
 'Keep all passing AK8 jets.\nReserve their overlapping AK4 jets.',
 'Highest score first. Reject shared AK8\nor AK4 overlap with selected tops.',
 'Highest score first. Reject AK4 overlap\nwith selected tops.',
 'Reject shared AK8 or AK4 overlap\nwith selected tops. Keep independent W.'
].forEach((v,i)=>text(s3,'stage-'+i,v,298,ys[i],492,66,23.5));
text(s3,'overlap-heading','AK4–AK8 overlap',828,146,394,35,29,navy,true);
text(s3,'two-subjets','AK8 has two valid SubJets',828,196,394,31,24,black,true);
text(s3,'subjet-rule','ΔR(AK4, SubJet 1) < 0.4\nor ΔR(AK4, SubJet 2) < 0.4',828,234,396,71,24.5);
text(s3,'fallback-heading','Otherwise',828,326,394,31,24,black,true);
text(s3,'fallback-rule','ΔR(AK4, AK8 axis) < 0.8',828,366,394,45,25);
text(s3,'overlap-meaning','AK4 jets satisfying this rule\nenter the overlap veto.',828,435,394,61,24);
text(s3,'candidate-not-event','Only the overlapping candidate is rejected. Independent Mixed and W can coexist.',65,523,1150,38,25,navy);
text(s3,'current-limitations','No Boosted–Boosted or W–W pairwise veto. No direct ΔR(AK8, AK8) cut.',65,568,1150,34,23);
text(s3,'topology-selection','High-dM: Nbst + Nmix + Nres + Nw ≥ 1. Low-dM: all four counts = 0.',65,610,1020,35,24);
text(s3,'nb-independent','The overlap veto does not reduce Nb.',65,649,970,24,21);

const plotRoot=root+'/stats/nominal_20260924_sr118/plots';
const assets=[
 [s4,'predictions/highdm118_sr_2024_2025.png','fe93991bbd8eafe4d327c4f0e8943d24be7195927c1c9d92a5a9492aa29d0e5d',53,151,1174,368.32],
 [s5,'contours/expected_limit_t2tt_2024_2025_highdm118_lowdm30_xsec_topology_thermal_relic.png','63362d9bc8ed52d29934f7fbc63fa1e978bfd3761e8f3ba482f218aa802fe35c',660,201,525,437.5],
 [s6,'contours/expected_limit_t2bw_2024_2025_highdm118_lowdm30_xsec_topology_thermal_relic.png','99c3530b9dd0388d460b390b2f37c0c16fb0b3955d2efac2700e9bb23b67f294',660,201,525,437.5],
 [s7,'contours/expected_limit_t2tb_2024_2025_highdm118_lowdm30_xsec_topology_thermal_relic.png','3b1cebb3680f70bebc98517c4a267bf51dd879f77d1417e1fedbdfa7dce2095c',660,201,525,437.5]
];
for(const [slide,relative,hash,x,y,w,h] of assets){
 const bytes=await fs.readFile(plotRoot+'/'+relative);
 if(crypto.createHash('sha256').update(bytes).digest('hex')!==hash)throw new Error('Plot hash changed: '+relative);
 slide.images.add({blob:bytes,contentType:'image/png',alt:relative,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
text(s4,'sr-summary','21 classes, 118 high-dM bins. Low-dM retains 30 bins. SR data remain blinded.',65,533,1150,38,26);
text(s4,'sr-order','Category order: Nb, Nbst, Nmix, Nres, Nw. CRs retain the existing Nb streams.',65,577,1150,37,25);
text(s4,'sr-merges','B < 2 category merges, then the last two bins of classes 2, 15 and 20 combined.',65,622,1015,41,22.5);
const previousRoot=path.dirname(root)+'/autonomous_allhad/reports/lowdm_cr_nisr_merged_20260921/limits';
const previousHashes=['35d67948398f77dd8bcdc3a19d7f1e97c079f2a007d2673cfb330e46e699572c','3535088bddcbec57e9314c11583beb0965b3b92feb405067c8ca1a65e1c02dfc','0ca524ae8c59055c22878c68555c41707d7428c135d935ceb5817debfaf93ff4'];
for(const [i,slide] of limitSlides.entries()){
 const model=['T2tt','T2bW','T2tb'][i];
 const file=previousRoot+'/'+model+'/expected_limit_'+model.toLowerCase()+'_2024_2025_highdm73_lowdm30_xsec_topology_thermal_relic.png';
 const bytes=await fs.readFile(file);
 if(crypto.createHash('sha256').update(bytes).digest('hex')!==previousHashes[i])throw new Error('Previous plot changed');
 slide.images.add({blob:bytes,contentType:'image/png',alt:'2026-09-21 adopted '+model+' result without Nmix',fit:'contain',position:{left:65,top:201,width:525,height:437.5}});
 text(slide,'before-heading','Without Nmix (21 Sep)',65,134,555,38,28,navy,true);
 text(slide,'after-heading','With Nmix (24 Sep)',660,134,555,38,28,navy,true);
 text(slide,'before-model','HighSR73 + lowSR30, adopted syst., '+['203/205','350/354','356/365'][i]+' valid',65,176,570,27,21);
 text(slide,'after-model','HighSR118 + lowSR30, nominal, '+['201/205','339/354','353/365'][i]+' valid',660,176,570,27,21);
 text(slide,'comparison-scope','Binning and nuisance sets differ. New TROTA JES/JER propagation is pending on the right.',65,643,1015,32,21,red);
}

const notes={
 selection:`Definitions from TROTASR/utils/shape_kinematics.py:73-78, trota_candidates.py:20-61,218-272, trota_inference.py:16-17,103-133 and event_selections.py:68-125. Cuts and working points are the current implementation for 2024 and 2025. Mixed/Resolved discriminant is TT/(TT+QCD), not raw TT probability. Mixed builder requires positive candidate pT but adds no top mass or eta window. Boosted/W pT uses the calibrated view. Stored TROTA nominal plots on slides 4-5 use the earlier integrated TROTA momentum convention; the new central CMS-p4 and JES/JER re-inference campaign is separate and ongoing. Region cleaning uses GCR medium photons or DY medium leptons, with AK4-reference deltaR>0.2 and AK8-reference deltaR>0.4. No such extra reference cleaning occurs in the SR/LLCR/QCDCR topology stage.`,
 hierarchy:`Source: TROTASR/utils/event_selections.py:24-56,68-138,181-204 and constituents.py:94-172. This is candidate arbitration, not mutually exclusive event branches. Continue to each stage even if the preceding stage selects no candidate. Mixed and Resolved sort by descending discriminant, then ascending candidateIndex. AK8 footprints are retained AK4 jets matched to its two distinct valid SubJets by deltaR<0.4, or to its axis by deltaR<0.8 when fewer than two valid SubJets exist. SubJet IDs and AK4 IDs belong to separate collections. Geometric matching does not mean AK8 reconstruction from AK4, and is not PF constituent-ID matching. A selected Mixed reserves its explicit AK4 plus all AK4 in its AK8 footprint. Different AK8 jets can conflict through a shared matched AK4. Current code has no Boosted-Boosted or W-W pairwise removal. Other region selections remain in addition to the stated topology count requirements.`,
 sr:`Source figure: ${plotRoot}/predictions/highdm118_sr_2024_2025.png. SHA256 ${assets[0][2]}. Manifest: same directory plot_manifest.json. Combined 2024+2025, 220.66/fb at 13.6 TeV. Nominal prefit predictions, blinded SR. Immutable source highSR162 projects to 21 classes/118 bins. Original category merges [4,5], [7,8], [9,10,11,12], [19,24], [23,28] follow combined-year nominal full-category B<2, not per-bin or per-year B<2. Then displayed classes 2,15,20 (source categories 2,21,26) combine 500-800 and >=800 GeV into >=500 GeV. CR and low-dM binning remain unchanged. Dedicated Mixed SF inputs are not provided and none are fabricated. The original S/sqrt(B) axis stops at 5, so larger values are clipped in that panel.`,
 limits:`Sources: ${plotRoot}/contours/plot_manifest.json and the three embedded original contour PNGs. T2tt 201/205, T2bW 339/354, T2tb 353/365 valid results. Total 893/924 with 31 failed points. Existing plot mass display excludes 20 low-mass points each for T2bW and T2tb, displaying 319 and 333 valid points respectively. Red curves show current nominal expected median and expected bands. Black curves show the pinned SUS-19-010 observed/expected reference in the existing renderer. This is not a controlled with/without TopMixed sensitivity comparison. Do not attribute all differences to TopMixed. These stored-TROTA nominal results do not include the newly requested full CMS-p4 central plus JES/JER re-inference campaign. No old highSR162/121 fit results substitute for SR118.\n`+assets.slice(1).map(a=>a[1]+' SHA256 '+a[2]).join('\n')
};
s2.speakerNotes.textFrame.setText(notes.selection);
s3.speakerNotes.textFrame.setText(notes.hierarchy);
s4.speakerNotes.textFrame.setText(notes.sr);
for(const [i,slide] of limitSlides.entries()){
 slide.speakerNotes.textFrame.setText(`Previous Run-3 baseline: ${previousRoot}, lowdm_cr_nisr_merged_20260921. Source policy: autonomous_allhad/configs/adopted_statistics_2024_2025.json. Nmix is absent. HighSR73 + lowSR30, pooled low-dM CRs, 338 analysis bins, 94 free rate parameters, autoMCStats 10 1 1. Existing adopted systematics and correlated signal xsec lnN. Limit coverage: T2tt 203/205, T2bW 350/354, T2tb 356/365, total 909/924 with 15 failed points. Previous PNG SHA256 ${previousHashes[i]}. These are the actual archived calculations, not a new fit or a SUS-19-010 substitute. Both panels retain the older Run-2 black overlay as context, but the requested comparison is between the two red Run-3 expected contours. The right-hand SR118 nominal calculation has different binning and an incomplete systematic set. This comparison therefore does not isolate the effect of adding TopMixed alone.\n`+notes.limits);
}

const candidatePath=build+'/candidate0924.pptx';
await (await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/draft0924-inspect.ndjson',(await p.inspect({kind:'slide,textbox,table,shape,image',maxChars:65000})).ndjson);
for(let i=0;i<p.slides.items.length;i++){
 const slide=p.slides.items[i];
 const img=await slide.export({format:'png',scale:1.5});
 await fs.writeFile(build+`/draft0924-${i+1}.png`,new Uint8Array(await img.arrayBuffer()));
 await fs.writeFile(build+`/draft0924-${i+1}.json`,await(await slide.export({format:'layout'})).text());
}
if(process.argv.includes('--finalize')){
 const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
 const result=await finalizePresentation({workspaceDir,candidatePath,finalPath:workspaceDir+'/output/stop_meeting_0924_filled.pptx',pythonExecutable:python,
  integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','2'],
  explicitTotalSlideCount:8,requiredNativeTableOwnerSlides:[2],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'reference',families:['Helvetica Neue','Helvetica Neue Light','Helvetica Neue Medium'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},
  verifyArtifactToolImport:true,receiptPath:build+'/validation0924.json'});
 console.log(JSON.stringify(result));
}
console.log('Wrote '+candidatePath);
