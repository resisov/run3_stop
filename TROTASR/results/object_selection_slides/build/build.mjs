import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const source='/Users/taiwoomac/Desktop/Meeting/stop_meeting_0917.pptx';
const workspaceDir='/Users/taiwoomac/Documents/All Hadronic Stop Analysis/TROTASR/results/object_selection_slides';
const build=path.join(workspaceDir,'build');
const skill='/Users/taiwoomac/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const python='/Users/taiwoomac/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const cover=p.resolve('sl/y90nupkv'), s2=p.resolve('sl/hwbqtkby'), s3=p.resolve('sl/ofy9wn61'), end=p.resolve('sl/jyx0ra1s');
const blue='#4472C4', navy='#002060', red='#C00000', black='#000000', white='#FFFFFF';
const font='Helvetica Neue Light', medium='Helvetica Neue Medium';

function text(slide,name,value,x,y,w,h,size=28,color=black,weight=false) {
 const s=slide.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;
 s.text.style={typeface:weight?medium:font,fontSize:size,color,bold:weight,autoFit:'none',wrap:'square',verticalAlignment:'middle',insets:{left:0,right:0,top:0,bottom:0}};
 return s;
}
function box(slide,name,value,x,y,w,h,size=28,fill=blue) {
 const s=slide.shapes.add({name,geometry:'roundRect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:black,width:1},borderRadius:10});
 s.text=value;
 s.text.style={typeface:font,fontSize:size,color:white,alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:{left:12,right:12,top:8,bottom:8}};
 return s;
}
function connect(slide,a,b,from='bottom',to='top') {
 return slide.shapes.connect(a,b,{kind:'elbow',fromSide:from,toSide:to,line:{fill:navy,width:2},head:{type:'triangle',width:'sm',length:'sm'}});
}
function clean(slide,titleId,footerId,lineId) {
 for(const s of [...slide.shapes.items]) if(![titleId,footerId,lineId].includes(s.id)) s.delete();
}
clean(s2,'9072xkry','cnupgny5','mtwrmxg7');
clean(s3,'cza94vmx','ja54na9g','943mhgre');
const s4=s2.duplicate(); s4.moveTo(3);
const s5=s2.duplicate(); s5.moveTo(4);
end.moveTo(5);
function frame(slide,title,page) {
 const titleShape=slide.shapes.items.find(s=>s.isPlaceholder && s.placeholderType==='title');
 if(!titleShape) throw new Error('Missing imported title');
 titleShape.text=title;
 titleShape.text.style={typeface:medium,fontSize:58.6667,color:black,verticalAlignment:'middle',alignment:'left',autoFit:'none',insets:{left:9.6,right:9.6,top:4.8,bottom:4.8}};
 const footer=slide.shapes.items.find(s=>String(s.text).includes('Weekly Stop Meeting'));
 if(!footer) throw new Error('Missing imported footer');
 footer.text=`${page}     2026. 9. 24.     Taiwoo KIM – Weekly Stop Meeting`;
 footer.text.style={typeface:font,fontSize:18.6667,color:black,autoFit:'none',verticalAlignment:'middle',insets:{left:9.6,right:9.6,top:0,bottom:0}};
}
frame(s2,'Top/W object selection',1);
frame(s3,'Top/W candidate priority',2);
frame(s4,'AK8–AK4 overlap removal',3);
frame(s5,'Mixed and W: overlap examples',4);

const title=p.resolve('sh/dgzetcfa');
title.text='Top/W object selection\nOverlap removal';
title.text.style={typeface:'Helvetica Neue',fontSize:58.6667,bold:true,color:black,autoFit:'none',verticalAlignment:'middle',insets:{left:9.6,right:9.6,top:4.8,bottom:4.8}};
p.resolve('sh/k3yl0zql').text='Weekly All-hadronic Stop Meeting (NPS-26-012)';
p.resolve('sh/zadcv2p8').text.replace('17','24');

// Native selection table. The pT values refer to the corrected current view.
const vals=[
 ['Object','Inputs','pT and η cuts','Mass / candidate η','Tag requirement'],
 ['Boosted top\nNbst','1 AK8','pT(AK8) > 400 GeV\n|η(AK8)| < 2','mSD > 105 GeV','Top score > 0.5078'],
 ['Mixed top\nNmix','2 or 3 AK4\n+ 1 AK8','AK4: pT > 25 GeV, |η| < 2.5\nAK8: pT > 200 GeV, |η| < 2.5','No extra top\nmass or η cut','TT / (TT + QCD)\n≥ 0.9027690887451172'],
 ['Resolved top\nNres','3 distinct AK4','Each AK4:\npT > 25 GeV, |η| < 2.5','100 ≤ m(3 AK4)\n≤ 250 GeV\n|η(3 AK4)| < 2','TT / (TT + QCD)\n≥ 0.9433798789978027'],
 ['W\nNw','1 AK8','pT(AK8) > 200 GeV\n|η(AK8)| < 2','60 < mSD\n< 105 GeV','W score > 0.9385']
];
const table=s2.tables.add({rows:5,columns:5,left:65,top:154,width:1150,height:397,columnWidths:[170,150,340,240,250],values:vals});
table.styleOptions={headerRow:true,bandedRows:false};
table.borders.assign({fill:'#A5A5A5',width:.7,style:'solid'});
for(let r=0;r<5;r++) for(let c=0;c<5;c++) {
 const cell=table.getCell(r,c); cell.fill=r===0?blue:white;
 cell.text.style={typeface:r===0?medium:font,fontSize:r===0?23:21.5,color:r===0?white:black,autoFit:'none',wrap:'square',verticalAlignment:'middle',insets:{left:10,right:10,top:8,bottom:8}};
}
[48,80,97,97,75].forEach((h,i)=>{table.rows[i].height=h;});
text(s2,'common-selection','Jet ID and region-dependent photon/lepton cleaning apply before candidate selection.',65,573,1150,32,25);
text(s2,'mass-definition','mSD: AK8 soft-drop mass. m(3 AK4): invariant mass of the three AK4 jets.',65,613,1020,27,22);

const ys=[145,253,361,469];
const bnodes=[
 box(s3,'boosted-stage','1  Boosted top\nNbst',85,ys[0],260,83,28),
 box(s3,'mixed-stage','2  Mixed top\nNmix',85,ys[1],260,83,28),
 box(s3,'resolved-stage','3  Resolved top\nNres',85,ys[2],260,83,28),
 box(s3,'w-stage','4  W\nNw',85,ys[3],260,83,28,navy)
];
for(let i=0;i<3;i++) connect(s3,bnodes[i],bnodes[i+1]);
[
 'Keep all passing AK8 jets.\nVeto AK4 overlaps in later top candidates.',
 'Highest Mixed score first. Reject a shared AK8\nor AK4 overlap with selected Boosted/Mixed.',
 'Highest Resolved score first. Reject any AK4 overlap\nwith selected Boosted/Mixed/Resolved.',
 'Reject a shared AK8 or AK4 overlap with any\nselected top. Keep independent W candidates.'
].forEach((v,i)=>text(s3,'priority-'+i,v,392,ys[i]+2,820,80,27));
text(s3,'candidate-not-event','Reject the overlapping candidate only. Continue even if a stage selects zero candidates.',85,577,1120,34,24);
text(s3,'category-order','SR category order: Nb, Nbst, Nmix, Nres, Nw. This veto does not reduce Nb.',85,616,980,29,23);

text(s4,'collections','AK4, AK8 and SubJet are separate collections.',85,142,1110,42,32);
const q=box(s4,'subjet-question','Two distinct valid SubJets\nfor this AK8?',450,207,380,83,28,navy);
const yes=box(s4,'subjet-condition','ΔR(AK4, SubJet 1) < 0.4\nor ΔR(AK4, SubJet 2) < 0.4',80,350,530,102,28);
const no=box(s4,'axis-condition','ΔR(AK4, AK8 axis) < 0.8',700,350,500,102,28);
connect(s4,q,yes); connect(s4,q,no);
text(s4,'yes-label','Yes',298,299,90,37,25);
text(s4,'no-label','No',910,299,90,37,25);
const matched=box(s4,'matched-ak4','Condition true: flag this AK4\nas overlapping this AK8',380,509,550,79,28,navy);
connect(s4,yes,matched); connect(s4,no,matched);
text(s4,'false-case','Condition false: no overlap flag for this AK4–AK8 pair.',85,605,980,30,24);
text(s4,'proxy-caveat','Jet-axis matching. No PF constituent-ID comparison.',85,643,960,25,21);

text(s5,'example-assumption','Example: one selected Mixed top; test one W candidate.',85,142,1110,42,31);
const examples=[
 ['Same AK8','Mixed AK8 index = 2\nW AK8 index = 2','Reject W\nNmix = 1, Nw = 0'],
 ['AK4 overlap','Mixed includes AK4 #5\nΔR(AK4 #5, W SubJet 1) = 0.20','Reject W\nNmix = 1, Nw = 0'],
 ['Independent W','Different AK8\nNo shared or ΔR-matched AK4','Keep W\nNmix = 1, Nw = 1']
];
examples.forEach((row,i)=>{
 const y=215+i*119;
 text(s5,'case-'+i,row[0],85,y,245,90,29,black,true);
 text(s5,'case-condition-'+i,row[1],342,y,555,90,26);
 box(s5,'case-result-'+i,row[2],943,y+3,269,84,25,i===2?blue:navy);
});
text(s5,'subjet-assumption','The ΔR = 0.20 example assumes two valid SubJets for the W AK8.',85,564,1120,29,23);
text(s5,'pairwise-caveat','Current code: no Boosted–Boosted or W–W pairwise overlap veto.',85,606,1080,30,24,red);
text(s5,'ak8-caveat','No direct ΔR(AK8, AK8) cut. Different AK8s can share a matched AK4.',85,643,970,25,21);

const base='TROTASR/utils/';
const noteSources={
 cover:'Current executable implementation, 2026-09-24. Source deck provides visual template only; historical datacard and timing values are not reused.',
 selection:`Sources: ${base}shape_kinematics.py:73-78; ${base}trota_candidates.py:20-61,218-272; ${base}trota_inference.py:16-17,103-133; ${base}event_selections.py:68-125. All pT values are those supplied by the current calibrated view. Mixed/Resolved discriminant is TT/(TT+QCD), not raw TT probability. Mixed has no extra analysis candidate-mass or candidate-eta window. The builder requires positive candidate pT. Region cleaning: GCR medium photon; DY2E medium electron; DY2M medium muon. Require AK4-reference ΔR>0.2 and AK8-reference ΔR>0.4. This topology stage adds no such reference cleaning in SR/LLCR/QCDCR.`,
 priority:`Source: ${base}event_selections.py:24-56,68-138,181-204. Greedy candidate arbitration, not event-exclusive categorization. Rank Mixed and Resolved separately by descending score, then ascending candidateIndex. Compare original source jet identities. Reserve a selected Mixed candidate's explicit AK4 plus all AK4 matched to its AK8. Reject only the conflicting candidate. For each JES/JER endpoint rerun Resolved and Mixed inference, working-point selection, cleaning and arbitration. The multiplicity requirement is sum>=1 for high-dM, sum==0 for low-dM in addition to the existing region cuts. Nb remains independent.`,
 overlap:`Sources: ${base}constituents.py:94-172; ${base}event_selections.py:93-101. Two valid distinct subjet indices use the OR of two strict ΔR<0.4 cones. Fewer than two use the AK8-axis strict ΔR<0.8 cone. Match retained AK4 source IDs, not SubJet IDs. This is geometric overlap tagging, not AK8 reconstruction from AK4 and not PF constituent identity. Two distinct AK8s can conflict through a common matched AK4. No direct AK8-AK8 angular cut is imposed. Mixed internal four-vector construction is a separate step in trota_candidates.py:260-272: test AK4-AK8 ΔR<0.8; use AK8 plus nonoverlapping AK4, except the 3-AK4/no-overlap branch uses only the 3-AK4 sum. Arbitration still reserves all recorded Mixed input identities.`,
 examples:`Sources: ${base}event_selections.py:24-56; tests/test_arbitration.py. These are constructed logical examples, not measured event counts. Assume exactly one selected Mixed, no other selected top and one eligible W. The ΔR=0.20 case assumes two valid W subjets. An independent W must also avoid the Mixed AK8's matched AK4. Current code keeps distinct passing Boosted AK8s without Boosted-Boosted arbitration. Selected W candidates do not update the veto lists, so W-W arbitration is absent. Preserve these limitations rather than claiming universal disjointness.`
};
cover.speakerNotes.textFrame.setText(noteSources.cover);
s2.speakerNotes.textFrame.setText(noteSources.selection);
s3.speakerNotes.textFrame.setText(noteSources.priority);
s4.speakerNotes.textFrame.setText(noteSources.overlap);
s5.speakerNotes.textFrame.setText(noteSources.examples);

console.log('ORDER',p.slides.items.map(s=>s.id));
await fs.writeFile(build+'/draft-inspect.ndjson',(await p.inspect({kind:'slide,textbox,table,shape',maxChars:60000})).ndjson);
for(let i=0;i<p.slides.items.length;i++) {
 const slide=p.slides.items[i];
 const image=await slide.export({format:'png',scale:1.5});
 await fs.writeFile(build+`/slide-${i+1}.png`,new Uint8Array(await image.arrayBuffer()));
 await fs.writeFile(build+`/slide-${i+1}.json`,await (await slide.export({format:'layout'})).text());
}
const candidatePath=build+'/candidate.pptx';
await (await PresentationFile.exportPptx(p)).save(candidatePath);
if(process.argv.includes('--finalize')) {
 const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
 const finalPath=workspaceDir+'/output/stop_meeting_topw_selection_20260924.pptx';
 const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:python,
  integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
  layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','2'],
  explicitTotalSlideCount:6,requiredNativeTableOwnerSlides:[2],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'reference',families:['Helvetica Neue','Helvetica Neue Light','Helvetica Neue Medium'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},
  verifyArtifactToolImport:true,receiptPath:build+'/validation.json'});
 console.log(JSON.stringify(result));
}
