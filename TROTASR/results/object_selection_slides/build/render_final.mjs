import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='/Users/taiwoomac/Documents/All Hadronic Stop Analysis/TROTASR/results/object_selection_slides';
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/output/stop_meeting_0924_filled.pptx'));
for(let i=0;i<p.slides.items.length;i++){
 const image=await p.slides.items[i].export({format:'png',scale:1.5});
 await fs.writeFile(root+`/build/final0924-${i+1}.png`,new Uint8Array(await image.arrayBuffer()));
}
console.log('Rendered '+p.slides.items.length+' final slides');
