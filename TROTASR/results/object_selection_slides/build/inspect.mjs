import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const source = '/Users/taiwoomac/Desktop/Meeting/stop_meeting_0924.pptx';
const dir = '/Users/taiwoomac/Documents/All Hadronic Stop Analysis/TROTASR/results/object_selection_slides/build';
const p = await PresentationFile.importPptx(await FileBlob.load(source));
const s = await p.inspect({kind:'slide,textbox,shape,image,table,layout',maxChars:40000});
await fs.writeFile(dir+'/source0924-inspect.ndjson',s.ndjson);
console.log(s.ndjson);
console.log('MASTERS', p.masters.items.map(x=>({id:x.id,name:x.name})));
console.log('LAYOUTS', p.layouts.items.map(x=>({id:x.id,name:x.name,placeholders:x.placeholders.summary()})));
for(let i=0;i<p.slides.items.length;i++) {
 const slide=p.slides.items[i];
 const png=await slide.export({format:'png',scale:1});
 await fs.writeFile(dir+`/source0924-${i+1}.png`,new Uint8Array(await png.arrayBuffer()));
 const layout=await slide.export({format:'layout'});
 await fs.writeFile(dir+`/source0924-${i+1}.json`,await layout.text());
}
