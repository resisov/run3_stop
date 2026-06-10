import json, gzip
meta="metadata/KNU_2024_v4.json.gz"
with gzip.open(meta, "rt") as f:
    d=json.load(f)
keys=sorted(d.keys())
with open("datasets/datasets_2024.txt","w") as out:
    for k in keys:
        #if "Muon" in k:
            #out.write(k+"\n")
        out.write(k+"\n")
print("wrote datasets_2024.txt with", len(keys), "datasets")

## only mc
with gzip.open(meta, "rt") as f:
    d=json.load(f)
keys=sorted(d.keys())
with open("datasets/datasets_2024_onlymc.txt","w") as out:
    for k in keys:
        if "JetMET" in k or "Muon" in k or "EGamma" in k:
            pass
        else:
            out.write(k+"\n")
print("wrote datasets_2024_onlymc.txt with", len(keys), "datasets")