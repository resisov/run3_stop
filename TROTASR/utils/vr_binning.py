"""Define the 40 high-dM and 30 low-dM validation bins."""

N_LOW_BINS = 30
N_HIGH_BINS = 40
N_VR_BINS = N_LOW_BINS + N_HIGH_BINS
HIGH_VR_DEFINITION = "run2_style_tagged_mtb_v1"
HIGH_VR_MET_SPLIT = 400.
# Union of the approved normalization recoil boundaries and the VR edges.
# As in the display, MET overflow is included in the last bin.
HIGH_VR_RATE_EDGES = (250., 300., 350., HIGH_VR_MET_SPLIT, 450., 500., 800.)
HIGH_TAG_AXES = ("Nbst", "Nmix", "Nres", "Nw")


def _tag_predicates(tag_index):
    if tag_index == len(HIGH_TAG_AXES):
        return {"Ntag": {"ge": 2}}
    return {axis: {"eq": int(i == tag_index)} for i, axis in enumerate(HIGH_TAG_AXES)}


HIGH_VR_CATEGORIES = [
    dict(predicates={"Nb": nb, **_tag_predicates(i)}, tag_class=tag,
         mtb_min=low, mtb_max=high, edges=[250., HIGH_VR_MET_SPLIT, 1500.])
    for low, high in ((175., None), (0., 175.))
    for nb in ({"eq": 1}, {"ge": 2})
    for i, tag in enumerate(("boosted", "mixed", "resolved", "w", "multi"))
]
