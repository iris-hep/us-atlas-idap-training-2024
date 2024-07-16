# %%
import awkward as ak
import dask
import hist
from hist.dask import Hist
import json
from coffea import processor
from coffea.nanoevents import BaseSchema, NanoAODSchema 
from coffea.dataset_tools import apply_to_dataset, apply_to_fileset, preprocess, rucio_utils
from coffea.dataset_tools import max_chunks, max_files, slice_chunks, slice_files
import corrections
import matplotlib.pyplot as plt


class MyZPeak(processor.ProcessorABC):
    def process(self, events):
        dataset = events.metadata['dataset']
        isRealData = "genWeight" not in events.fields
        sumw = 0. if isRealData else ak.sum(events.genWeight, axis=0)
        cutflow = {"start": ak.num(events, axis=0)}
        
        if isRealData:
            events = events[
                corrections.lumimask(events.run, events.luminosityBlock)
            ]
            cutflow["lumimask"] = ak.num(events, axis=0)
    
        events["goodmuons"] = events.Muon[
            (events.Muon.pt >= 20.)
            & events.Muon.tightId
        ]

        events = events[
            (ak.num(events.goodmuons) == 2)
            & (ak.sum(events.goodmuons.charge, axis=1) == 0)
        ]
        cutflow["ossf"] = ak.num(events, axis=0)
        
        # add first and second muon p4 in every event together
        events["zcand"] = events.goodmuons[:, 0] + events.goodmuons[:, 1]

        # require trigger
        events = events[
            # https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLT2018
            events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8
        ]
        weight = 1 * ak.ones_like(events.event) if isRealData else events.genWeight
        cutflow["trigger"] = ak.num(events, axis=0)

        return {
                "entries": ak.num(events, axis=0),
                "sumw": sumw,
                "cutflow": cutflow,
                "mass": (
                    Hist.new
                    .Reg(120, 0., 120., label="$m_{\mu\mu}$ [GeV]")
                    .Weight()
                    .fill(events.zcand.mass, weight=weight)
                )
            }

    def postprocess(self, accumulator):
        return accumulator


# %%
from dask.distributed import Client

client = Client("tls://localhost:8786")
client

# %%
import shutil
shutil.make_archive("corrections", "zip", base_dir="corrections")

# %%
client.upload_file("corrections.zip")

# %%
with open("fileset.json", "rt") as file:
    initial_fileset = json.load(file)

# %%
preprocessed_available, preprocessed_total = preprocess(
        initial_fileset,
        step_size=100_000,
        align_clusters=None,
        skip_bad_files=True,
        recalculate_steps=False,
        files_per_batch=1,
        file_exceptions=(OSError,),
        save_form=True,
        uproot_options={},
        step_size_safety_factor=0.5,
    )
    #with gzip.open(f"{output_file}_available.json.gz", "wt") as file:
    #    print(f"Saved available fileset chunks to {output_file}_available.json.gz")

# %%
test_preprocessed_files = max_files(preprocessed_available, 5)
test_preprocessed = max_chunks(test_preprocessed_files, 7)

# %%
small_tg, small_rep = apply_to_fileset(data_manipulation=MyZPeak(),
                            fileset=test_preprocessed,
                            schemaclass=NanoAODSchema,
                            uproot_options={"allow_read_errors_with_report": (OSError, KeyError)},
                           )

# %%
small_result, small_report = dask.compute(small_tg, small_rep)

# %%
small_result


# %%
def total_data(events):
    isRealData = "genWeight" not in events.fields
    if isRealData:
        return ak.num(events, axis=0)
    else:
        return -1


# %%
dfd, _ = apply_to_fileset(data_manipulation=total_data,
                                     fileset=preprocessed_available,
                                     schemaclass=NanoAODSchema,
                                     uproot_options={"allow_read_errors_with_report": (OSError, KeyError)},
                                    )

# %%
data_fraction_num = small_result["DoubleMuon2018A"]["cutflow"]["start"]
data_fraction_den = dfd["DoubleMuon2018A"].compute()
data_fraction = data_fraction_num / data_fraction_den
print(data_fraction)

# %%
data = small_result["DoubleMuon2018A"]["mass"]

lumi = 14.0
#xsweight = lumi * 1e3 * 6225.42 * data_fraction / small_result["ZJets2018"]["cutflow"]["start"]
xsweight = lumi * 1e3 * 6225.42 * data_fraction / small_result["ZJets2018"]["sumw"]
sim = small_result["ZJets2018"]["mass"] * xsweight

# %%
fig, ax = plt.subplots()
sim.plot(ax=ax, histtype="fill", label="Z+jets")
data.plot(ax=ax, histtype="errorbar", color="k", label="Data")
ax.set_xlim(60, 120)
ax.legend()
