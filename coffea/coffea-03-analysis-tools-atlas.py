# %% [markdown]
# # Analysis tools
#
# Now that we [know how to access data with `NanoEvents`](https://github.com/iris-hep/us-atlas-idap-training-2024/tree/main/PHYSLITE), let's go through some useful columnar analysis tools and idioms for building collections of results, namely, the eventual output of a `coffea` callable (or processor).
# The most familiar type of output may be the histogram (one type of accumulator).
#
# We'll use a small sample file to demonstrate the utilities, although it won't be very interesting to analyze.

# %%
import numpy as np
import uproot
import awkward as ak
import dask
from hist.dask import Hist
from coffea.nanoevents import NanoEventsFactory, PHYSLITESchema

# %%
xcache_caching_server = "root://xcache.af.uchicago.edu:1094//"
file_path = "root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000001.pool.root.1"

file_uri = f"{xcache_caching_server}{file_path}"

# %% [markdown]
# You can download those files at your local machine
#
# ```console
# xrdcp --allow-http --force root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000001.pool.root.1 example.root
# ```
#
# or you can stream them directly. For this demo I will use the [XCache](http://slateci.io/XCache/) service of our compute facility to speed things up.

# %%
# # ! mkdir -p data
# # ! xrdcp --allow-http "{file_path}" data/example.root

# %%
from pathlib import Path
_local_path = Path().cwd() / "data" / "example.root"
if _local_path.exists():
    file_name = _local_path
else:
    file_name = file_uri


# %% [markdown]
# There is lots of information in the files, but for this example we're only going to look at a few fields:
# * Event information
# * Electrons
# * Muons
# * Jets
# * B-tagging

# %%
def filter_name(name):
    """
    Load only the properties/variables needed.
    """
    return name in (
        "EventInfoAuxDyn.mcEventWeights",
        #
        "AnalysisElectronsAuxDyn.pt",
        "AnalysisElectronsAuxDyn.eta",
        "AnalysisElectronsAuxDyn.phi",
        "AnalysisElectronsAuxDyn.m",
        "AnalysisElectronsAuxDyn.DFCommonElectronsLHLoose",
        "AnalysisElectronsAuxDyn.charge",
        #
        "AnalysisMuonsAuxDyn.pt",
        "AnalysisMuonsAuxDyn.eta",
        "AnalysisMuonsAuxDyn.phi",
        "AnalysisMuonsAuxDyn.m",
        "AnalysisMuonsAuxDyn.charge",
        "AnalysisMuonsAuxDyn.quality",
        #
        "AnalysisJetsAuxDyn.pt",
        "AnalysisJetsAuxDyn.eta",
        "AnalysisJetsAuxDyn.phi",
        "AnalysisJetsAuxDyn.m",
        #
        "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb",
        "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pc",
        "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pu",
    )


# %%
import warnings
warnings.filterwarnings("ignore")

events = NanoEventsFactory.from_root(
    {file_name: "CollectionTree"},
    schemaclass=PHYSLITESchema,
    uproot_options=dict(filter_name=filter_name),
    delayed=True,
).events()

# %%
# %%time

events = events.compute()

# %% [markdown]
# and we get the fields we requested

# %%
events.fields

# %% [markdown]
# and the subfields that were requested for each field

# %%
for _field in events.fields:
    print(f"* {_field}: {events[_field].fields}")

# %% [markdown]
# To generate some mock systematics, we'll use one of the scale factors from the applying_corrections notebook (note you will have to at least execute the cell that downloads test data in that notebook for this to work)

# %%
from coffea.lookup_tools import extractor

ext = extractor()
ext.add_weight_sets(["* * data/testSF2d.histo.root"])
ext.finalize()
evaluator = ext.make_evaluator()
evaluator.keys()

# %% [markdown]
# ## Weights
#
# This is a container for event weights and associated systematic shifts, which helps track the product of the weights (i.e. the total event weight to be used for filling histograms) as well as systematic variations to that product. Here we demo its use by constructing an event weight consisting of the generator weight, the $\alpha_s$ uncertainty variation, and the electron ID scale factor with its associated systematic.

# %%
from coffea.analysis_tools import Weights

if delayed:
    weights = Weights(None)
else:
    weights = Weights(len(events))

weights.add("genWeight", events.genWeight)

weights.add(
    "alphaS",
    # in NanoAOD, the generator weights are already stored with respect to nominal
    weight=ak.ones_like(events.run),
    # 31 => alphas(MZ)=0.1165 central value; 32 => alphas(MZ)=0.1195
    # per https://lhapdfsets.web.cern.ch/current/PDF4LHC15_nnlo_30_pdfas/PDF4LHC15_nnlo_30_pdfas.info
    # which was found by looking up the LHA ID in events.LHEPdfWeight.__doc__
    weightUp=events.LHEPdfWeight[:, 32],
    weightDown=events.LHEPdfWeight[:, 31],
)

eleSF = evaluator["scalefactors_Tight_Electron"](events.Electrons.eta, events.Electrons.pt)
eleSFerror = evaluator["scalefactors_Tight_Electron_error"](events.Electrons.eta, events.Electrons.pt)
weights.add(
    "eleSF",
    # the event weight is the product of the per-electron weights
    # note, in a real analysis we would first have to select electrons of interest
    weight=ak.prod(eleSF, axis=1),
    weightUp=ak.prod(eleSF + eleSFerror, axis=1),
)

# %% [markdown]
# A [WeightStatistics](https://coffeateam.github.io/coffea/api/coffea.analysis_tools.WeightStatistics.html) object tracks the smallest and largest weights seen per type, as well as some other summary statistics. It is kept internally and can be accessed via `weights.weightStatistics`. This object is addable, so it can be used in an accumulator.

# %%
weights.weightStatistics

# %% [markdown]
# Then the total event weight is available via

# %%
weights.weight()

# %% [markdown]
# And the total event weight with a given variation is available via

# %%
weights.weight("eleSFUp")

# %% [markdown]
# all variations tracked by the `weights` object are available via

# %%
weights.variations

# %% [markdown]
# ## `PackedSelection`
#
# This class can store several boolean arrays in a memory-efficient mannner and evaluate arbitrary combinations of boolean requirements in an CPU-efficient way. Supported inputs include 1D `numpy` or `awkward` arrays. This makes it a good tool to form analysis signal and control regions, and to implement cutflow or "N-1" plots.
#
# Below we create a packed selection with some typical selections for a $Z$+jets study, to be used later to form same-sign and opposite-sign $ee$ and $\mu\mu$ event categories/regions.

# %%
from coffea.analysis_tools import PackedSelection

selection = PackedSelection()

selection.add("twoElectrons", ak.num(events.Electrons, axis=1) == 2)
selection.add("eleOppSign", ak.sum(events.Electrons.charge, axis=1) == 0)
selection.add("noElectrons", ak.num(events.Electrons, axis=1) == 0)

selection.add("twoMuons", ak.num(events.Muons, axis=1) == 2)
selection.add("muOppSign", ak.sum(events.Muons.charge, axis=1) == 0)
selection.add("noMuons", ak.num(events.Muons, axis=1) == 0)


selection.add(
    "leadPt20",
    # assuming one of `twoElectrons` or `twoMuons` is imposed, this implies at least one is above threshold
    ak.any(events.Electrons.pt >= 20.0, axis=1) | ak.any(events.Muons.pt >= 20.0, axis=1)
)

print(selection.names)

# %% [markdown]
# To evaluate a boolean mask (e.g. to filter events) we can use the `selection.all(*names)` function, which will compute the AND of all listed boolean selections

# %%
selection.all("twoElectrons", "noMuons", "leadPt20")

# %% [markdown]
# We can also be more specific and require that a specific set of selections have a given value (with the unspecified ones allowed to be either `True` or `False`) using `selection.require`

# %%
selection.require(twoElectrons=True, noMuons=True, eleOppSign=False)

# %% [markdown]
# Using the Python syntax for passing an arguments variable, we can easily implement a "N-1" style selection

# %%
all_cuts = {"twoElectrons", "noMuons", "leadPt20"}
results = {}
for cut in all_cuts:
    n_events = ak.sum(selection.all(*(all_cuts - {cut})), axis=0)
    results[cut] = n_events
    
results["None"] = ak.sum(selection.all(*all_cuts), axis=0)

cut_results, *_ = dask.compute(results)

for cut, n_events in cut_results.items():
    print(f"Events passing all cuts, ignoring '{cut}': {n_events}")

# %%
from coffea import processor

# %% [markdown]
# ### Bringing it together
#
# Let's build a callable function that books a few results, per dataset:
#  - the sum of weights for the events processed, to use for later luminosity-normalizing the yields;
#  - a histogram of the dilepton invariant mass, with category axes for various selection regions of interest and  systematics; and
#  - the weight statistics, for debugging purposes
# And, additionally, we'll switch to delayed mode and compute the results with an explicit call through dask's interface
#  

# %%
file_name = "https://raw.githubusercontent.com/CoffeaTeam/coffea/e06c4b84d0a641ab569ae7c16fecc39fe74c9743/tests/samples/nano_dy.root"
devents = NanoEventsFactory.from_root(
    {file_name: "Events"},
    schemaclass=NanoAODSchema,
    metadata={"dataset": "DYJets"},
    delayed=True,
).events()


# %%
def results_taskgraph(events):

    regions = {
        "ee": {"twoElectrons": True, "noMuons": True, "leadPt20": True, "eleOppSign": True},
        "eeSS": {"twoElectrons": True, "noMuons": True, "leadPt20": True, "eleOppSign": False},
        "mm": {"twoMuons": True, "noElectrons": True, "leadPt20": True, "muOppSign": True},
        "mmSS": {"twoMuons": True, "noElectrons": True, "leadPt20": True, "muOppSign": False},
    }

    mass_hist = (
        Hist.new
        .StrCat(regions.keys(), name="region")
        .StrCat(["nominal"] + list(weights.variations), name="systematic")
        .Reg(60, 60, 120, name="mass", label="$m_{ll}$ [GeV]")
        .Weight()
    )

    for region, cuts in regions.items():
        good_event = selection.require(**cuts)

        if region.startswith("ee"):
            leptons = events.Electrons[good_event]
        elif region.startswith("mm"):
            leptons = events.Muons[good_event]
        lep1 = leptons[:, 0]
        lep2 = leptons[:, 1]
        mass = (lep1 + lep2).mass

        mass_hist.fill(
            region=region,
            systematic="nominal",
            mass=mass,
            weight=weights.weight()[good_event],
        )
        for syst in weights.variations:
            mass_hist.fill(
                region=region,
                systematic=syst,
                mass=mass,
                weight=weights.weight(syst)[good_event],
            )

    out = {
        events.metadata["dataset"]: {
            "sumw": ak.sum(events.genWeight, axis=0),
            "mass": mass_hist,
            "weightStats": weights.weightStatistics,
        }
    }
    return out


# %%
out = results_taskgraph(devents)

# %%
out

# %%
c_out, *_ = dask.compute(out)

# %%
c_out

# %% [markdown]
# The mass histogram itself is not very interesting with only 40 input events, however

# %%
c_out["DYJets"]["mass"][sum, "nominal", :]
