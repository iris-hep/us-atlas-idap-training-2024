# %% [markdown]
# # Analysis tools
#
# Now that we [know how to access data with `NanoEvents`](https://github.com/iris-hep/us-atlas-idap-training-2024/tree/main/PHYSLITE), let's go through some useful columnar analysis tools and idioms for building collections of results, namely, the eventual output of a `coffea` callable (or processor).
# The most familiar type of output may be the histogram (one type of accumulator).
#
# We'll just look at single files for the time being to keep things simple.

# %% [markdown]
# ## Rapid review of what we've already seen

# %%
from pathlib import Path

from matplotlib import pyplot as plt
import awkward as ak
import dask
from hist.dask import Hist
from coffea.nanoevents import NanoEventsFactory, PHYSLITESchema
from coffea.analysis_tools import PackedSelection
import mplhep

PHYSLITESchema.warn_missing_crossrefs = False

# %%
from importlib.metadata import version

for package in ["numpy", "awkward", "uproot", "coffea", "dask"]:
    print(f"# {package}: v{version(package)}")

# %%
xcache_caching_server = "root://xcache.af.uchicago.edu:1094//"
open_data_storage = "root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/"
file_path = "DAOD_PHYSLITE.38191712._000001.pool.root.1"

file_uri = f"{xcache_caching_server}{open_data_storage}{file_path}"

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
# # ! xrdcp --allow-http "{open_data_storage}{file_path}" data/example.root

# %%
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
        "EventInfoAuxDyn.mcChannelNumber",
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


# %% [markdown]
# There will be some warnings from `coffea`, but in this case they can be ignored.

# %%
events = NanoEventsFactory.from_root(
    {file_name: "CollectionTree"},
    schemaclass=PHYSLITESchema,
    uproot_options=dict(filter_name=filter_name),
    delayed=True,
).events()

# %% [markdown]
# The `events` haven't yet actually been evaluated (it is still a Dask task graph) but let's go ahead and evaluate them now so we can inspect the events more.

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
# ## `PackedSelection`
#
# This class can store several boolean arrays in a memory-efficient mannner and evaluate arbitrary combinations of boolean requirements in an CPU-efficient way. Supported inputs include 1D `numpy` or `awkward` arrays. This makes it a good tool to form analysis signal and control regions, and to implement cutflow or "N-1" plots.
#
# Below we create a packed selection with some typical selections for a $Z$+jets study, to be used later to form same-sign and opposite-sign $ee$ and $\mu\mu$ event categories/regions.

# %% [markdown]
# We'll use [ATLAS open data electroweak boson simulation](https://opendata.cern.ch/record/80010) for this ( DOI:[10.7483/OPENDATA.ATLAS.K5SU.X65Y](http://doi.org/10.7483/OPENDATA.ATLAS.K5SU.X65Y))

# %%
file_path = "DAOD_PHYSLITE.37621317._000001.pool.root.1"

file_uri = f"{xcache_caching_server}{open_data_storage}{file_path}"

# %%
# # ! mkdir -p data
# # ! xrdcp --allow-http "{open_data_storage}{file_path}" data/Z_jets.root

# %%
_local_path = Path().cwd() / "data" / "Z_jets.root"
if _local_path.exists():
    file_name = _local_path
else:
    file_name = file_uri

# %%
events = NanoEventsFactory.from_root(
    {file_name: "CollectionTree"},
    schemaclass=PHYSLITESchema,
    uproot_options=dict(filter_name=filter_name),
    delayed=True,
).events()

# %%
events = events.compute()

# %%
selection = PackedSelection()

selection.add("two_electrons", ak.num(events.Electrons, axis=1) == 2)
selection.add("electrons_opposite_sign", ak.sum(events.Electrons.charge, axis=1) == 0)
selection.add("no_electrons", ak.num(events.Electrons, axis=1) == 0)

selection.add("two_muons", ak.num(events.Muons, axis=1) == 2)
selection.add("muons_opposite_sign", ak.sum(events.Muons.charge, axis=1) == 0)
selection.add("no_muons", ak.num(events.Muons, axis=1) == 0)


selection.add(
    "lead_pt_20",
    # assuming one of `two_electrons` or `two_muons` is imposed, this implies at least one is above threshold
    ak.any(events.Electrons.pt >= 20.0, axis=1)
    | ak.any(events.Muons.pt >= 20.0, axis=1),
)

print(selection.names)

# %% [markdown]
# To evaluate a boolean mask (e.g. to filter events) we can use the `selection.all(*names)` function, which will compute the AND of all listed boolean selections

# %%
selection.all("two_electrons", "no_muons", "lead_pt_20")

# %% [markdown]
# We can also be more specific and require that a specific set of selections have a given value (with the unspecified ones allowed to be either `True` or `False`) using `selection.require`

# %%
selection.require(two_electrons=True, no_muons=True, electrons_opposite_sign=False)

# %% [markdown]
# Using the Python syntax for passing an arguments variable, we can easily implement a "N-1" style selection

# %%
all_cuts = {"two_electrons", "no_muons", "lead_pt_20"}
results = {}
for cut in all_cuts:
    n_events = ak.sum(selection.all(*(all_cuts - {cut})), axis=0)
    results[cut] = n_events

results["None"] = ak.sum(selection.all(*all_cuts), axis=0)

cut_results, *_ = dask.compute(results)

for cut, n_events in cut_results.items():
    print(f"Events passing all cuts, ignoring '{cut}': {n_events}")

# %% [markdown]
# ## Bringing it together
#
# Let's build a callable function that books a few results, per dataset:
# * the sum of weights for the events processed (to use for later luminosity-normalizing the yields)
# * a histogram of the dilepton invariant mass, with category axes for various selection regions of interest
#
# And, additionally, we'll switch to delayed mode and compute the results with an explicit call through `dask`'s interface

# %%
distributed_events = NanoEventsFactory.from_root(
    {file_name: "CollectionTree"},
    schemaclass=PHYSLITESchema,
    uproot_options=dict(filter_name=filter_name),
    delayed=True,
).events()

# %%
distributed_events


# %%
def results_taskgraph(events):
    regions = {
        "ee": {
            "two_electrons": True,
            "no_muons": True,
            "lead_pt_20": True,
            "electrons_opposite_sign": True,
        },
        "ee_same_sign": {
            "two_electrons": True,
            "no_muons": True,
            "lead_pt_20": True,
            "electrons_opposite_sign": False,
        },
        "mm": {
            "two_muons": True,
            "no_electrons": True,
            "lead_pt_20": True,
            "muons_opposite_sign": True,
        },
        "mumu_same_sign": {
            "two_muons": True,
            "no_electrons": True,
            "lead_pt_20": True,
            "muons_opposite_sign": False,
        },
    }

    mass_hist = (
        Hist.new.StrCat(regions.keys(), name="region")
        .Reg(60, 60, 120, name="mass", label="$m_{ll}$ [GeV]")
        .Weight()
    )

    for region, cuts in regions.items():
        good_event = selection.require(**cuts)

        if region.startswith("ee"):
            leptons = events.Electrons[good_event]
        elif region.startswith("mm"):
            # Hack for the time being given PHYSLITESchema needs fixing
            _muons = events.Muons[good_event]
            _muons["m"] = ak.zeros_like(_muons.pt)
            leptons = _muons
        lep1 = leptons[:, 0]
        lep2 = leptons[:, 1]
        mass = (lep1 + lep2).mass

        mass_hist.fill(
            region=region,
            mass=mass,
        )

    out = {
        "sumw": ak.sum(events.EventInfo.mcEventWeights, axis=0),
        "mass": mass_hist,
    }

    return out


# %% [markdown]
# So when we reun we get a `dict` of task graphs

# %%
out_task_graph = results_taskgraph(distributed_events)

out_task_graph

# %% [markdown]
# So we used `dask` to now evaluate the graph with `.compute`

# %%
output, *_ = dask.compute(out_task_graph)

output

# %% [markdown]
# Thanks to `hist` we can slo see nice Jupyter [`reprs`](https://docs.python.org/3/library/functions.html#repr) of the objects

# %%
output["mass"]

# %%
output["mass"][sum, :]

# %%
plot_dir = Path().cwd() / "plots"
plot_dir.mkdir(exist_ok=True)

# %%
mplhep.style.use(mplhep.style.ATLAS)

fig, ax = plt.subplots()

output["mass"][sum, :].plot1d(ax=ax, label="$ll$ mass")
ax.legend()

fig.savefig(plot_dir / "ll_mass.png")

# %%
fig, ax = plt.subplots()

output["mass"]["ee", :].plot1d(ax=ax, label=r"$ee$")
output["mass"]["ee_same_sign", :].plot1d(ax=ax, label=r"$ee$ same sign")
ax.legend()

fig.savefig(plot_dir / "ee_mass.png")
