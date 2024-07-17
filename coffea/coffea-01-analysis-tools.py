# %% [markdown]
# # Analysis tools
#
# Now that we [know how to access data with `NanoEvents`](https://github.com/iris-hep/us-atlas-idap-training-2024/tree/main/PHYSLITE), let's go through some useful columnar analysis tools and idioms for building collections of results, namely, the eventual output of a `coffea` callable (or processor).
# The most familiar type of output may be the histogram (one type of accumulator).
#
# We'll just look at single files for the time being to keep things simple.

# %%
import numpy as np
import awkward as ak
import dask
from hist.dask import Hist
from coffea.nanoevents import NanoEventsFactory, PHYSLITESchema

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
    ak.any(events.Electrons.pt >= 20.0, axis=1)
    | ak.any(events.Muons.pt >= 20.0, axis=1),
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

# %% [markdown]
# ### Higgs example

# %%
GeV = 1000


def object_selection(events):
    """
    Select objects based on kinematic and quality criteria
    """

    electrons = events.Electrons
    muons = events.Muons

    electron_reqs = (
        (electrons.pt / GeV > 20)
        & (np.abs(electrons.eta) < 2.47)
        & (electrons.DFCommonElectronsLHLoose == 1)
    )

    muon_reqs = (muons.pt / GeV > 20) & (np.abs(muons.eta) < 2.7) & (muons.quality == 2)

    # only keep objects that pass our requirements
    electrons = electrons[electron_reqs]
    muons = muons[muon_reqs]

    return electrons, muons


def region_selection(electrons, muons):
    """
    Select events based on object multiplicity
    """

    selections = PackedSelection(dtype="uint64")
    # basic selection criteria
    selections.add("exactly_4e", ak.num(electrons) == 4)
    selections.add("total_e_charge_zero", ak.sum(electrons.charge, axis=1) == 0)
    selections.add("exactly_0m", ak.num(muons) == 0)
    # selection criteria combination
    selections.add(
        "4e0m", selections.all("exactly_4e", "total_e_charge_zero", "exactly_0m")
    )

    return selections.all("4e0m")


def calculate_inv_mass(electrons):
    """
    Construct invariant mass observable
    """

    # reconstruct Higgs as 4e system
    candidates = ak.combinations(electrons, 4)
    e1, e2, e3, e4 = ak.unzip(candidates)
    candidates["p4"] = e1 + e2 + e3 + e4
    higgs_mass = candidates["p4"].mass
    observable = ak.flatten(higgs_mass / GeV)

    return observable


# %%
# select objects and events
el, mu = object_selection(events)
selection_4e0m = region_selection(el, mu)

# %% [markdown]
# The metadata for open data is available by the [metadata table](https://opendata.atlas.cern/docs/documentation/overview_data/data_research_2024#metadata).

# %%
fileset = {
    "Higgs": {
        "files": {
            "root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000001.pool.root.1": "CollectionTree",
            "root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000002.pool.root.1": "CollectionTree",
            "root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000005.pool.root.1": "CollectionTree",
            "root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000006.pool.root.1": "CollectionTree",
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000007.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000008.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000009.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000010.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000011.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000012.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000013.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000014.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000016.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000017.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000018.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000019.pool.root.1' : 'CollectionTree',
            # 'root://xcache.af.uchicago.edu:1094//root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000020.pool.root.1' : 'CollectionTree'
        },
        "metadata": {
            "process": "Higgs",
            "xsec": 28.3,
            "genFiltEff": 1.240e-04,
            "kFactor": 1.45,
            "sumOfWeights": 114108.08,
        },
    }
}

# pre-process
from coffea import dataset_tools

samples, _ = dataset_tools.preprocess(fileset)


# %%
# # XCache
# fileset = {
#             "Higgs"  : {
#                         'files': {
#                                    f"{xcache_caching_server}{open_data_storage}DAOD_PHYSLITE.38191712._000001.pool.root.1": 'CollectionTree',
#                                    f"{xcache_caching_server}{open_data_storage}DAOD_PHYSLITE.38191712._000002.pool.root.1": 'CollectionTree',
#                                    f"{xcache_caching_server}{open_data_storage}DAOD_PHYSLITE.38191712._000003.pool.root.1": 'CollectionTree',
#                                    f"{xcache_caching_server}{open_data_storage}DAOD_PHYSLITE.38191712._000020.pool.root.1": 'CollectionTree',
#                                  },
#                         'metadata': {'process': 'Higgs', 'xsec': 28.3, 'genFiltEff': 1.240E-04, 'kFactor': 1.45, 'sumOfWeights': 114108.08}
#                       }
#           }

# # pre-process
# from coffea import dataset_tools
# samples, _ = dataset_tools.preprocess(fileset)


# %%
# create histogram with observables
def create_histogram(events):
    hist_4e0m = (
        Hist.new.Reg(50, 100, 150, name="m_inv", label=r"$m_{inv.}(4e)$ [GeV]")
        .StrCat([], name="process", label="Process", growth=True)
        .Weight()
    )

    # read metadata
    process_name = events.metadata["process"]
    x_sec = events.metadata["xsec"]
    gen_filt_eff = events.metadata["genFiltEff"]
    k_factor = events.metadata["kFactor"]
    sum_of_weights = events.metadata["sumOfWeights"]

    # as mentined already, the actual analysis code remains the same!
    # select objects and events
    el, mu = object_selection(events)
    selection_4e0m = region_selection(el, mu)

    # normalization for MC
    lumi = 36100.0  # /pb This is the luminosity (the amount of real data collected) corresponding to the open data released
    xsec_weight = x_sec * gen_filt_eff * k_factor * lumi / sum_of_weights
    print(f"Processing {process_name} with xsec weight {xsec_weight}")
    mc_weight = events.EventInfo[selection_4e0m][:, 1]["mcEventWeights"]

    # observable calculation and histogram filling
    inv_mass = calculate_inv_mass(el[selection_4e0m])
    hist_4e0m.fill(inv_mass, weight=mc_weight * xsec_weight, process=process_name)

    return hist_4e0m


# %%
# create the task graph
tasks = dataset_tools.apply_to_fileset(
    create_histogram,
    samples,
    schemaclass=PHYSLITESchema,
    uproot_options=dict(filter_name=filter_name),
)

# %%
# %%time

# execute
(out,) = dask.compute(tasks)

# %%
# stack all the histograms together, as we processed each file separately
full_histogram = sum(hist for hist in out.values())

# %%
plot_dir = Path().cwd() / "plots"
plot_dir.mkdir(exist_ok=True)

# %%
# plot
artists = full_histogram.plot(histtype="fill")

ax = artists[0].stairs.axes
ax.legend()
ax.set_ylabel("A.U.")

fig = ax.get_figure()
fig.savefig(plot_dir / "higgs_mass.png")
