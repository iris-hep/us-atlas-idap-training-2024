# %% [markdown]
# # Scaleout

# %%
from pathlib import Path

import numpy as np
import awkward as ak
import dask
from hist.dask import Hist
from coffea import dataset_tools
from coffea.nanoevents import PHYSLITESchema
from coffea.analysis_tools import PackedSelection

PHYSLITESchema.warn_missing_crossrefs = False

# %%
from importlib.metadata import version

for package in ["numpy", "awkward", "uproot", "coffea", "dask"]:
    print(f"# {package}: v{version(package)}")


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
xcache_caching_server = "root://xcache.af.uchicago.edu:1094//"

# %% [markdown]
# The metadata for open data is available by the [metadata table](https://opendata.atlas.cern/docs/documentation/overview_data/data_research_2024#metadata).

# %%
fileset = {
    "Higgs": {
        "files": {
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000001.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000002.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000005.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000006.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000007.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000008.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000009.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000010.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000011.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000012.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000013.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000014.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000016.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000017.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000018.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000019.pool.root.1": "CollectionTree",
            f"{xcache_caching_server}root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc20_13TeV/DAOD_PHYSLITE.38191712._000020.pool.root.1": "CollectionTree",
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
samples, _ = dataset_tools.preprocess(fileset)

# %%
# create the task graph
tasks = dataset_tools.apply_to_fileset(
    create_histogram,
    samples,
    schemaclass=PHYSLITESchema,
    uproot_options=dict(filter_name=filter_name),
)

# %% [markdown]
# This will take about 1 mintue

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

# %% [markdown]
# ## Scaleout with Dask
#
# 1. Select the Dask cluster in the Dask dashboard on the left
# 2. Ensure that adaptive scaling is set (click the "Scale" button)
# 3. Click and drag the box to the Jupyter notebook which will create a cell like
#
# ```python
# from dask.distributed import Client
#
# client = Client(<scheduler address>)
# client
# ```

# %%
#
# Drag here
#

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
