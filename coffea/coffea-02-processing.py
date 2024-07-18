# %% [markdown]
# # Coffea Processors
# Coffea relies mainly on [uproot](https://github.com/scikit-hep/uproot) to provide access to ROOT files for analysis.
# As a usual analysis will involve processing tens to thousands of files, totalling gigabytes to terabytes of data, there is a certain amount of work to be done to build a parallelized framework to process the data in a reasonable amount of time.
#
# In coffea up to 0.7 (SemVer), a `coffea.processor` module was provided to encapsulate the core functionality of the analysis, which could be run locally or distributed via a number of Executors. This allowed users to worry just about the actual analysis code and not about how to implement efficient parallelization, assuming that the parallization is a trivial map-reduce operation (e.g. filling histograms and adding them together).
#
# In coffa 2024 (CalVer), integration with `dask` is deeper (via `dask_awkward` and `uproot.dask`), and whether an analysis is to be executed on local or distributed resources, a TaskGraph encapsulating the analysis is created (with a bypass functionality for an eager version still possible when the scale of your input data is sufficiently small). We will demonstrate how to use callable code to build these TGs.
#
# (Sidenote: with some adaptations for the new version of scikit-hep/coffea, a SemVer coffea processor module's `process` function can serve as the callable function - we will follow this model for convenience as we are transitioning from SemVer to CalVer coffea)
#
#
# Let's start by writing a simple processor class that reads some CMS open data and plots a dimuon mass spectrum.
# We'll start by copying the [ProcessorABC](https://coffeateam.github.io/coffea/api/coffea.processor.ProcessorABC.html#coffea.processor.ProcessorABC) skeleton and filling in some details:
#
#  * Remove `flag`, as we won't use it
#  * Adding a new histogram for $m_{\mu \mu}$
#  * Building a [Candidate](https://coffeateam.github.io/coffea/api/coffea.nanoevents.methods.candidate.PtEtaPhiMCandidate.html#coffea.nanoevents.methods.candidate.PtEtaPhiMCandidate) record for muons, since we will read it with `BaseSchema` interpretation (the files used here could be read with `NanoAODSchema` but we want to show how to build vector objects from other TTree formats)
#  * Calculating the dimuon invariant mass

# %%
import awkward as ak
from coffea import processor
from coffea.nanoevents.methods import candidate
import dask
from hist.dask import Hist


# %%
class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):
        _ = events.metadata["dataset"]
        muons = ak.zip(
            {
                "pt": events.Muon_pt,
                "eta": events.Muon_eta,
                "phi": events.Muon_phi,
                "mass": events.Muon_mass,
                "charge": events.Muon_charge,
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        h_mass = (
            Hist.new.StrCat(["opposite", "same"], name="sign")
            .Log(1000, 0.2, 200.0, name="mass", label="$m_{\mu\mu}$ [GeV]")
            .Int64()
        )

        cut = (ak.num(muons) == 2) & (ak.sum(muons.charge, axis=1) == 0)
        # add first and second muon in every event together
        dimuon = muons[cut][:, 0] + muons[cut][:, 1]
        h_mass.fill(sign="opposite", mass=dimuon.mass)

        cut = (ak.num(muons) == 2) & (ak.sum(muons.charge, axis=1) != 0)
        dimuon = muons[cut][:, 0] + muons[cut][:, 1]
        h_mass.fill(sign="same", mass=dimuon.mass)

        return {
            "entries": ak.num(events, axis=0),
            "mass": h_mass,
        }

    def postprocess(self, accumulator):
        pass


# %% [markdown]
# If we were to just use bare uproot to execute this processor, we could do that with the following example, which:
#
#  * Opens a CMS open data file
#  * Creates a NanoEvents object using `BaseSchema` (roughly equivalent to the output of `uproot.lazy`)
#  * Creates a `MyProcessor` instance
#  * Runs the `process()` function, which returns our accumulators
#

# %%
xcache_caching_server = "root://xcache.af.uchicago.edu:1094//"
mumu_data_filename = f"{xcache_caching_server}root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root"

# %%
from coffea.nanoevents import NanoEventsFactory, BaseSchema

events = NanoEventsFactory.from_root(
    {mumu_data_filename: "Events"},
    entry_stop=10000,
    metadata={"dataset": "DoubleMuon"},
    schemaclass=BaseSchema,
    delayed=True,
).events()
p = MyProcessor()
task_graph = p.process(events)
task_graph

# %%
dask.visualize(task_graph["mass"], optimize_graph=False)

# %%
dask.visualize(task_graph["mass"], optimize_graph=True)

# %%
# %%time

out, *_ = dask.compute(task_graph)

# %%
from pathlib import Path

plot_dir = Path().cwd() / "plots"
plot_dir.mkdir(exist_ok=True)

# %%
import matplotlib.pyplot as plt
import mplhep

mplhep.style.use(mplhep.style.ATLAS)

fig, ax = plt.subplots()
out["mass"].plot1d(ax=ax)
ax.set_xscale("log")
ax.legend(title="Dimuon charge")

fig.savefig(plot_dir / "dimuon_charge.png")

# %% [markdown]
# # Filesets
# We'll need to construct a fileset to run over

# %%
mumu_simulation_filename = f"{xcache_caching_server}root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/ZZTo4mu.root"

# %%
initial_fileset = {
    "DoubleMuon": {
        "files": {
            mumu_data_filename: "Events",
        },
        "metadata": {
            "is_mc": False,
        },
    },
    "ZZ to 4mu": {
        "files": {
            mumu_simulation_filename: "Events",
        },
        "metadata": {
            "is_mc": True,
        },
    },
}

# %% [markdown]
# # Preprocessing
# There are dataset discovery tools inside of `coffea` to help construct such datasets. Those will not be demonstrated here. For now, we'll take the above `initial_fileset` and preprocess it.

# %%
from coffea.dataset_tools import apply_to_fileset, max_chunks, max_files, preprocess

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

# %% [markdown]
# # Preprocessed fileset
# Lets have a look at the contents of the `preprocessed_available` part of the fileset

# %%
preprocessed_available

# %% [markdown]
# ## Saving a preprocessed fileset
# We can use the `gzip`, `pickle`, and `json` modules/libraries to both save and reload datasets directly. We'll do this short example below

# %%
from pathlib import Path

fileset_dir = Path().cwd() / "filesets"
fileset_dir.mkdir(exist_ok=True)

# %%
import gzip
import json

output_file = "example_fileset"
with gzip.open(fileset_dir / f"{output_file}_available.json.gz", "wt") as file:
    json.dump(preprocessed_available, file, indent=2)
    print(f"Saved available fileset chunks to {output_file}_available.json.gz")
with gzip.open(fileset_dir / f"{output_file}_all.json.gz", "wt") as file:
    json.dump(preprocessed_total, file, indent=2)
    print(f"Saved complete fileset chunks to {output_file}_all.json.gz")

# %% [markdown]
# We could then reload these filesets and quickly pick up where we left off. Often we'll want to preprocess again "soon" before analyzing data because this will let us catch which files are accessible now and which are not. The saved filesets may be useful for tracking, and we may have enough stability to reuse it for some period of time.

# %%
with gzip.open(fileset_dir / f"{output_file}_available.json.gz", "rt") as file:
    reloaded_available = json.load(file)
with gzip.open(fileset_dir / f"{output_file}_all.json.gz", "rt") as file:
    reloaded_all = json.load(file)

# %% [markdown]
# # Slicing chunks and files
# Given this preprocessed fileset, we can test our processor on just a few chunks of a handful of files. To do this, we use the `max_files` and `max_chunks` functions from the dataset tools

# %%
test_preprocessed_files = max_files(preprocessed_available, 1)
test_preprocessed = max_chunks(test_preprocessed_files, 3)

# %%
test_preprocessed

# %%
small_task_graph, small_rep = apply_to_fileset(
    data_manipulation=MyProcessor(),
    fileset=test_preprocessed,
    schemaclass=BaseSchema,
    uproot_options={"allow_read_errors_with_report": (OSError, KeyError)},
)

# %%
dask.visualize(small_task_graph, optimize_graph=True)

# %%
small_computed, small_rep_computed = dask.compute(small_task_graph, small_rep)

# %%
small_rep_computed["DoubleMuon"]

# %%
small_computed

# %% [markdown]
# Now, if we want to use more than a single core on our machine, we simply change [IterativeExecutor](https://coffeateam.github.io/coffea/api/coffea.processor.IterativeExecutor.html) for [FuturesExecutor](https://coffeateam.github.io/coffea/api/coffea.processor.FuturesExecutor.html), which uses the python [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) standard library. We can then set the most interesting argument to the `FuturesExecutor`: the number of cores to use (2):

# %%
full_task_graph, rep = apply_to_fileset(
    data_manipulation=MyProcessor(),
    fileset=preprocessed_available,
    schemaclass=BaseSchema,
    uproot_options={"allow_read_errors_with_report": (OSError, KeyError)},
)

# %%
# %%time

out, rep = dask.compute(full_task_graph, rep)

# %%
out

# %% [markdown]
# Hopefully this ran faster than the previous cell, but that may depend on how many cores are available on the machine you are running this notebook.

# %%
plot_dir = Path().cwd() / "plots"
plot_dir.mkdir(exist_ok=True)

# %%
mplhep.style.use(mplhep.style.ATLAS)

fig, ax = plt.subplots()
out["DoubleMuon"]["mass"].plot1d(ax=ax)
ax.set_xscale("log")
ax.legend(title="Dimuon charge")

fig.savefig(plot_dir / "dimuon_charge.png")
