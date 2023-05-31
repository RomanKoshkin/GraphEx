# Import PPSeq
import PPSeq
const seq = PPSeq

# Other Imports
import PyPlot: plt
import DelimitedFiles: readdlm
import Random
import StatsBase: quantile


dataset_len = parse(Int, ARGS[1])


# Songbird metadata
num_neurons = 76
max_time = dataset_len

# Randomly permute neuron labels.
# (This hides the sequences, to make things interesting.)
_p = Random.randperm(num_neurons)

# Load spikes.
spikes = seq.Spike[]
for (n, t) in eachrow(readdlm("data/one_seq_$(max_time).txt", '\t', Float64, '\n'))
    push!(spikes, seq.Spike(_p[Int(n)], t))
end

config = Dict(

    # Model hyperparameters
    :num_sequence_types =>  1,
    :seq_type_conc_param => 1.0,
    :seq_event_rate => 1.0,

    :mean_event_amplitude => 100.0,
    :var_event_amplitude => 1000.0,
    
    :neuron_response_conc_param => 0.1,
    :neuron_offset_pseudo_obs => 1.0,
    :neuron_width_pseudo_obs => 1.0,
    :neuron_width_prior => 0.5,
    
    :num_warp_values => 1,
    :max_warp => 1.0,
    :warp_variance => 1.0,

    :mean_bkgd_spike_rate => 30.0,
    :var_bkgd_spike_rate => 30.0,
    :bkgd_spikes_conc_param => 0.3,
    :max_sequence_length => Inf,
    
    # MCMC Sampling parameters.
    :num_threads => 64,
    :num_anneals => 10,
    :samples_per_anneal => 100,
    :max_temperature => 40.0,
    :save_every_during_anneal => 10,
    :samples_after_anneal => 2000,
    :save_every_after_anneal => 10,
    :split_merge_moves_during_anneal => 10,
    :split_merge_moves_after_anneal => 10,
    :split_merge_window => 1.0,

);



# Initialize all spikes to background process.
init_assignments = fill(-1, length(spikes))

# Construct model struct (PPSeq instance).
model = seq.construct_model(config, float(max_time), num_neurons)

# Run Gibbs sampling with an initial annealing period.
results = seq.easy_sample!(model, spikes, init_assignments, config);


# Grab the final MCMC sample
final_globals = results[:globals_hist][end]
final_events = results[:latent_event_hist][end]
final_assignments = results[:assignment_hist][:, end]

# Helpful utility function that sorts the neurons to reveal sequences.
neuron_ordering = seq.sortperm_neurons(final_globals)

# Plot model-annotated raster.
# fig = seq.plot_raster(
#     spikes,
#     final_events,
#     final_assignments,
#     neuron_ordering;
#     color_cycle=["red", "blue", "green", "magenta"] # colors for each sequence type can be modified.
# )
# fig.set_size_inches([7, 3]);

events = zeros(0)

for e in final_events
    plt.axvline(e.timestamp)
    append!(events, e.timestamp)
end

using NPZ

npzwrite("data_outputs/neuron_ordering_$(max_time).npy", neuron_ordering)
npzwrite("data_outputs/final_assignments_$(max_time).npy", final_assignments)
npzwrite("data_outputs/events_$(max_time).npy", events)