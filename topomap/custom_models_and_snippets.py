import numpy as np

from string import Template

from pygenn import (create_custom_connectivity_update_model,
                    create_neuron_model, create_sparse_connect_init_snippet,
                    create_weight_update_model)

################################################################################
# Template for calculating squared distance taking into account periodic
# boundary conditions based on neuron ids and grid shape

distance_squared = Template("""
    // x- and y-positions
    const scalar xPre = $id_pre % grid_num_x;
    const scalar xPost = $id_post % grid_num_x;
    const scalar yPre = $id_pre / grid_num_x;
    const scalar yPost = $id_post / grid_num_x;

    // Absolute displacement in x- and y-direction
    scalar dx = fabs( xPost - xPre );
    scalar dy = fabs( yPost - yPre );

    // Periodic boundary conditions
    if (dx > 0.5 * grid_num_x){
        dx = dx - grid_num_x;
    };
    if (dy > 0.5 * grid_num_y){
        dy = dy - grid_num_y;
    };

    // Squared distance
    const scalar dd = dx * dx + dy * dy;
    """)

################################################################################
# Poisson firing rate depends on distance from stimulus specified by idStim.
# After each stimulus interval a new random location is picked.
# If scale > 1, the stimulus location is replicated in the additional blocks.

PoissonSpatiallyCorrelatedInput = create_neuron_model(
    "PoissonSpatiallyCorrelatedInput",
    params=["stim_interval"],
    vars=[("timeStepToSpike", "scalar"), ("stim_rates", "scalar")],
    sim_code="""
        // Reset timeStepToSpike after each stimulus interval such that the
        // new stimulus location immediately becomes effective
        if ( (int) (t / dt) % (int) (stim_interval / dt) == 0){
            timeStepToSpike = 0.;
        }

        if(timeStepToSpike <= 0.0) {
            const scalar isi = 1000. / (stim_rates * dt);
            timeStepToSpike += isi * gennrand_exponential();
        }
        timeStepToSpike -= 1.0;
        """,
    threshold_condition_code="""
        timeStepToSpike <= 0.0
        """
)

################################################################################
# Gaussian connectivity profile
# Spatial pair-wise Bernoulli: Loop over all potential presynaptic neurons once
# and create synapses according to Gaussian profile.
# In-degree is not fixed.
# Multapses are prohibited.
# Make sure that the generated in-degree is smaller than max_indegree.

GaussianProfileWithoutReplacement = \
    create_sparse_connect_init_snippet(
        "GaussianProfileWithoutReplacement",
        params=[
            "peak_prob", "std",
            ("grid_num_x", "unsigned int"), ("grid_num_y", "unsigned int")],
        col_build_code="""
            // iterate over all potential source neurons
            for (unsigned int idPre=0; idPre<num_pre; idPre++) {
        """
        + distance_squared.substitute(id_pre="idPre", id_post="id_post") +
        """
                // Gaussian spatial profile
                const scalar gauss = peak_prob * exp(-dd / ( 2.0 * std * std));

                // Establish synapse if random number is smaller than Gaussian
                // profile at given distance
                if (gennrand_uniform() < gauss)
                {
                    addSynapse(idPre + id_pre_begin);
                }
            }
            """,
        # maximum row length is twice the maximum in-degree of Bamford
        # (this is an arbitrary choice here)
        calc_max_row_len_func=lambda num_pre, num_post, pars: 2 * 32,
        # maximum column length is maximum in-degree
        calc_max_col_len_func=lambda num_pre, num_post, pars: 32)

################################################################################
# Gaussian connectivity profile with fixed in-degree
# (based on FixedNumberPreWithReplacement).
# Loop over postsynaptic neurons (columns) until given in-degree is reached.
# Multapses are allowed.

GaussianProfileFixedNumberPreWithReplacement = \
    create_sparse_connect_init_snippet(
        "GaussianProfileFixedNumberPreWithReplacement",
        params=[
            ("in_degree", "unsigned int"),
            "peak_prob", "std",
            ("grid_num_x", "unsigned int"), ("grid_num_y", "unsigned int")],
        col_build_code="""
            for (unsigned int c = in_degree; c !=0 ;) {
                // Randomly pick a potential source neuron
                const unsigned int idPre = gennrand() % num_pre;
        """
        + distance_squared.substitute(id_pre="idPre", id_post="id_post") +
        """

                // Gaussian spatial profile
                const scalar gauss = peak_prob * exp(-dd / ( 2.0 * std * std));

                // Establish synapse if random number is smaller than Gaussian
                // profile at given distance 
                if (gennrand_uniform() < gauss)
                {
                    addSynapse(idPre + id_pre_begin);
                    c--;
                }
            }
            """,
        # maximum row length is twice the maximum in-degree of Bamford
        # (this is an arbitrary choice here)
        calc_max_row_len_func=lambda num_pre, num_post, pars: 2 * 32,
        # maximum column length is maximum in-degree
        calc_max_col_len_func=lambda num_pre, num_post, pars: 32)

################################################################################
# STDP model with all-to-all spike pairing taken from ("stdp_additive_2_model")
# https://genn-team.github.io/genn/documentation/4/html/d5/d24/sectSynapseModels.html
# (sign of A- in documentation wrong, but correct in code)

STDPAllToAllSpikePairing = create_weight_update_model(
    class_name="STDPAllToAllSpikePairing",
    params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
    derived_params=[
        ("tauPlusDecay", lambda pars, dt: np.exp(-dt / pars["tauPlus"])),
        ("tauMinusDecay", lambda pars, dt: np.exp(-dt / pars["tauMinus"]))
    ],
    vars=[("g", "scalar")],
    pre_vars=[("preTrace", "scalar")],
    post_vars=[("postTrace", "scalar")],

    pre_spike_syn_code="""
        addToPost(g);
        const scalar tDiff = t - st_post;
        if(tDiff > 0) {
            const scalar newWeight = g - (aMinus * postTrace);
            g = fmin(wMax, fmax(wMin, newWeight));
        }
        """,
    post_spike_syn_code="""
        const scalar tDiff = t - st_pre;
        if(tDiff > 0) {
            const scalar newWeight = g + (aPlus * preTrace);
            g = fmin(wMax, fmax(wMin, newWeight));
        }
        """,
    pre_spike_code="""
        preTrace += 1.0;
        """,
    pre_dynamics_code="""
        preTrace *= tauPlusDecay;
        """,
    post_spike_code="""
        postTrace += 1.0;
        """,
    post_dynamics_code="""
        postTrace *= tauMinusDecay;
        """,
)

################################################################################
# Structural plasticity
# 1. Select random synapse slots
# 2. If synapse -> elimination rule
# 3. If not synapse -> formation rule
#
# In host_update_code select the number of rewiring attempts for each row
# (presynaptic neurons)
# note:
# Iterating over all presynaptic neurons and sampling from a binomial
# distribution has here the disadvantage that many zeros would be sampled since
# the number of rewiring attempts per update is typically much smaller than
# the number of presynaptic neurons.
#
# TODO max indegree not used yet
BamfordStructuralPlasticity = \
    create_custom_connectivity_update_model(
        class_name="BamfordStructuralPlasticity",
        params=[("num_rewiring_attempts", "unsigned int"),
                "weight_threshold",
                "new_weight_init",
                "peak_prob",
                "std",
                ("grid_num_x", "unsigned int"),
                ("grid_num_y", "unsigned int"),
                "elimination_probability_dep",
                "elimination_probability_pot"],
        extra_global_params=[("num_rewiring_attempts_per_row", "unsigned int*"),
                             ("bitfield", "uint32_t*")
                             ],
        var_refs=[("g", "scalar")],
        host_update_code="""
            // Set the number of rewiring attempts per presynaptic neuron to zero
            for(unsigned int i = 0; i < num_pre; i++) {
                num_rewiring_attempts_per_row[i] = 0;
            }
        
            // Distribute the rewiring attempts uniformly over the presynaptic neurons.
            // gennrand() samples a random 32-bit unsigned integer and the
            // modulo operation restricts it to [0, num_pre).
            // Note: Multiple attempts for the same idPre are possible, up to
            // the total number of postsynaptic neurons.
            for (unsigned int i = 0; i < num_rewiring_attempts; i++){
                unsigned int idPre;
                bool num_post_not_reached = true;
                while (num_post_not_reached == true) {
                idPre = gennrand() % num_pre;
                    if (num_rewiring_attempts_per_row[idPre] < num_post){
                        num_rewiring_attempts_per_row[idPre]++;
                        num_post_not_reached = false;
                    }
                }
            }

            // Push to device
            pushnum_rewiring_attempts_per_rowToDevice(num_pre);
            """,
        row_update_code="""
            // Code for each presynaptic neuron
            // (id_pre and id_post (in for_each_synapse{}) are set elsewhere)
            // Sample (without replacement) postsynaptic indices for rewiring
            // attempts

            // Number of rewiring attempts of this row
            const unsigned int n = num_rewiring_attempts_per_row[id_pre];

            // Identify section of bitfield this presynaptic neuron operates on
            const unsigned int bitfieldRowWords = (num_post + 31) / 32;
            uint32_t *bitfieldRow = bitfield + (bitfieldRowWords * id_pre);

            // Number of idPosts visited
            unsigned int t = 0;
            
            // Number of idPosts selected so far
            for (unsigned int m = 0; m < n;) {
                // Sample uniform random number in [0.0, 1.0]
                if ((num_post - t) * gennrand_uniform() >= n - m){
                    t++;
                }
                else{
                    const unsigned int word = t / 32;
                    const unsigned int bit = 1 << (t % 32);
                    bitfieldRow[word] |= bit;
            
                    t++;
                    m++;
                }
            }

            // Elimination rule:
            // Iterate over all existing synapses of this presynaptic neuron.
            // If a synapse has been selected, it gets eliminated with a given
            // probability.
            for_each_synapse {
                // Check if bit is set
                const unsigned int word = id_post / 32;
                const unsigned int bit = 1 << (id_post % 32);
                if ( bitfieldRow[word] & bit ){

                    // Check if weight is below threshold
                    const scalar elimProb =  ((g < weight_threshold) 
                                              ? elimination_probability_dep 
                                              : elimination_probability_pot);

                    // Remove dependent on elimination probability
                    if ( gennrand_uniform() < elimProb ){
                        remove_synapse();
                    }
                    // Clear bit
                    bitfieldRow[word] &= ~bit;
                }
            }

            // Formation rule:
            // Iterate over bitfield row of this presynaptic neuron.
            // If a bit it set, a synapse to the corresponding postsynaptic neuron
            // is created.
            const unsigned int numPostWords =  (num_post + 31) / 32;
            for ( unsigned int word = 0; word < numPostWords; word++ ) {
                if ( bitfieldRow[word] != 0 ) {
                    for ( unsigned int bit = 0; bit < 32; bit++ ) {
                        // Check if bit is set
                        if ( bitfieldRow[word] & (1 << bit) ){
                            const unsigned int idPost = (word * 32) + bit;
                            """
        + distance_squared.substitute(id_pre="id_pre", id_post="idPost") +
        """
                            // Gaussian spatial profile
                            const scalar gauss = peak_prob * exp(-dd / (2.0 * std * std));

                            // Add synapse with distance-dependent probability
                            if ( gennrand_uniform() < gauss){
                                add_synapse(idPost, new_weight_init);
                            }

                            // Clear bit
                            bitfieldRow[word] &=  ~(1 << bit);
                        }
                    }
                }
            }
            """
    )
