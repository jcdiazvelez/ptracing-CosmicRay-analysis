import json

job = {
    "nside": 16,
    "binnings": [10, 30, 50],
    "imposed_distribution": [0.001, 0.001],

    "observational?": True,
    "observational_parameters": [0.25, 24.0],

    "kolmogorov?": True,
    "kolmogorov_width": 3,
    "kolmogorov_limits": [1.0, 20.0],

    "plot_unweighted?": True,
    "plot_imposed?": True,
    "physical_index": 2.70,

    "raw_data_location": "../data/raw/",
    "particle_data_location": "../data/particles/",
    "weighted_data_location": "../data/weighted/",
    "map_data_location": "../data/maps/",

    "output_location": "../figs/"
}

with open('config.json', 'w') as f:
    json.dump(job, f, indent=4)
