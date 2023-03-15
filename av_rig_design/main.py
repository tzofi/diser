from fire import Fire

import src


if __name__ == '__main__':
    Fire({
        'scrape': src.sim_nuscenes.scrape,

        'scrape_calib': src.calib.scrape_calib,
        'check_parsing': src.calib.check_parsing,
        'scrape_ncars': src.calib.scrape_ncars,
        'plot_ncardist': src.calib.plot_ncardist,

        'visualize': src.viz.visualize,
        'copy_map_data': src.viz.copy_map_data,
        'find_lyft_run': src.viz.find_lyft_run,
    })