from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home-gxu/wwj22/data/got10k_lmdb'
    settings.got10k_path = '/home-gxu/wwj22/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home-gxu/wwj22/data/itb'
    settings.lasot_extension_subset_path = '/home-gxu/wwj22/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home-gxu/wwj22/data/lasot_lmdb'
    settings.lasot_path = '/home-gxu/wwj22/data/lasot'
    settings.network_path = '/home-gxu/wwj22/AQTP-NL/output/test/networks'  # Where tracking networks are stored.
    settings.nfs_path = '/home-gxu/wwj22/data/nfs'
    settings.otb_lang_path = '/home-gxu/wwj22/data/otb_lang'
    settings.otb_path = '/home-gxu/wwj22/data/otb'
    settings.prj_dir = '/home-gxu/wwj22/AQTP-NL'
    settings.result_plot_path = '/home-gxu/wwj22/AQTP-NL/output/test/result_plots'
    settings.results_path = '/home-gxu/wwj22/AQTP-NL/output/test/tracking_results'  # Where to store tracking results
    settings.save_dir = '/home-gxu/wwj22/AQTP-NL/output'
    settings.segmentation_path = '/home-gxu/wwj22/AQTP-NL/output/test/segmentation_results'
    settings.tc128_path = '/home-gxu/wwj22/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home-gxu/wwj22/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home-gxu/wwj22/data/trackingnet'
    settings.uav_path = '/home-gxu/wwj22/data/uav'
    settings.vot18_path = '/home-gxu/wwj22/data/vot2018'
    settings.vot22_path = '/home-gxu/wwj22/data/vot2022'
    settings.vot_path = '/home-gxu/wwj22/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings






