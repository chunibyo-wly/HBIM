import nlopt


class Settings:
    folder_gt = "../data/gt1/"
    folder_export = "../data/results/"
    max_eval_fast = 1000
    opt_alg = nlopt.GN_DIRECT  # LN_NEWUOA
    algorithms = {
        "GN_DIRECT": nlopt.GN_DIRECT,
        "GN_MLSL": nlopt.GN_MLSL,
        "LN_BOBYQA": nlopt.LN_BOBYQA,
        "LN_NEWUOA_BOUND": nlopt.LN_NEWUOA_BOUND,
        "LN_NELDERMEAD": nlopt.LN_NELDERMEAD,
    }  # 42-43
    all_algorithms = {
        "nlopt": [
            nlopt.GN_DIRECT,
            nlopt.GN_DIRECT_L,
            nlopt.GN_DIRECT_L_RAND,
            nlopt.GN_DIRECT_NOSCAL,
            nlopt.GN_DIRECT_L_NOSCAL,  # 0-4
            nlopt.GN_DIRECT_L_RAND_NOSCAL,
            nlopt.GN_ORIG_DIRECT,
            nlopt.GN_ORIG_DIRECT_L,
            nlopt.GD_STOGO,
            nlopt.GD_STOGO_RAND,  # 5-9
            nlopt.LN_PRAXIS,
            nlopt.GN_CRS2_LM,  # 12, 19
            nlopt.GN_MLSL,
            nlopt.GD_MLSL,
            nlopt.GN_MLSL_LDS,
            nlopt.GD_MLSL_LDS,  # 20-23
            nlopt.LN_COBYLA,
            nlopt.LN_NEWUOA,
            nlopt.LN_NEWUOA_BOUND,
            nlopt.LN_NELDERMEAD,
            nlopt.LN_SBPLX,  # 25-29
            nlopt.LN_AUGLAG,
            nlopt.LN_AUGLAG_EQ,
            nlopt.LN_BOBYQA,  # 30, 32, 34
            nlopt.GN_ISRES,
            nlopt.AUGLAG,
            nlopt.AUGLAG_EQ,
            nlopt.G_MLSL,
            nlopt.G_MLSL_LDS,  # 35-39
            nlopt.GN_ESCH,
            nlopt.GN_AGS,
        ]
    }  # 42-43
    pcd_test_path = r"data/test/MA-room1-2cm.ply"
    mesh_test_path = r"data/comp/column1.obj"
