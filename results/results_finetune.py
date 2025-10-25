import numpy as np

def main():

    # early fusion
    ef_upmc_food = np.array([0.8928, 0.8930, 0.8930])
    ef_mm_imdb = np.array([0.9287, 0.9289, 0.9296])
    ef_hm = np.array([0.6893, 0.6790, 0.6737])
    ef_hm_auc = np.array([0.7264, 0.7231, 0.7183])

    ef_upmc_food_mean, ef_upmc_food_std = np.mean(ef_upmc_food), np.std(ef_upmc_food)
    ef_mm_imdb_mean, ef_mm_imdb_std = np.mean(ef_mm_imdb), np.std(ef_mm_imdb)
    ef_hm_mean, ef_hm_std = np.mean(ef_hm), np.std(ef_hm)
    ef_hm_auc_mean, ef_hm_auc_std = np.mean(ef_hm_auc), np.std(ef_hm_auc)



    # middle fusion
    mf_upmc_food = np.array([0.9167, 0.9198, 0.9175])
    mf_mm_imdb = np.array([0.9300, 0.9301, 0.9296])
    mf_hm = np.array([0.7003, 0.7007, 0.6900])
    mf_hm_auc = np.array([0.7555, 0.7566, 0.7369])

    mf_upmc_food_mean, mf_upmc_food_std = np.mean(mf_upmc_food), np.std(mf_upmc_food)
    mf_mm_imdb_mean, mf_mm_imdb_std = np.mean(mf_mm_imdb), np.std(mf_mm_imdb)
    mf_hm_mean, mf_hm_std = np.mean(mf_hm), np.std(mf_hm)
    mf_hm_auc_mean, mf_hm_auc_std = np.mean(mf_hm_auc), np.std(mf_hm_auc)

    # late fusion
    lf_upmc_food = np.array([0.9278, 0.9273, 0.9284])
    lf_mm_imdb = np.array([0.9305, 0.9309, 0.9311])
    lf_hm = np.array([0.7110, 0.7060, 0.6887])
    lf_hm_auc = np.array([0.7693, 0.7620, 0.7597])

    lf_upmc_food_mean, lf_upmc_food_std = np.mean(lf_upmc_food), np.std(lf_upmc_food)
    lf_mm_imdb_mean, lf_mm_imdb_std = np.mean(lf_mm_imdb), np.std(lf_mm_imdb)
    lf_hm_mean, lf_hm_std = np.mean(lf_hm), np.std(lf_hm)
    lf_hm_auc_mean, lf_hm_auc_std = np.mean(lf_hm_auc), np.std(lf_hm_auc)

    # asymmetric fusion
    af_upmc_food = np.array([0.9011, 0.9008, 0.9011])
    af_mm_imdb = np.array([0.9290, 0.9291, 0.9292])
    af_hm = np.array([0.6700, 0.6797, 0.6860])
    af_hm_auc = np.array([0.7224, 0.7269, 0.7242])

    af_upmc_food_mean, af_upmc_food_std = np.mean(af_upmc_food), np.std(af_upmc_food)
    af_mm_imdb_mean, af_mm_imdb_std = np.mean(af_mm_imdb), np.std(af_mm_imdb)
    af_hm_mean, af_hm_std = np.mean(af_hm), np.std(af_hm)
    af_hm_auc_mean, af_hm_auc_std = np.mean(af_hm_auc), np.std(af_hm_auc)


    # no coattn
    nc_upmc_food = np.array([0.8863, 0.8861, 0.8887])
    nc_mm_imdb = np.array([0.9262, 0.9271, 0.9271])
    nc_hm = np.array([0.6277, 0.6013, 0.6347])
    nc_hm_auc = np.array([0.6561, 0.6522, 0.6543])

    nc_upmc_food_mean, nc_upmc_food_std = np.mean(nc_upmc_food), np.std(nc_upmc_food)
    nc_mm_imdb_mean, nc_mm_imdb_std = np.mean(nc_mm_imdb), np.std(nc_mm_imdb)
    nc_hm_mean, nc_hm_std = np.mean(nc_hm), np.std(nc_hm)
    nc_hm_auc_mean, nc_hm_auc_std = np.mean(nc_hm_auc), np.std(nc_hm_auc)


    #optuna 1
    upmc_food = np.array([0.9181, 0.9172, 0.9181])
    mm_imdb = np.array([0.9294, 0.9297, 0.9299])
    hm = np.array([0.6947, 0.6853, 0.6963])
    hm_auc = np.array([0.7476, 0.7425, 0.7480])

    op1_upmc_food_mean, op1_upmc_food_std = np.mean(upmc_food), np.std(upmc_food)
    op1_mm_imdb_mean, op1_mm_imdb_std = np.mean(mm_imdb), np.std(mm_imdb)
    op1_hm_mean, op1_hm_std = np.mean(hm), np.std(hm)
    op1_hm_auc_mean, op1_hm_auc_std = np.mean(hm_auc), np.std(hm_auc)


    print("early fusion:")
    print(f"upmc accuracy: {ef_upmc_food_mean:.4f} ± {ef_upmc_food_std:.4f}")
    print(f"imdb accuracy: {ef_mm_imdb_mean:.4f} ± {ef_mm_imdb_std:.4f}")
    print(f"hm   accuracy: {ef_hm_mean:.4f} ± {ef_hm_std:.4f}")
    print(f"hm rocauc    : {ef_hm_auc_mean:.4f} ± {ef_hm_auc_std:.4f}")
    print("-"*35)

    print("middle fusion:")
    print(f"upmc accuracy: {mf_upmc_food_mean:.4f} ± {mf_upmc_food_std:.4f}")
    print(f"imdb accuracy: {mf_mm_imdb_mean:.4f} ± {mf_mm_imdb_std:.4f}")
    print(f"hm   accuracy: {mf_hm_mean:.4f} ± {mf_hm_std:.4f}")
    print(f"hm rocauc    : {mf_hm_auc_mean:.4f} ± {mf_hm_auc_std:.4f}")
    print("-"*35)

    print("late fusion:")
    print(f"upmc accuracy: {lf_upmc_food_mean:.4f} ± {lf_upmc_food_std:.4f}")
    print(f"imdb accuracy: {lf_mm_imdb_mean:.4f} ± {lf_mm_imdb_std:.4f}")
    print(f"hm   accuracy: {lf_hm_mean:.4f} ± {lf_hm_std:.4f}")
    print(f"hm rocauc    : {lf_hm_auc_mean:.4f} ± {lf_hm_auc_std:.4f}")
    print("-"*35)

    print("asymmetric fusion:")
    print(f"upmc accuracy: {af_upmc_food_mean:.4f} ± {af_upmc_food_std:.4f}")
    print(f"imdb accuracy: {af_mm_imdb_mean:.4f} ± {af_mm_imdb_std:.4f}")
    print(f"hm   accuracy: {af_hm_mean:.4f} ± {af_hm_std:.4f}")
    print(f"hm rocauc    : {af_hm_auc_mean:.4f} ± {af_hm_auc_std:.4f}")
    print("-"*35)


    print("no coattn:")
    print(f"upmc accuracy: {nc_upmc_food_mean:.4f} ± {nc_upmc_food_std:.4f}")
    print(f"imdb accuracy: {nc_mm_imdb_mean:.4f} ± {nc_mm_imdb_std:.4f}")
    print(f"hm   accuracy: {nc_hm_mean:.4f} ± {nc_hm_std:.4f}")
    print(f"hm rocauc    : {nc_hm_auc_mean:.4f} ± {nc_hm_auc_std:.4f}")
    print("-"*35)

    print("optuna1:")
    print(f"upmc accuracy: {op1_upmc_food_mean:.4f} ± {op1_upmc_food_std:.4f}")
    print(f"imdb accuracy: {op1_mm_imdb_mean:.4f} ± {op1_mm_imdb_std:.4f}")
    print(f"hm   accuracy: {op1_hm_mean:.4f} ± {op1_hm_std:.4f}")
    print(f"hm rocauc    : {op1_hm_auc_mean:.4f} ± {op1_hm_auc_std:.4f}")
    


main()