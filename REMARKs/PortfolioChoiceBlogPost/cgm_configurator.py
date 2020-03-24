import argparse
import yaml
import subprocess

ORIGIN = f"https://github.com/econ-ark/REMARK"
REMARK_BRANCH = f"master"
DOCKER_IMAGE = f"econark/econ-ark-notebook"
DO_FILE = f"do_MIN.py"
PATH_TO_PARAMS = f"/home/jovyan/REMARK/REMARKs/CGMPortfolio/Code/Python/Calibration/"
PATH_TO_FIGURES = f"/home/jovyan/REMARK/REMARKs/CGMPortfolio/Code/Python/Figures/"
PATH_TO_SCRIPT = f"REMARK/REMARKs/CGMPortfolio"
RESULTS_DIR = f"figures"

# Take the file as an argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "config", help="A YAML config file for custom parameters for REMARKs"
)
args = parser.parse_args()


with open(args.config, "r") as stream:
    config_parameters = yaml.safe_load(stream)

print(config_parameters)

pwd = subprocess.run(["pwd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
mount = str(pwd.stdout)[2:-3] + ":/home/jovyan/work"
# mount the present directory and start up a container
container_id = subprocess.run(
    ["docker", "run", "-v", mount, "-d", DOCKER_IMAGE], stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
container_id = container_id.stdout.decode("utf-8")[:-1]
# pull the master branch
subprocess.run(
    [
        f'docker exec -it {container_id} bash -c "cd REMARK; git pull {ORIGIN} {REMARK_BRANCH}"'
    ],
    shell=True,
)
subprocess.run([f"docker exec -it  {container_id} bash -c 'pip uninstall -y econ-ark; pip install econ-ark==0.10.5'"], shell=True)
# copy the params file to params_init file
subprocess.run(
    [
        f"docker exec -it  {container_id} bash -c 'cp {PATH_TO_PARAMS}params.py {PATH_TO_PARAMS}params_init.py'"
    ],
    shell=True,
)
# copy the params files to current work directory
subprocess.run(
    [
        f"docker exec -it  {container_id} bash -c 'cp {PATH_TO_PARAMS}params* /home/jovyan/work'"
    ],
    shell=True,
)
# create a directory to store results from the run
subprocess.run(
    [f"docker exec -it  {container_id} bash -c 'mkdir /home/jovyan/work/{RESULTS_DIR}'"],
    shell=True,
)

dict_portfolio_keys = [
    "CRRA",
    "Rfree",
    "DiscFac",
    "T_age",
    "T_cycle",
    "T_retire",
    "LivPrb",
    "PermGroFac",
    "cycles",
    "PermShkStd",
    "PermShkCount",
    "TranShkStd",
    "TranShkCount",
    "UnempPrb",
    "UnempPrbRet",
    "IncUnemp",
    "IncUnempRet",
    "BoroCnstArt",
    "tax_rate",
    "RiskyAvg",
    "RiskyStd",
    "RiskyAvgTrue",
    "RiskyStdTrue",
    "RiskyCount",
    "RiskyShareCount",
    "aXtraMin",
    "aXtraMax",
    "aXtraCount",
    "aXtraExtra",
    "aXtraNestFac",
    "vFuncBool",
    "CubicBool",
    "AgentCount",
    "pLvlInitMean",
    "pLvlInitStd",
    "T_sim",
    "PermGroFacAgg",
    "aNrmInitMean",
    "aNrmInitStd",
]

parameters_update = [
    "from .params_init import dict_portfolio, time_params, Mu, Rfree, Std, det_income, norm_factor, age_plot_params, repl_fac, a, b1, b2, b3, std_perm_shock, std_tran_shock",
    "import numpy as np",
]
for parameter in config_parameters:
    print(f"Running docker instance against parameters: {parameter} ")
    for key, val in config_parameters[parameter].items():
        # check if it's in time_params
        if key in ["Age_born", "Age_retire", "Age_death"]:
            parameters_update.append(f"time_params['{key}'] = {val}")
            # changing time_params effect dict_portfolio elements too
            parameters_update.append(
                f"dict_portfolio['T_age'] = time_params['Age_death'] - time_params['Age_born'] + 1"
            )
            parameters_update.append(
                f"dict_portfolio['T_cycle'] = time_params['Age_death'] - time_params['Age_born']"
            )
            parameters_update.append(
                f"dict_portfolio['T_retire'] = time_params['Age_retire'] - time_params['Age_born'] + 1"
            )
            parameters_update.append(
                f"dict_portfolio['T_sim'] = (time_params['Age_death'] - time_params['Age_born'] + 1)*50"
            )
            # fix notches (income growth and more parameters depends on age parameters)
            age_varying_paramters = [
            "f = np.arange(time_params['Age_born'], time_params['Age_retire'] + 1, 1)",
            "f = a + b1*f + b2*(f**2) + b3*(f**3)",
            "det_work_inc = np.exp(f)",
            "det_ret_inc = repl_fac*det_work_inc[-1]*np.ones(time_params['Age_death'] - time_params['Age_retire'])",
            "det_income = np.concatenate((det_work_inc, det_ret_inc))",
            "gr_fac = np.exp(np.diff(np.log(det_income)))",
            "std_tran_vec = np.array([std_tran_shock]*(time_params['Age_death'] - time_params['Age_born']))",
            "std_perm_vec = np.array([std_perm_shock]*(time_params['Age_death'] - time_params['Age_born']))",
            "dict_portfolio['PermGroFac'] = gr_fac.tolist()",
            "dict_portfolio['pLvlInitMean'] = np.log(det_income[0])",
            "dict_portfolio['TranShkStd'] = std_tran_vec",
            "dict_portfolio['PermShkStd'] = std_perm_vec"
            ]
            for para_age in age_varying_paramters:
                parameters_update.append(para_age)
        # check if it's det_income
        elif key in ["det_income"]:
            parameters_update.append(f"det_income = np.array({val})")
            parameters_update.append("dict_portfolio['pLvlInitMean'] = np.log(det_income[0])")
        # check if it's in dict_portfolio
        elif key in dict_portfolio_keys:
            parameters_update.append(f"dict_portfolio['{key}'] = {val}")
        elif key in ["age_plot_params"]:
            parameters_update.append(f"age_plot_params = {val}")
        else:
            print("Parameter provided in config file not found")
    parameters_update.append(
                f"dict_portfolio['LivPrb'] = dict_portfolio['LivPrb'][(time_params['Age_born'] - 20):(time_params['Age_death'] - 20)]"
            )
    for i in parameters_update:
        print(i)
        print('\n')
    with open("params.py", "w") as f:
        for item in parameters_update:
            f.write("%s\n" % item)
    # restart parameter update list
    parameters_update = parameters_update[0:2]
    # copy new parameters file to the REMARK
    subprocess.run(
        [
            f"docker exec -it  {container_id} bash -c 'cp /home/jovyan/work/params.py {PATH_TO_PARAMS}params.py'"
        ],
        shell=True,
    )
    # remove previous figures from the REMARK
    subprocess.run(
        [f"docker exec -it {container_id} bash -c 'rm {PATH_TO_FIGURES}*'"], shell=True
    )
    # run the do_X file and get the results
    subprocess.run(
        [
            f"docker exec -it  {container_id} bash -c 'cd {PATH_TO_SCRIPT}; ipython {DO_FILE}'"
        ],
        shell=True,
    )
    # create a folder to store the figures for this parameter
    subprocess.run(
        [
            f"docker exec -it  {container_id} bash -c 'mkdir /home/jovyan/work/{RESULTS_DIR}/figure_{parameter}'"
        ],
        shell=True,
    )
    # copy the files created in figures to results
    subprocess.run(
        [
            f"docker exec -it {container_id} bash -c 'cp {PATH_TO_FIGURES}* /home/jovyan/work/{RESULTS_DIR}/figure_{parameter}/'"
        ],
        shell=True,
    )


subprocess.run([f"docker stop {container_id}"], shell=True)
subprocess.run([f"rm params.py params_init.py"], shell=True)
