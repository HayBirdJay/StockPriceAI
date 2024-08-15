import argparse
import get_training_data
import subprocess
import utils

"""
Run this file with the associated parameters to run our news sentiment/stock price predictive model.
"""


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparse.addArgument("-t", "--ticker", help="stock ticker name to run model on (ex: ATT, APPL, MSFT)", required=True)
    argparse.addArgument("-m", "--model", help="model you'd liked to use", choices=["lstm", "gradboost","combined"], required=True)
    argparse.addArgument("-oc", "--outputcsv",help="output file to save csv data", required=True)
    argparse.addArgument("-g","--graphfile", help="if set, will graph the data and save it at the file.")
    args = argparser.parse_args()

    get_training_data.getTrainingData(args.ticker)
    utils.json_to_csv(args.ticker)

    if args.model == "lstm":
        script="models/lstm.py"
    elif args.model == "gradboost":
        script="models/hybrid_grad_boost.py"
    elif args.model == "combined":
        script="models/wombo_combo_v2.py"
    else:
        raise ValueError("not a supported model")
    
    cmd = ['python', script, "-t", args.ticker]

    if args.graphfile:
        cmd.append(["-g", args.graphfile])

    subprocess.run(cmd)
    