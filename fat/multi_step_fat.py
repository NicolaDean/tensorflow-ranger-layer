

import pandas as pd
from single_layer_experiment import *


def load_report(path):
    report = pd.read_csv(path,header = None, decimal = ',', sep=';')

    report = report[[1,6,7]]
    report = report.rename(columns={1: "layer_names", 6: "robustness",7:"f1_score"})

    report = report.sort_values(by=['robustness'])

    most_vulnerable_layers = report.head(5)
    print(most_vulnerable_layers)


    return list(most_vulnerable_layers['layer_names'])


STARTING_POINT_REPORT   = "./reports/yolo/F1_REPORT_BOATS_DEV_PRE_FAT_SINGLE_LAYER.csv"
HARDEN_CHECKPOINTS_PATH = "./MSFAT_checkpoints"


if __name__ == '__main__':

    args = parser.parse_args()
    STARTING_POINT_REPORT = args.starting_point_report

    injection_points = load_report(STARTING_POINT_REPORT)

    print(f'Selected injection points => [{injection_points}]')
    fat_experiment(injection_points=injection_points,args=args)

    

