import sys
sys.path.append("./src")
import os
from IntegratedGradients import *
from util import *
import json


if __name__ == '__main__':
    with open("./config/age_parameters.json") as param:
        parameters = json.load(param)
        integrated_grad = parameters["integrated_grad"]
    param.close()

    integrated_grad_pic(**integrated_grad)
    
    
    


    
