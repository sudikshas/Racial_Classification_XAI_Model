import sys
sys.path.append("./src")
import os
from IntegratedGradients import *
from util import *
from test_func import *
import json


if __name__ == '__main__':
    with open("./config/parameters.json") as param:
        parameters = json.load(param)
        integrated_grad = parameters["integrated_grad"]
    param.close()
    
    
    targets = sys.argv[1:]
    
    #############testing##############
    if "test" in targets:
        with open("./test_data/parameters.json") as param:
            parameters = json.load(param)
            integrated_grad = parameters["integrated_grad"]
        param.close()
    #############testing##############
        
    integrated_grad_pic(**integrated_grad)
    
    
    


    
