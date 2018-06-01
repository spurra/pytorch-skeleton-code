# Misc.
from src.util.logger import Logger
import numpy as np
# Python
import pickle
# Pytorch
import torch


# TODO Add comments for all functions. Rename file to utilities

### UTILITY FUNCTIONS


### UTILITY CLASSES

class LogInterface(object):

    def __init__(self, log_dir):
        self.logger =  Logger(log_dir)
        self.registered_models = []
        self.step_dict = {}

    def __get_step(self, name, step):
        if step==-1:
            if name in self.step_dict:
                currStep = self.step_dict[name]
            else:
                currStep = self.step_dict[name] = 0

            self.step_dict[name] += 1
        else:
            currStep = step

        return currStep


    def log_scalar(self, name, item, step=-1):
        currStep = self.__get_step(name, step)
        self.logger.scalar_summary(name, item, currStep)

    def log_scalar_dict(self, item_dict, prefix='', step=-1):
        for name, item in item_dict.items():
            currStep = self.__get_step(prefix + '_' + str(name), step)
            self.log_scalar(prefix + '_' + str(name), item, step)


    def log_vector(self, name, vector, step=-1):
        """Logs each element of a numpy vector as a seperate graph on the same
        plot.
        """
        currStep = self.__get_step(name, step)
        for i,elem in enumerate(np.nditer(vector, flags=["refs_ok"])):
            self.log_scalar(name + "_" + str(i), float(elem), currStep)



    def register_model(self, model):
        self.registered_models.append(model)

    def log_model(self):

        def to_np(val):
            return val.data.cpu().numpy()

        for model in self.registered_models:
            for name, value in model.named_parameters():
                name = name.replace('.', '/')
                currStep = self.__get_step(name, -1)
                self.logger.histo_summary(name, to_np(value), currStep)
                self.logger.histo_summary(name+'/grad', to_np(value.grad),
                    currStep)


    def log_image(self, name, image, step=-1):
        currStep = self.__get_step(name, step)
        self.logger.image_summary(name, [image], currStep)


    def log_image_dict(self, image_dict, step=-1):
        for name, image in image_dict.items():
            self.log_image(name, image, step)

# TODO Add a step/next function which outputs console log and calls log_output
# and saves results and evaluate_functions_batch and returns the results_dict
class Analyzer(object):

    def __init__(self, logger, functions, groups):
        self.function_dict = {}
        self.results_dict = {}
        self.logger = logger
        self.updated = {}
        self.group_names = {}
        self.batch_results_dict = {}

        if groups == -1:
            self.add_group({0 : 'main'})
        else:
            self.add_group(groups)
        self.add_function(functions)


    def add_group(self, groups):
        for g_key, group_name in groups.items():
            self.group_names[g_key] = group_name
            self.results_dict[g_key] = {}
            self.batch_results_dict[g_key] = {}
            self.updated[g_key] = {}


    def add_function(self, functions):
        for f_key, func in functions.items():
            self.function_dict[f_key] = func
            for g_key, _ in self.group_names.items():
                self.results_dict[g_key][f_key] = []
                self.batch_results_dict[g_key][f_key] = 0
                self.updated[g_key][f_key] = 0

    # Evaluates all the functions for a group
    def evaluate_functions(self, input_data, target, group=-1):
        if group == -1:
            group = 0

        result_dict = {}
        for f_key, func in self.function_dict.items():
            result_dict[f_key] = func(input_data, target)
            self.results_dict[group][f_key].append(result_dict[f_key])
            self.updated[group][f_key] = 1

        return result_dict

    def add_functions_batch(self, input_data, target, batch_size, group=-1):
        if group == -1:
            group = 0

        for f_key, func in self.function_dict.items():
            result = func(input_data, target)
            self.batch_results_dict[group][f_key] += result * batch_size


    def evaluate_functions_batch(self, dataset_size, group=-1):
        if group == -1:
            group = 0

        result_dict = {}
        for f_key, func in self.function_dict.items():
            result_dict[f_key] = self.batch_results_dict[group][f_key] / dataset_size
            self.results_dict[group][f_key].append(result_dict[f_key])

            self.batch_results_dict[group][f_key] = 0
            self.updated[group][f_key] = 1

        return result_dict

    def save_results(self, file_name):
        with open(file_name, 'wb') as f:
            to_save = {
                'functions' : list(self.function_dict.keys()),
                'groups' : self.group_names,
                'results' : self.results_dict
            }
            pickle.dump(to_save, f)

    def log_results_all(self):
        pass

    # Logs the latest element of the results list
    def log_update(self):
        for g_key, group_res in self.results_dict.items():
            for f_key, res in group_res.items():
                if self.updated[g_key][f_key] == 1:
                    self.logger.log_scalar(self.group_names[g_key] + '_' + f_key, res[-1])
                    self.updated[g_key][f_key] = 0
