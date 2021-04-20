import copy

class gmp_feature_selector:
    def __init__(self, data, model_eval_params):
        self.data = data
        self.model_eval_params = copy.deepcopy(model_eval_params)

        #these are set after the run() method is called
        self.best_params = None
        self.best_error = -1.
        self.stats = None

    def get_best_params(self):
        if not self.best_params:
            raise RuntimeError("best_params not set. It's possible the run() method has not yet been called")

        return self.best_params

    def get_best_error(self):
        if self.best_error < 0.:
            raise RuntimeError("best_error not set. It's possible the run() method has not yet been called")

        return self.best_error

    def get_stats(self):
        return self.stats
