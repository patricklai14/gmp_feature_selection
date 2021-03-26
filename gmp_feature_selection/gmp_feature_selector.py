class gmp_feature_selector:
    def __init__(self, data, model_eval_params):
        self.data = data
        self.model_eval_params = model_eval_params

        #these are set after the run() method is called
        self.best_features = None
        self.best_error = -1.
        self.stats = None

    def get_best_features():
        if not self.best_features:
            raise RuntimeError("best_features not set. It's possible the run() method has not yet been called")

        return self.best_features

    def get_best_error():
        if self.best_error < 0.:
            raise RuntimeError("best_error not set. It's possible the run() method has not yet been called")

        return self.best_error

    def get_stats():
        return stats
