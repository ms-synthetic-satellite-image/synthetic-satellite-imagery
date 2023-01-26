

class EarlyStopping():
    """
    Early stops the training if FID score does not improve after a given patience period
    with epsilon the minimum change in the FID score that can be viewed as improvement
    """
    def __init__(self, opt):
        """
        Input:
            patience - number of epochs with no improvement after which to stop training
            min_epsilon - the minimum absolute change in target quantity that can be viewed as improvement
            monitor - the target quantity, default to be FID score
            target_dir - the direction for better target quantity, default to be low (lower FID better)
        """
        self.use_earlystop = opt.earlystop
        self.patience = opt.es_patience
        self.min_epsilon = opt.es_epsilon
        self.monitor = opt.es_monitor
        self.target_dir = opt.es_target_dir
        self.count = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if not self.use_earlystop:
            # don't check or set if we don't use early stop; keep it False
            self.early_stop = False  
        elif self.best_score is None:
            self.best_score = score
        else:
            # check wether this epoch gives better target quantity
            is_better = None
            if self.target_dir=='low':
                is_better = score < (self.best_score-self.min_epsilon)
            elif self.target_dir=='high':
                is_better = score > (self.best_score+self.min_epsilon)
            else:
                raise NotImplementedError("Only accept 'low' or 'high' for the target_dir argument!")
            # replace best score if better, else increase count, and if count reaches patience, early stops
            if is_better:
                self.count = 0
                self.best_score = score
            else:
                self.count += 1
                if self.count >= self.patience:
                    self.early_stop = True

    def get_earlystop_state(self):
        '''for tracking early stopping status'''
        return self.count, self.patience
            