class EarlyStopping:
    def __init__(self, patience=1000, delta=0):
        """
        Initialize EarlyStopping object.

        Args:
            patience (int, optional): How many steps to wait after last improvement before stopping. Defaults to 1000.
            delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def should_stop(self, val_loss):
        """
        Check if training should stop based on the validation loss.

        Args:
            val_loss (float): Current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop
