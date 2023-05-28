
class MinMaxPenalty():
    """
    Learn the highest reward/penalty that minimises the probability of reaching bad terminal states 
    Arguments:
        - rmin (optional) (Str): The lower bound for environment rewards
        - rmax (optional) (Str): The upper bound for environment rewards   
    Return:
        - The minmax penalty estimate
    Usage:
    Symlink to the desired folder and import, or copy-paste to where needed
    In training loop: 
        minmaxpenalty = MinMaxPenalty()
        for each step:
            - take an action and get reward and q_value (or just [value] if using policy gradient)
            penalty = minmaxpenalty.update(reward, Q[state])
            if info["unsafe"]:
                reward = penalty
    """

    def __init__(self, rmin=0, rmax=0):
        self.rmin = rmin
        self.rmax = rmax
        self.vmin = self.rmin
        self.vmax = self.rmax
        self.penalty = min([self.rmin, (self.vmin-self.vmax)])
    
    def update(self, reward, value):
        self.rmin = min([self.rmin, reward])
        self.rmax = max([self.rmax, reward])
        self.vmin = min([self.vmin, self.rmin, min(value)])
        self.vmax = max([self.vmax, self.rmax, max(value)])

        self.penalty = min([self.rmin, (self.vmin-self.vmax)])

        return self.penalty
