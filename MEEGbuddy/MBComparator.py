class MBComparator:
    ''' Compares MEEG data generated by M/EEGbuddy for subject level comparisons.'''
    def __init__(self,data_struct,groups=None):
        # the data structure is a dictionary with M/EEGBuddies by name
        self.data_struct = data_struct
        self.groups = groups

    def plotnoreunPhiComparison(self,event,condition,values=None):
        values = self._default_values(values,condition)

    def _default_values(self,values,condition,contrast=False):
        values = None