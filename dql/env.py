class Env:
    def __init__(self,list_radar_TRs,list_radar_RRs,target,delta_t=0.1):
        self.list_radar_TRs=list_radar_TRs
        self.list_radar_RRs=list_radar_RRs
        self.target=target
        self.delta=delta_t
    
    def getState(self):
        

