import pandas as pd

class GPSInfo:
    def __init__(self,gps_file):
        self.gps_file = gps_file
        pass
    
    def match_gps_info(self):
        gps_file = self.gps_file
        # change datatype to datetime
        time = pd.to_datetime(gps_file['date']).dt.strftime("%y-%m-%d %H:%M:%S")
        
        # find idx containing Last GPS information per seconds
        last_gps_list = []
        for i in range(len(time)-1):
            if time[i]==time[i+1]:
                continue
            else:
                last_gps_list.append(int(i))
           
        # DF for Last GPS information per seconds
        last_gps_df = gps_file.iloc[last_gps_list,:]
        
        return last_gps_df

