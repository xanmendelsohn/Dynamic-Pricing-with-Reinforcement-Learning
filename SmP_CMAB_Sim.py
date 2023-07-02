import numpy as np
import pandas as pd

class cmab_sim:
    """
    Class for online deposit sale simulation 

    """
    # initialization
    def __init__(self, n_choices, bp_conv_fac, parameter_map, feature_list):

        #number of actions
        self.n_choices = n_choices
        #factor used to convert k in (1, ..., n_choices) to its corresponding interest rate in basis points 
        self.bp_conv_fac = bp_conv_fac
        #list of all actions 
        self.all_arms = np.arange(1,self.n_choices+1)
        #mapping table with parameters used to simulate reactions
        self.parm_map = parameter_map
        #list of features/exogenous variables, i.e. the context
        self.feature_list = feature_list
        
    def get_real_reward(self, ordinal_action):
        
        """
        Function to convert ordinal action to monetary reward in basis points
        
        Parameters:
        ordinal_action: integer 
        """
            
        return (10-ordinal_action*self.bp_conv_fac)
        
    def get_reactions(self, X, a):
        
        """
        Function to simulate client reactions
        
        Parameters:
        X: context (array)
        a: actions (array)
        """
        
        df_temp = pd.DataFrame(X, columns = self.feature_list).merge(self.parm_map, on =self.feature_list,  how = "left")
        df_temp["action"] = a
        df_temp["rand"] = np.random.random(df_temp.shape[0])
        df_temp["reaction"] = (df_temp["action"]*self.bp_conv_fac) >= df_temp["parm_u"] + 2*(df_temp["rand"]-0.5)
        df_temp["opt_arm"] = df_temp["parm_u"]/self.bp_conv_fac

        #df_test["reaction"] = np.random.random_sample() < 1/(1+np.exp(-(df_test["action"]/100-df_test["parm_u"])/df_test["parm_s"]))
        #df_test["opt_arm"] = (df_test["parm_u"]+df_test["parm_s"]*np.log(desired_win_rate/(1-desired_win_rate)))*100
        return df_temp["reaction"].to_numpy(), df_temp["opt_arm"].to_numpy()
    
