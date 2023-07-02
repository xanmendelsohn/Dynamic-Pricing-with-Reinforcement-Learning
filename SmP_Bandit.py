import numpy as np
import pandas as pd
from copy import deepcopy

class Smart_Bandit:
    """
    Class for implementation online contextual multi-arm bandit algorithm,
    Implements Online BTS, Epsilon-Greedy with d=1 and Softmax algorithm

    """
    
    # initialization
    def __init__(self, algorithm, epsilon, n_boots, quantile, n_choices, base_model, feature_list, reward_fct, seed):
    
        #algorithm: "epsilon_greedy", "softmax", "thompson", "ucb"  
        self.algorithm = algorithm
        #epsilon for epsilon greedy algorithm
        self.epsilon = epsilon
        #number of bootstrap models; should be greater than the batch size
        self.n_boots = n_boots
        #numeration of bootstrap models
        self.boot_num = np.arange(0,self.n_boots)
        #quantile used for ucb algorithm 
        self.quantile = quantile
        #base classifier 
        self.base_model = base_model
        #number of actions
        self.n_choices = n_choices
        #list of all actions 
        self.all_arms = np.arange(1,self.n_choices+1)
        #list of features/exogenous variables, i.e. the context
        self.feature_list = feature_list
        #function which converts ordinal action to monetary reward in basis points
        self.reward_fct = reward_fct
        #set seed
        self.seed = seed
        
        self.bmodels = []
        for i in range(self.n_boots):
            self.bmodels.append(deepcopy(self.base_model)) 
    
    def fit(self, X, a, r):
        
        """
        Function to fit base classifier
        Can be changed to implement partial_fit for online algorithm
        
        Parameters:
        X: context (array)
        a: actions (array)
        r: reactions (array)
        """
        
        df_temp = pd.DataFrame(X, columns = self.feature_list)
        df_temp["action"] = a
        df_temp["reaction"] = r
        
        #weight observations depending on algorithm
        if self.algorithm in ["thompson" , "ucb"]:
            for j in range(self.n_boots):
                np.random.seed(j*self.seed)
                w = np.random.gamma(shape = 1, scale = 1, size = df_temp.shape[0])
                self.bmodels[j].fit(df_temp.drop(["reaction"], axis = 1).values, df_temp["reaction"].values, sample_weight = w)
        else:
            w = np.ones_like(a, dtype=np.int8) 
            self.base_model.fit(df_temp.drop(["reaction"], axis = 1), df_temp["reaction"], sample_weight = w)
        
    def predict_proba_true(self, X):
        
        """
        Function to get win probability as float
        
        Parameters:
        X: context (array)
        """

        return self.base_model.predict_proba(X)[:, 1]
    
    def predict_proba_true_model(self, X, model):
        
        """
        Function to get win probability as float from list of bootstrap models 
        
        Parameters:
        X: context (array)
        model: specified model [0:n_boot -1] (int) 
        """
            
        return self.bmodels[model].predict_proba(X.values.reshape(1, -1))[:, 1]
    
    def infer_observations(self, X, a, r):
        
        """
        Function to infer observations, i.e.:
            if offer accepted with x rate update also >=x to 'accepted'
            if offer declined with x rate update also <=x to 'declined'
        
        Parameters:
        X: context (array)
        a: actions (array)
        r: reactions (array)
        """
        
        df_temp = pd.DataFrame(X, columns = self.feature_list) 
        df_temp["action_ind"] = a
        df_temp["reaction"] = r
        df_temp["action"]= df_temp.apply(lambda row: [a for a in self.all_arms if a >= row["action_ind"]] if row["reaction"] else
                                       [a for a in self.all_arms if a <= row["action_ind"]], axis=1)
        df_temp = df_temp.drop("action_ind", axis=1).explode("action")
        temp_X = (df_temp.drop(["reaction","action"], axis=1)).to_numpy()
        temp_r = df_temp["reaction"].to_numpy()
        temp_a = df_temp["action"].to_numpy()

        return temp_X, temp_a, temp_r
    
    def get_actions_epsilon(self, X):
                
        """
        Function to produce actions based on the Epsilon Greedy Algorithm
        
        Parameters:
        X: context (array)
        """
        
        #array to pandas
        df_temp = pd.DataFrame(X, columns = self.feature_list)
        #explode dataset with all possible actions 
        df_temp["action"]= df_temp.apply(lambda row: [a for a in self.all_arms], axis=1)
        df_temp["obs"] = np.arange(0, df_temp.count()[0])
        df_temp = df_temp.explode("action")
        #determine 'win' probability for each action
        df_temp["p"] = self.predict_proba_true(df_temp.drop("obs", axis=1))
        #determine reward for each action
        df_temp["reward"] = df_temp.apply(lambda row: self.reward_fct(row["action"]), axis=1)
        #determine expected reward for each action
        df_temp["exp_reward"] = df_temp["p"]*df_temp["reward"]
        #determine action with max expected reward
        df_temp['max'] = df_temp.groupby(self.feature_list+["obs"])['exp_reward'].transform('max')
        #get best arm
        best_arm = pd.DataFrame(df_temp[df_temp["exp_reward"] == df_temp["max"]]["action"], columns = ["action"])
        #choose action based on ranom number and epsilon
        best_arm["choice"] = best_arm["action"].apply(lambda row: row if np.random.random()>self.epsilon else np.random.choice(np.delete(self.all_arms, row)))
        
        return best_arm["choice"].to_numpy()
    
    def get_actions_thompson(self, X):
        
        """
        Function to produce actions based on the Thompson Sampling Bootstrap Algorithm
        
        Parameters:
        X: context (array)
        """
        #array to pandas
        df_temp = pd.DataFrame(X, columns = self.feature_list)
        #explode dataset with all possible actions
        df_temp["action"]= df_temp.apply(lambda row: [a for a in self.all_arms], axis=1)
        #select bootstrap model at random uniformly per observation
        df_temp["model"] = np.random.randint(self.n_boots, size= df_temp.count()[0])
        df_temp["obs"] = np.arange(0, df_temp.count()[0])
        df_temp = df_temp.explode("action")
        df_temp=df_temp.reset_index(drop=True)
        df_temp=df_temp[self.feature_list+["action","model","obs"]]
        #determine 'win' probability for each action
        df_temp["p"] = df_temp.apply(lambda row: self.predict_proba_true_model(row.drop(["model","obs"]), row["model"])[0], axis=1)
        #determine reward for each action
        df_temp["reward"] = df_temp.apply(lambda row: self.reward_fct(row["action"]), axis=1)
        #determine expected reward for each action
        df_temp["exp_reward"] = df_temp["p"]*df_temp["reward"]
        #determine action with max expected reward
        df_temp['max'] = df_temp.groupby(self.feature_list+["obs"])['exp_reward'].transform('max')
        #get best action
        best_arm = pd.DataFrame(df_temp[df_temp["exp_reward"] == df_temp["max"]]["action"], columns = ["action"])

        return best_arm["action"].to_numpy()
    
    def get_actions_ucb(self, X):
        
        """
        Function to produce actions based on the Bootstrap Upper Confidence Bound Algorithm
        
        Parameters:
        X: context (array)
        """
        #array to pandas
        df_temp = pd.DataFrame(X, columns = self.feature_list)
        #explode dataset with all possible actions
        df_temp["action"]= df_temp.apply(lambda row: [a for a in self.all_arms], axis=1)
        #select bootstrap model at random uniformly per observation
        df_temp["model"] = df_temp.apply(lambda row: [b for b in self.boot_num], axis=1)
        df_temp["obs"] = np.arange(0, df_temp.count()[0])
        df_temp = df_temp.explode("action")
        df_temp = df_temp.explode("model")
        df_temp=df_temp.reset_index(drop=True)
        df_temp=df_temp[self.feature_list+["action","model","obs"]]
        #determine 'win' probability for each action
        df_temp["p"] = df_temp.apply(lambda row: self.predict_proba_true_model(row.drop(["model","obs"]), row["model"])[0], axis=1)
        #determine reward for each action
        df_temp["reward"] = df_temp.apply(lambda row: self.reward_fct(row["action"]), axis=1)
        #determine expected reward for each action
        df_temp["exp_reward"] = df_temp["p"]*df_temp["reward"]
        #determine upper confidence bound of expected reward for each action
        df_temp = df_temp.groupby(self.feature_list+["action","obs"])['exp_reward'].quantile(self.quantile).reset_index(name = 'ucb_exp_reward')
        #determine action with max expected reward
        df_temp['max'] = df_temp.groupby(self.feature_list+["obs"])['ucb_exp_reward'].transform('max')
        #get best action
        best_arm = pd.DataFrame(df_temp[df_temp["ucb_exp_reward"] == df_temp["max"]]["action"], columns = ["action"])

        return best_arm["action"].to_numpy()
    
    def get_actions_softmax(self, X, recency_weight):
        
        """
        Function to produce actions based on the Softmax Algorithm
        
        Parameters:
        X: context (array)
        recency_weight: (array) weight 'm' applied to observations
        """
        
        #array to pandas
        df_temp = pd.DataFrame(X, columns = self.feature_list)
        #recency weight
        df_temp["w"] = recency_weight
        #number the observations
        df_temp["obs"] = np.arange(0, df_temp.count()[0])
        #explode dataset with all possible actions 
        df_temp["action"]= df_temp.apply(lambda row: [a for a in self.all_arms], axis=1)
        df_temp = df_temp.explode("action")
        #determine 'win' probability for each action
        df_temp["p"] = self.predict_proba_true(df_temp.drop(["w","obs"], axis = 1))
        #apply inverse sigmoid function
        df_temp["sigmoid"] = np.log(df_temp["p"]/(1-df_temp["p"]))
        #determine reward for each action
        df_temp["reward"] = df_temp.apply(lambda row: self.reward_fct(row["action"]), axis=1)
        #softmax summand
        df_temp["softmax_summand"] = np.exp((df_temp["w"]*df_temp["sigmoid"]*df_temp["reward"]).astype(float)) #/sum(np.exp((df_temp["sigmoid"]*df_temp["reward"]).astype(float)))
        #softmax denominator
        df_temp["softmax_denominator"] = df_temp.groupby(self.feature_list+["obs"])['softmax_summand'].transform('sum')
        #softmax value
        df_temp["softmax"] = df_temp["softmax_summand"]/df_temp["softmax_denominator"]
        
        #select action randomly according to softmax probability
        fn = lambda obj: np.random.choice(obj["action"].to_numpy(), p=obj["softmax"].to_numpy())
        
        choice_map = (df_temp.groupby(self.feature_list+["obs"], as_index=False).apply(fn)).rename(columns = {None:"choice"})
        df_temp = df_temp.merge(choice_map, on=self.feature_list+["obs"], how="left")
        
        return df_temp[df_temp["action"] == df_temp["choice"]]["action"].to_numpy()
    

    def get_actions(self, X, recency_weight):
        
        """
        Wrapper function of the other get_action functions
        
        Parameters:
        X: context (array)
        recency_weight: (array) weight 'm' applied to observations
        """
        if self.algorithm == "thompson":
            x = self.get_actions_thompson(X)
        elif self.algorithm == "ucb":
            x = self.get_actions_ucb(X)
        elif self.algorithm == "softmax":
            x = self.get_actions_softmax(X, recency_weight)
        else:
            x = self.get_actions_epsilon(X)
            
        return x 
        
#Patched classifier to converge to given success probability
# from sklearn.linear_model import SGDClassifier
# #patch y gets larger the closer we are to desired win rate
# class SGDClassifierWithNormalization(SGDClassifier):
#     def __init__(self, loss, desired_win_rate, eps = 0.0001, normalizing_factor = 1e-3, **kwargs):
#         super().__init__(loss, **kwargs)
#         self.desired_win_rate = desired_win_rate
#         self.eps = eps
#         self.normalizing_factor = normalizing_factor
        
#     def predict_proba(self,X):
#         probs = super().predict_proba(X)[:, 1]
#         y = 1/np.abs(probs - self.desired_win_rate + self.eps)
#         y = 1- np.exp(-y*self.normalizing_factor)
        
#         return np.vstack((1-y,y)).T