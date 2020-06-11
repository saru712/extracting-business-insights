# --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
from scipy.stats import pearsonr
import statistics

def visual_summary(type_, df, col):
    """Summarize the Data using Visual Method.
    
    This function accepts the type of visualization, the data frame and the column to be summarized.
    It displays the chart based on the given parameters.
    
    Keyword arguments:
    type_ -- visualization method to be used
    df -- the dataframe
    col -- the column in the dataframe to be summarized
    """
    plt.figure(figsize=(10,4))
            
    if type_ == "barplot":
        c=df[col].value_counts(normalize=True)
        c.plot(kind='bar')
        plt.xlabel("Country")
        plt.ylabel("Probability")       
    elif type_ == "histplot":
        df[col].hist(density=True,bins=50)
        plt.xlabel("Year")
        plt.ylabel("Probability")              
    else:
        print("Pass either barplot/histplot")
        plt.tight_layout()
        plt.show()



def central_tendency(type_,df, col):
    if type_ == "mean":
        mean=df[col].mean()
        print("Mean of",col,"is",round(mean,2))
    elif type_ == "median":
        median=df[col].median()
        print("Median of",col,"is",round(median,2))            
    elif type_ == "mode":
        mode=df[col].mode()
        print("Mode of",col,"is",mode)
    else:
        print("Pass either mean/median/mode")
    
    """Calculate the measure of central tendency.
    
    This function accepts the type of central tendency to be calculated, the data frame and the required column.
    It returns the calculated measure.
    
    Keyword arguments:
    type_ -- type of central tendency to be calculated
    df -- the dataframe
    col -- the column in the dataframe to do the calculations
    
    Returns:
    cent_tend -- the calculated measure of central tendency
    """


    
    


def measure_of_dispersion(type_, df, col):
    
    if type_ == "mean":
        mean=df[col].mean()
        print("Mean of",col,"is",round(mean,2))
    elif type_ == "Std":
        std=df[col].std()
        print("Standard Deviation of",col,"is",round(std,2))            
    elif type_ == "variance":
        vari=df[col].var()
        print("Variance of",col,"is",vari)
    elif type_=='Range':
        rang=(df[col].max())-(df[col].min())
        print("Range of",col,"is",rang)
    elif type_=='IQR':
        IQR=iqr(df[col], axis=0 , rng=(25, 75), interpolation= 'lower')
        print("IQR of",col,"is",round(IQR,2))
    elif type_=='MAD':
        mad=sum(abs(df[col]-df[col].mean())/len(df[col]))
        print("MAD of",col,"is",round(mad,2))
    elif type_=='CV':
        cv=(df[col].std()/df[col].mean())*100
        print("CV of",col,"is",round(cv,2))
    else:
        print("Pass either mean/Std/variance")

    """Calculate the measure of dispersion.
    
    This function accepts the measure of dispersion to be calculated, the data frame and the required column(s).
    It returns the calculated measure.
    
    Keyword arguments:
    type_ -- type of central tendency to be calculated
    df -- the dataframe
    col -- the column(s) in the dataframe to do the calculations, this is a list with 2 elements if we want to calculate covariance
    
    Returns:
    disp -- the calculated measure of dispersion
    """




def calculate_correlation(type_, df, col1, col2):
    if type_ =='pearson':
        corr, _ = pearsonr(df[col1], df[col2]) 
        print('Pearsons correlation: %.3f' % corr)
    elif type_ =='Spearman':
        newdf = df[[col1,col2]].copy()
        newdf[col1] = newdf[col1].rank()
        newdf[col2] =newdf[col2].rank()
        newdf['d']=newdf[col2] -newdf[col1]
        newdf["d^2"]= newdf['d']**2
        d_square= sum(newdf['d^2'])
        l= len(newdf[col2])
        d= 6*d_square
        spearman = 1- d/(l*(l**2 -1))
        print("Spearman rank correlation is",np.round(spearman,2))
    else:
        print(print("Pass either pearson/Spearman"))

    """Calculate the defined correlation coefficient.
    
    This function accepts the type of correlation coefficient to be calculated, the data frame and the two column.
    It returns the calculated coefficient.
    
    Keyword arguments:
    type_ -- type of correlation coefficient to be calculated
    df -- the dataframe
    col1 -- first column
    col2 -- second column
    
    Returns:
    corr -- the calculated correlation coefficient
    """
    


def calculate_probability_discrete(data, event):
    

    """Calculates the probability of an event from a discrete distribution.
    
    This function accepts the distribution of a variable and the event, and returns the probability of the event.
    
    Keyword arguments:
    data -- series that contains the distribution of the discrete variable
    event -- the event for which the probability is to be calculated
    
    Returns:
    prob -- calculated probability fo the event
    """
    






def event_independence_check(prob_event1, prob_event2, prob_event1_event2):
    """Checks if two events are independent.
    
    This function accepts the probability of 2 events and their joint probability.
    And prints if the events are independent or not.
    
    Keyword arguments:
    prob_event1 -- probability of event1
    prob_event2 -- probability of event2
    prob_event1_event2 -- probability of event1 and event2
    """
    
    


def bayes_theorem(df, col1, event1, col2, event2):
    """Calculates the conditional probability using Bayes Theorem.
    
    This function accepts the dataframe, two columns along with two conditions to calculate the probability, P(B|A).
    You can call the calculate_probability_discrete() to find the basic probabilities and then use them to find the conditional probability.
    
    Keyword arguments:
    df -- the dataframe
    col1 -- the first column where the first event is recorded
    event1 -- event to define the first condition
    col2 -- the second column where the second event is recorded
    event2 -- event to define the second condition
    
    Returns:
    prob -- calculated probability for the event1 given event2 has already occured
    """
    


# Load the dataset
df=pd.read_csv(path)

# Using the visual_summary(), visualize the distribution of the data provided.
# You can also do it at country level or based on years by passing appropriate arguments to the fuction.

visual_summary('barplot',df,'country')
visual_summary('histplot',df,'year')

# You might also want to see the central tendency of certain variables. Call the central_tendency() to do the same.
# This can also be done at country level or based on years by passing appropriate arguments to the fuction.
central_tendency('mean',df,'year')
central_tendency('median',df,'year')
central_tendency('mode',df,'country')

# Measures of dispersion gives a good insight about the distribution of the variable.
# Call the measure_of_dispersion() with desired parameters and see the summary of different variables.
measure_of_dispersion('Std',df,'year')
measure_of_dispersion('variance',df,'year')
measure_of_dispersion('IQR',df,'year')
measure_of_dispersion('Range',df,'year')
measure_of_dispersion('MAD',df,'year')
measure_of_dispersion('CV',df,'year')


# There might exists a correlation between different variables. 
# Call the calculate_correlation() to check the correlation of the variables you desire.
calculate_correlation('pearson',df,'year','exch_usd')
calculate_correlation('Spearman',df,'year','exch_usd')


# From the given data, let's check the probability of banking_crisis for different countries.
# Call the calculate_probability_discrete() to check the desired probability.
# Also check which country has the maximum probability of facing the crisis.  
# You can do it by storing the probabilities in a dictionary, with country name as the key. Or you are free to use any other technique.
df1= df.groupby('country')['banking_crisis'].value_counts().unstack()
df1['Prob of Crisis']=df1['crisis']/(df1['crisis']+df1['no_crisis'])
print(df1)
print("Country with highest probability of crisis is:{}".format(df1['Prob of Crisis'].idxmax()))


# Next, let us check if banking_crisis is independent of systemic_crisis, currency_crisis & inflation_crisis.
# Calculate the probabilities of these event using calculate_probability_discrete() & joint probabilities as well.
# Then call event_independence_check() with above probabilities to check for independence.
# df2= df.groupby('country')['inflation_crises'].value_counts().unstack()
# df2['Prob of in_crisis']=df2[1]/(df2[0]+df1[1])
# print("Probability of inflation crisis is",'\n', df2)

# df3= df.groupby('country')['currency_crises'].value_counts().unstack()
# df3['Prob of c_crisis']=df3[1]/(df2[0]+df2[1])
# print("Probability of currency crisis is",'\n', df3)

# df4= df.groupby('country')['systemic_crisis'].value_counts().unstack()
# df4['Prob of s_crisis']=df4[1]/(df3[0]+df3[1])
# prob_s=df4.fillna(value=0)
# print("Probability of systemic crisis is",'\n', prob_s)

# Calculate the P(A|B)
# Finally, let us calculate the probability of banking_crisis given that other crises (systemic_crisis, currency_crisis & inflation_crisis one by one) have already occured.
# This can be done by calling the bayes_theorem() you have defined with respective parameters.
filt11=(df['systemic_crisis']==1)
filt22=(df['currency_crises']==1)
filt33=(df['inflation_crises']==1)

prob_ =[]
temp=df[filt11]['banking_crisis'].value_counts()[0]/len(df[filt11])
prob_.append(temp)
temp=df[filt22]['banking_crisis'].value_counts()[1]/len(df[filt22])
prob_.append(temp)
temp=df[filt33]['banking_crisis'].value_counts()[1]/len(df[filt33])
prob_.append(temp)
print('The value of prob_is:{}'.format(prob_))



# Code ends


