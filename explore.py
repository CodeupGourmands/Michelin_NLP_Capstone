def get_sentiment(sentiment_df):
    '''Function calculates sentiment score and creates dataframe of scores'''
    # reindex dataframe
    sentiment_df.reset_index(drop =True, inplace=True)
    #create sntiment object
    sia = nltk.sentiment.SentimentIntensityAnalyzer()
    
    # create row id column
    sentiment_df["row_id"] =sentiment_df.index +1
    
    # create subsets
    df_subset = sentiment_df[['row_id', 'clean_text']].copy()
    
    # set up empty dataframe for staging output
    df1=pd.DataFrame()
    df1['row_id']=['99999999999']
    df1['sentiment_type']='NA999NA'
    df1['sentiment_score']=0
    
    # run loop to calculate and save sentiment values
    t_df = df1
    for index,row in df_subset.iterrows():
        scores = sia.polarity_scores(row[1])
        for key, value in scores.items():
            temp = [key,value,row[0]]
            df1['row_id']=row[0]
            df1['sentiment_type']=key
            df1['sentiment_score']=value
            t_df=t_df.append(df1)
    #remove dummy row with row_id = 99999999999
    t_df_cleaned = t_df[t_df.row_id != '99999999999']
    #remove duplicates if any exist
    t_df_cleaned = t_df_cleaned.drop_duplicates()
    # only keep rows where sentiment_type = compound
    t_df_cleaned = t_df[t_df.sentiment_type == 'compound']
    
    df_output = pd.merge(sentiment_df, t_df_cleaned, on='row_id', how='inner')
    
    #generate mean of sentiment_score by period
  
    dfg = df_output.groupby(['language'])['sentiment_score'].mean()
    #create a bar plot
    dfg.plot(kind='bar', title='Sentiment Score', ylabel='Mean Sentiment Score',
         xlabel='',fontsize =20,color=['#beaed4','#f0027f','#7fc97f','#fdc086'])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=22)
    plt.show()
   
    return df_output;



def get_stats_ttest(df):
    '''Function returns statistical T test'''
    java_sem = df[df.language == 'Java']
    javaScript_sem =df[df.language == 'JavaScript']
    stat, pval =stats.levene( java_sem.sentiment_score, javaScript_sem.sentiment_score)
    alpha = 0.05
    if pval < alpha:
        variance = False

    else:
        variance = True
        
    t_stat, p_val = stats.ttest_ind( java_sem.sentiment_score,javaScript_sem.sentiment_score,
                                    equal_var=True,random_state=123)

    
    print (f't_stat= {t_stat}, p_value= {p_val/2}')
    print ('-----------------------------------------------------------')
    
    if (p_val/2) < alpha:
        print (f'We reject the null Hypothesis')
    else:
        print (f'We fail to reject the null Hypothesis.')
