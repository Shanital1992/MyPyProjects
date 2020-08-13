from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


loc_x = []
loc_y = []
#i is for iterating over all neurons in 'posx' 'posy'
i = 0
annots = loadmat('ExerciseData.mat')

for neuron in annots["spiketrain"]:
    loc_x = []
    loc_y = []

    #converting to a numpy array
    arr = np.array(neuron)
    #extraction of all positions in which spiketrain value = 1
    loc_arr = list(zip(*np.where(arr == 1)))

    for location in loc_arr:
        loc_x.append(annots["posx"][i][location])
        loc_y.append(annots["posy"][i][location])
    #scatter plot spikes vs. x and y position
    plt.plot(loc_x, loc_y, 'o')
    plt.title(i+1)
    plt.xlabel('X position (cm)')
    plt.ylabel('Y position (cm)')
    plt.show()



#obtaining a 599 elements long vector representing firing rates (spikes/sec)
    df_spikes = pd.Series(neuron)
    n = 1000  # chunk row size
    chunk_df = [df_spikes[i:i + n] for i in range(0, df_spikes.size, n)]

    list_of_spiking_rate = []
    for chunk in chunk_df:
        list_of_spiking_rate.append(chunk.values.tolist().count(1))

#obtaining a 599 elements long vector with
    # averaged x positions across each 1000 sample
    df_posx = pd.Series(annots['posx'][i])

    n = 1000  # chunk row size
    chunk_df = [df_posx[i:i + n] for i in range(0, df_posx.size, n)]

    list_of_xpos = []
    for chunk in chunk_df:
        list_of_xpos.append(chunk.mean())


#obtaining a 599 elements long vector with
    # averaged y positions across each 1000 samples

    df_posy = pd.Series(annots['posy'][i])

    n = 1000  # chunk row size
    chunk_df = [df_posy[i:i + n] for i in range(0, df_posy.size, n)]

    list_of_ypos = []
    for chunk in chunk_df:
        list_of_ypos.append(chunk.mean())


    #converting lists into arrays for them to be used as pandas data frame
    x = np.array(list_of_xpos)
    y = np.array(list_of_ypos)
    spikes = np.array(list_of_spiking_rate)

    # forming my data frame for heatmap preperation
    data = pd.DataFrame({'X': x, 'Y': y, 'Firing Rate': spikes})
    data = np.round(data)



    #rescaling the y and x values
    for index in list(range(0, 599)):
        data.at[index,'X'] = 5 * round(data.iloc[index]['X'] / 5)

    for index in list(range(0, 599)):
        data.at[index, 'Y'] = 5 * round(data.iloc[index]['Y'] / 5)


    #plotting a heatmap
    plotty = data.pivot_table(values = 'Firing Rate', index='Y',
                              columns='X').fillna(0)
    ax =sns.heatmap(plotty, xticklabels=2, yticklabels=2, cmap = 'afmhot')
    ax.set(xlabel='X position (cm)', ylabel='Y position (cm)')
    ax.collections[0].colorbar.set_label('Firing Rate (Hz)')
    plt.title(i+1)
    plt.show()



    #converting head direction from radians to degrees, sorting all
    #values into 60 discrete bins, each represents 6 degrees
    df_head = pd.DataFrame(annots['headDirection'][i])
    df_head = np.rad2deg(df_head)
    df_head.columns = ['headDirection']
    df_head['spiketrain'] = df_spikes
    df_head1 = pd.DataFrame()
    #I created this to do both sum and count on spiketrain
    df_head1['headDirection'] = df_head['headDirection']
    df_head1['spiketrain'] = df_head['spiketrain']
    #basically just replicating the dataframe so I can apply
    # both operations on it
    df_head1 = df_head.set_index('headDirection')
    df_head = df_head.set_index('headDirection')
    #I changed headdirection to index so that each bin is attached to
    #the corresponding spiketrain value
    df_head1 = df_head1.groupby(pd.cut(df_head1.index,
                                       np.arange(0,361,6))).count()
    df_head = df_head.groupby(pd.cut(df_head.index, np.arange(0,361,6))).sum()
    df_head['count'] = df_head1['spiketrain']
    #replacing the index from interval type object to int
    new_index1 = list(range(0,360,6))
    new_index = np.array(new_index1)
    df_head['Head Direction'] = new_index
    df_head = df_head.set_index('Head Direction')


    df_head= (df_head['spiketrain']/df_head['count'])*1000
    df_head = pd.DataFrame(df_head)
    df_head.columns = ['Firing Rate']


    plt.plot(df_head)
    plt.title(i+1)
    plt.xlabel('Head Direction (degrees)')
    plt.ylabel('Firing Rate (Hz)')
    plt.show()




    i = i + 1


