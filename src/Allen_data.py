import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# PROJECT EPOCH

output_dir = 'outputs'
input_dir = 'data'
manifest_path = os.path.join(input_dir, 'manifest.json')
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
probes = cache.get_probes()
channels = cache.get_channels()
analysis_metrics1 = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1',
                                                                    amplitude_cutoff_maximum=np.inf,
                                                                    presence_ratio_minimum=-np.inf,
                                                                    isi_violations_maximum=np.inf)

# analysis_metrics2 = cache.get_unit_analysis_metrics_by_session_type('functional_connectivity',
#                                                                     amplitude_cutoff_maximum=np.inf,
#                                                                     presence_ratio_minimum=-np.inf,
#                                                                     isi_violations_maximum=np.inf)
# all_units_with_metrics = pd.concat([analysis_metrics1, analysis_metrics2], sort=False)
all_units_with_metrics = analysis_metrics1


# unit_count2 = all_metrics[(all_metrics.isi_violations < 0.5) &
#                           (all_metrics.amplitude_cutoff < 0.1) &
#                           (all_metrics.presence_ratio >= 0.95)]

sessions_filtered = sessions[(sessions['ecephys_structure_acronyms'].
                              apply(lambda x: {'VISp', 'VISl', 'VISal'}.issubset(set(x))))
                             & (sessions['session_type'] == 'brain_observatory_1.1')]

structure_list = ['VISp', 'VISl', 'VISal']
filtered_units = all_units_with_metrics[
    (all_units_with_metrics['isi_violations'] < 0.5) &
    (all_units_with_metrics['amplitude_cutoff'] < 0.1) &
    all_units_with_metrics['ecephys_session_id'].isin(sessions_filtered.index)
    & all_units_with_metrics['ecephys_structure_acronym'].isin(structure_list)]
summary = filtered_units.groupby(['ecephys_session_id', 'ecephys_structure_acronym']).size().reset_index(name='count')
pivoted_df = pd.pivot_table(summary, index='ecephys_session_id', columns='ecephys_structure_acronym', values='count')
# select the index of the row with the highest VISp count in the pivoted_df
session_to_analyze = pivoted_df['VISl'].idxmax()
#select units from all_units_with_metrics where ecephys_session_id is session_to_analyze and ecephys_structure_acronym is in structure_list
all_units_with_metrics_filtered = all_units_with_metrics[(all_units_with_metrics['ecephys_session_id'] == session_to_analyze) &
                                                         all_units_with_metrics['ecephys_structure_acronym'].isin(structure_list)]

# SESSION EPOCH

session_data = cache.get_session_data(session_to_analyze,
                                      isi_violations_maximum=np.inf,
                                      amplitude_cutoff_maximum=np.inf,
                                      presence_ratio_minimum=-np.inf)

# session_data = cache.get_session_data(session_to_analyze,
#                                       isi_violations_maximum=0.5,
#                                       amplitude_cutoff_maximum=0.1,
#                                       presence_ratio_minimum=-np.inf)

stimulus_blocks = session_data.get_stimulus_epochs()
stimulus_blocks = stimulus_blocks[stimulus_blocks['stimulus_name'].isin(['drifting_gratings', 'spontaneous'])]

presentations = session_data.get_stimulus_table(['drifting_gratings', 'spontaneous'])
# SAME CODE AS ABOVE
# presentations = session_data.stimulus_presentations
# presentations = presentations[presentations['stimulus_name']. \
#     isin(['drifting_gratings', 'spontaneous'])]
# presentations = presentations.drop(presentations.columns[presentations.eq('null').all()], axis=1)
presentations = presentations.drop(['contrast', 'phase', 'size', 'spatial_frequency'], axis=1)
#group presentations by stimulus name, stimulus condition id, and stimulus block, and count the number of presentations
presentations_count = presentations.groupby(['stimulus_name', 'stimulus_condition_id']). \
    size().reset_index(name='num_trials')
#There is a drifting grating condition (stimulus_condition_id 273) with 30 trials. For this, both orientation and temporal frequency are null
# Calculate the lower and upper "confidence intervals" for the x axis
x_err = [presentations['duration']/2, presentations['duration']/2]
plt.errorbar(presentations['start_time'] + x_err[0],
             presentations['stimulus_name'].map({'spontaneous': 0, 'drifting_gratings': 1}).astype(str),
             xerr=x_err, ecolor='black', linestyle='')
# Add labels to the plot
plt.xlabel('Start Time')
plt.ylabel('Stimulus Block')
# Show the plot
plt.show()

conditions = session_data.stimulus_conditions
conditions = conditions[conditions['stimulus_name']. \
    isin(['drifting_gratings', 'spontaneous'])]
conditions = conditions.drop(conditions.columns[conditions.eq('null').all()], axis=1)
conditions = conditions.drop(['contrast', 'mask', 'opacity', 'phase', 'size',
                              'spatial_frequency', 'units', 'color_triplet'], axis=1)

#DRIFTING GRATINGS

# from the presentations table, get all stimulus presentation ids where stimulus_name is 'drifting_gratings'
drifting_gratings_presentation_ids = presentations[presentations['stimulus_name'] == 'drifting_gratings']
drifting_gratings_spike_times = \
    session_data.presentationwise_spike_times(stimulus_presentation_ids=drifting_gratings_presentation_ids.index.values). \
    reset_index()
drifting_gratings_spike_times_count = \
    drifting_gratings_spike_times.groupby(['unit_id','stimulus_presentation_id']). \
    size().reset_index(name='unit_spike_count_on_trial')
#select all units with more than 100 spikes
drifting_gratings_spike_times_count = \
    drifting_gratings_spike_times_count[drifting_gratings_spike_times_count['unit_spike_count_on_trial'] > 10]
#from a, select the minimum time since stimulus presentation onset for each unit_id, stimulus_presentation_id pair
unit_min_time_since_presentation_onset = \
    drifting_gratings_spike_times.groupby(['unit_id', 'stimulus_presentation_id']) \
    .apply(lambda x: x['time_since_stimulus_presentation_onset'].min()). \
        reset_index(name='unit_min_time_since_presentation_onset')
unit_min_time_since_presentation_onset = \
    unit_min_time_since_presentation_onset[
        unit_min_time_since_presentation_onset['unit_min_time_since_presentation_onset'] < 0.5]
#select the rows in the spike_times table where both the unit_id and stimulus_presentation_id are in the spike_times_count table
drifting_gratings_spike_times = \
    drifting_gratings_spike_times.merge(drifting_gratings_spike_times_count, on=['unit_id', 'stimulus_presentation_id']). \
    merge(unit_min_time_since_presentation_onset, on=['unit_id', 'stimulus_presentation_id']). \
    merge(presentations, on='stimulus_presentation_id'). \
    merge(all_units_with_metrics_filtered['ecephys_structure_acronym'].reset_index().
          rename(columns={'ecephys_unit_id': 'unit_id'}), on='unit_id')
unit_id_map = {unit_id: idx for idx, unit_id in enumerate(drifting_gratings_spike_times['unit_id'].unique())}
drifting_gratings_spike_times['unit_id_int'] = drifting_gratings_spike_times['unit_id'].map(unit_id_map)
# from the spike times table, plot the index as the x axis, and the unit id as the y axis. dont show the y axis text
ax = drifting_gratings_spike_times.plot(x='time_since_stimulus_presentation_onset', y='unit_id_int', kind='scatter', s=1, yticks=[])
plt.show()

# SUMMARY

# from the drifting_gratings_spike_times table, count the number of unique unit ids by ecephys_structure_acronym
region_counts = \
    drifting_gratings_spike_times.groupby('ecephys_structure_acronym')['unit_id']. \
    nunique().reset_index(name='num_units')

# FORMATTED SPIKE TIMES

time_bin_edges = np.linspace(-0.05, 2, 2050) # 2050 edges, 2049 bins
unit_ids = drifting_gratings_spike_times['unit_id'].unique()
#Dimensions: (stimulus_presentation_id, time_bin, unit_id)
drifting_gratings_spike_counts = session_data. \
    presentationwise_spike_counts(stimulus_presentation_ids=drifting_gratings_presentation_ids.index.values,
                                 bin_edges=time_bin_edges, unit_ids=unit_ids)
#plot data for one stimulus presentation id
drifting_gratings_spike_counts.sel(stimulus_presentation_id=31007).plot(x='time_bin', y='unit_id')


#in the dataArray above, sum over the stimulus_presentation_id dimension
drifting_gratings_spike_hist = drifting_gratings_spike_counts.sum(dim='stimulus_presentation_id'). \
    to_dataframe(name='spike_counts'). \
    pivot_table(index='time_relative_to_stimulus_onset', columns='unit_id', values='spike_counts')
#in the dataframe above, remove all columns for which the maximum value is less than 5
drifting_gratings_spike_hist = drifting_gratings_spike_hist.loc[:550, drifting_gratings_spike_hist.max() > 2]

# Iterate over sets of 10 columns in drifting_gratings_spike_hist
for i in range(0, drifting_gratings_spike_hist.shape[1], 10):
    # Select the next set of 10 columns
    a_subset = drifting_gratings_spike_hist.iloc[:550, i:i+10]
    # Create a figure with 10 subplots
    fig, axs = plt.subplots(nrows=10, sharex=True)
    # Plot each column of a_subset on a separate subplot
    for j, col in enumerate(a_subset.columns):
        axs[j].plot(a_subset.index, a_subset[col])
    # Add labels to the plot
    fig.suptitle(f'Columns {i+1}-{i+10} of a')
    plt.xlabel('Index')
    plt.ylabel('Value')
    # Save the plot to output_dir
    fig.savefig(os.path.join(output_dir, f'columns_{i + 1}-{i + 10}.png'))


# HARD CODED STUFF

drifting_gratings_spike_times_lite = \
    drifting_gratings_spike_times[drifting_gratings_spike_times['stimulus_presentation_id'].isin([31007])]
unit_id_map = {unit_id: idx for idx, unit_id in enumerate(drifting_gratings_spike_times_lite['unit_id'].unique())}
drifting_gratings_spike_times_lite['unit_id_int'] = \
    drifting_gratings_spike_times_lite['unit_id'].map(unit_id_map).sample(frac=1).values
ax = drifting_gratings_spike_times_lite.plot(x='time_since_stimulus_presentation_onset', y='unit_id_int', kind='scatter', s=1, yticks=[])
plt.show()





spike_times_lite = spike_times[(spike_times['stimulus_condition_id']==246) & (spike_times['stimulus_block']==2)]
trial_unit_count = spike_times_lite.groupby(['stimulus_presentation_id'])['unit_id'].unique(). \
    reset_index().rename(columns={'unit_id': 'trial_unit_count'})
# find the units that are common to all rows in the trial_unit_count column of the trial_unit_count table
common_units = set.intersection(*trial_unit_count['trial_unit_count'].apply(set))


#SPONTANEOUS ACTIVITY. Will come back to this later

spontaneous_presentation_ids = presentations[presentations['stimulus_name'] == 'spontaneous']
spontaneous_presentation_ids = \
    spontaneous_presentation_ids. \
        drop(spontaneous_presentation_ids.columns[spontaneous_presentation_ids.eq('null').all()], axis=1)
#select the row of spontaneous_presentation_ids where the duration is maximum
spontaneous_presentation_ids = \
    spontaneous_presentation_ids[spontaneous_presentation_ids['duration'] == spontaneous_presentation_ids['duration'].max()]
spontaneous_spike_times = \
    session_data.presentationwise_spike_times(stimulus_presentation_ids=spontaneous_presentation_ids.index.values). \
    reset_index()
#filter spontaneous_spike_times by the unit ids in the drifting_gratings_spike_time
spontaneous_spike_times = \
    spontaneous_spike_times[spontaneous_spike_times['unit_id'].isin(drifting_gratings_spike_times['unit_id'].unique())]
spontaneous_spike_times_count = spontaneous_spike_times.groupby(['unit_id','stimulus_presentation_id']). \
    size().reset_index(name='unit_spike_count_on_trial')
#select all units with more than 100 spikes
spontaneous_spike_times_count = spontaneous_spike_times_count[spontaneous_spike_times_count['unit_spike_count_on_trial'] > 10]
#from a, select the minimum time since stimulus presentation onset for each unit_id, stimulus_presentation_id pair
unit_min_time_since_presentation_onset = \
    spontaneous_spike_times.groupby(['unit_id', 'stimulus_presentation_id']) \
    .apply(lambda x: x['time_since_stimulus_presentation_onset'].min()). \
        reset_index(name='unit_min_time_since_presentation_onset')
unit_min_time_since_presentation_onset = \
    unit_min_time_since_presentation_onset[
        unit_min_time_since_presentation_onset['unit_min_time_since_presentation_onset'] < 0.5]
#select the rows in the spike_times table where both the unit_id and stimulus_presentation_id are in the spike_times_count table
drifting_gratings_spike_times = drifting_gratings_spike_times.merge(drifting_gratings_spike_times_count, on=['unit_id', 'stimulus_presentation_id']). \
    merge(unit_min_time_since_presentation_onset, on=['unit_id', 'stimulus_presentation_id']). \
    merge(presentations, on='stimulus_presentation_id', how='left')
unit_id_map = {unit_id: idx for idx, unit_id in enumerate(drifting_gratings_spike_times['unit_id'].unique())}
drifting_gratings_spike_times['unit_id_int'] = drifting_gratings_spike_times['unit_id'].map(unit_id_map)
# from the spike times table, plot the index as the x axis, and the unit id as the y axis. dont show the y axis text
ax = drifting_gratings_spike_times.plot(x='time_since_stimulus_presentation_onset', y='unit_id_int', kind='scatter', s=1, yticks=[])
plt.show()
#group spike_times by stimulus_presentation_id and select distict unit ids
spontaneous_trial_unit_count = spontaneous_spike_times. \
    groupby(['stimulus_presentation_id'])['unit_id'].unique().apply(len). \
    reset_index().rename(columns={'unit_id': 'trial_unit_count'})
ax = spontaneous_trial_unit_count.plot(y='trial_unit_count', kind='hist', bins=200)
plt.show()

spontaneous_spike_times_lite = \
    spontaneous_spike_times[spontaneous_spike_times['stimulus_presentation_id'].isin([53275])]
unit_id_map = {unit_id: idx for idx, unit_id in enumerate(spontaneous_spike_times_lite['unit_id'].unique())}
spontaneous_spike_times_lite['unit_id_int'] = \
    spontaneous_spike_times_lite['unit_id'].map(unit_id_map).sample(frac=1).values
ax = spontaneous_spike_times_lite.plot(x='time_since_stimulus_presentation_onset', y='unit_id_int', kind='scatter', s=1, yticks=[])
plt.show()

#UNIT TRIAL COUNTS. May not be important

#group spike_times by stimulus_presentation_id and select distict unit ids
drifting_gratings_trial_unit_count = \
    drifting_gratings_spike_times. \
    groupby(['stimulus_presentation_id'])['unit_id'].unique().apply(len). \
    reset_index().rename(columns={'unit_id': 'trial_unit_count'})
ax = drifting_gratings_trial_unit_count.plot(y='trial_unit_count', kind='hist', bins=200)
plt.show()
#There apparently are not the same number of units present in a single session across every
#On further thought, this may actually not matter we are building the model top down
#from latent variables, as opposed to from bottom up as I did for my ADA.
drifting_gratings_unit_trial_count = \
    drifting_gratings_spike_times. \
    groupby(['unit_id'])['stimulus_presentation_id'].unique().apply(len). \
    reset_index().rename(columns={'stimulus_presentation_id': 'unit_trial_count'})
ax = drifting_gratings_unit_trial_count.plot(y='unit_trial_count', kind='hist', bins=200)
plt.show()
#We see from drifting_gratings_unit_trial_count that there are some units that were recorded for only
#1 trial thoughout the entire session. Again, this may not matter.
#select unique 'stimulus_condition_id', 'stimulus_presentation_id', 'unit_id' rows fron the drifting_gratings_spike_times table
drifting_gratings_unit_trial_count_per_stimulus_condition = \
    drifting_gratings_spike_times[['stimulus_condition_id', 'stimulus_presentation_id', 'unit_id']]. \
        drop_duplicates(). \
    groupby(['stimulus_condition_id', 'unit_id']).size().reset_index(name='trial_count')
#We see from drifting_gratings_unit_trial_count_per_stimulus_condition that there are some units that were recorded
#for only one trial within a stimulus condition. Again, this may not matter.