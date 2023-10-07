from matplotlib import pyplot as plt
import numpy as np
from events import generate_traffic_events, generate_rf_events



# generate traffic events
traffic_events = generate_traffic_events(lam_t=5, time_limit=1000000)
# generate rf events
events_f_1_over_5000  = traffic_events
events_f_1_over_250 = traffic_events
for i in range(7):
    events_f_1_over_5000.extend(generate_rf_events(link_id=i, lam_f=5000, lam_r=100, time_limit=1000000))
    events_f_1_over_250.extend(generate_rf_events(link_id=i, lam_f=250, lam_r=100, time_limit=1000000))

# sort the events based on the occurrence time
events_f_1_over_5000.sort(key=lambda x: x[0])
events_f_1_over_250.sort(key=lambda x: x[0])

# calculate the duration of each event
durations_f_1_over_5000 = []
durations_f_1_over_250 = []
for i in range(len(events_f_1_over_5000)):
    if i == len(events_f_1_over_5000) - 1:
        durations_f_1_over_5000.append(1000000 - events_f_1_over_5000[i][0])
    else:
        durations_f_1_over_5000.append(events_f_1_over_5000[i+1][0] - events_f_1_over_5000[i][0])
for i in range(len(events_f_1_over_250)):
    if i == len(events_f_1_over_250) - 1:
        durations_f_1_over_250.append(1000000 - events_f_1_over_250[i][0])
    else:
        durations_f_1_over_250.append(events_f_1_over_250[i+1][0] - events_f_1_over_250[i][0])




# draw the histogram of the event durations
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))
#text size
plt.rcParams.update({'font.size': 15})
plt.hist(durations_f_1_over_5000, bins=600, alpha=0.5, color='green', label='Event_f_1_over_5000')
plt.hist(durations_f_1_over_250, bins=600, linewidth=0.3, edgecolor='red', fill=False, label='Event_f_1_over_250')
plt.legend(loc='upper right')
plt.xlim(0, 20)
plt.xlabel('Duration (ms)')
plt.ylabel('Frequency')
plt.title('Histogram of Event Durations')
plt.savefig('histogram.png')