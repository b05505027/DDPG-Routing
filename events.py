import numpy as np
import torch

# generate a sequence of time duraion based on the exponential distribution
def generate_time_sequence(n, lam) -> np.ndarray:
    return np.random.exponential(lam, n)

def generate_traffic_events(lam_t, time_limit) -> list:
    time_t = generate_time_sequence(50, lam_t)
    while time_t.sum() < time_limit:
        time_t = np.append(time_t, generate_time_sequence(50, lam_t))
    time_seq = np.cumsum(time_t)
    time_seq = time_seq[time_seq < time_limit] # remove the events that exceed the time limit
    time_seq = time_seq.tolist()
    for i in range(len(time_seq)):
        time_seq[i] = (time_seq[i], f"traffic_{i+1}")
    time_seq = [(0, f"traffic_0")] + time_seq
    return time_seq
    


def generate_rf_events(link_id, lam_f, lam_r, time_limit) -> list:
    link_id = link_id # link id
    lam_f = lam_f # avg duration for the failure event
    lam_r = lam_r # avg duration  for the recovery event

    # generate the time sequence for the failure and recovery events
    time_f = generate_time_sequence(50, lam_f)
    time_r = generate_time_sequence(50, lam_r)

    while time_f.sum() + time_r.sum() < time_limit:
        time_f = np.append(time_f, generate_time_sequence(50, lam_f))
        time_r = np.append(time_r, generate_time_sequence(50, lam_r))

    # interleave the failure and recovery events
    time_seq = np.empty((time_f.size + time_r.size,), dtype=time_f.dtype)
    time_seq[0::2] = time_f
    time_seq[1::2] = time_r

    # add the time sequence to the start time
    time_seq = np.cumsum(time_seq)
    time_seq = time_seq[time_seq < time_limit] # remove the events that exceed the time limit
    time_seq = time_seq.tolist()
    for i in range(len(time_seq)):
        time_seq[i] = (time_seq[i], f"failure_{link_id}") if i % 2 == 0 else (time_seq[i], f"recovery_{link_id}")
    
    return time_seq

events = []

num_of_links = 7
for i in range(num_of_links):
    events.extend(generate_rf_events(link_id=i, lam_f=100, lam_r=10, time_limit=500))


events.extend(generate_traffic_events(10, 500))
# sort the events based on the occurrence time
events.sort(key=lambda x: x[0])
for e in events:
    print(e)

