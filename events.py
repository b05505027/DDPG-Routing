import numpy as np

class Event:
    def __init__(self, time, duration=0):
        self.time = time
        self.duration = duration
class TrafficEvent(Event):
    def __init__(self, time):
        super().__init__(time)

class FailureEvent(Event):
    def __init__(self, time, link_id):
        super().__init__(time)
        self.link_id = link_id

class RecoveryEvent(Event):
    def __init__(self, time, link_id):
        super().__init__(time)
        self.link_id = link_id

class EventManager:
    def __init__(self, lam_t: float, max_broken_links: int, lam_f: float, lam_r: float, time_limit: float):
        """
        Initialize the EventManager class.
        
        :param lam_t: Lambda for traffic events.
        :param max_broken_links: Maximum number of broken links for RF events.
        :param lam_f: Lambda for failure events.
        :param lam_r: Lambda for recovery events.
        :param time_limit: Time limit for the events.
        """
        self.lam_t = lam_t
        self.max_broken_links = max_broken_links
        self.lam_f = lam_f
        self.lam_r = lam_r
        self.time_limit = time_limit

        self.traffic_events = self.generate_traffic_events()
        self.rf_events = self.generate_rf_events()
        self.events = self.prepare_and_sort_events()

        self.event_index = 0  # To keep track of the next event to be fetched

    def prepare_and_sort_events(self):
        # Combine and sort traffic and RF events
        events = sorted(self.traffic_events + self.rf_events, key=lambda event: event.time)

        # Calculate and assign duration for each event
        for i in range(len(events)):
            if i == 0:
                # For the first event, the duration is calculated from time zero
                events[i].duration = events[i].time
            else:
                # For subsequent events, calculate duration as the difference from the previous event
                events[i].duration = events[i].time - events[i - 1].time
        return events

    def generate_time_sequence(self, n: int, lam: float) -> np.ndarray:
        """
        Generate a sequence of time durations based on the exponential distribution.

        :param n: Number of samples to generate.
        :param lam: Lambda for the exponential distribution.
        :return: Numpy array of time durations.
        """
        return np.random.exponential(lam, n)

    def generate_traffic_events(self) -> list:
        """
        Generate traffic events.
        """
        # Estimate the number of events needed
        estimated_event_count = int(self.time_limit / self.lam_t)

        # Generate a larger sequence of event durations
        time_t = self.generate_time_sequence(estimated_event_count, self.lam_t)

        # Cumulative sum and trim to ensure it doesn't exceed time_limit
        time_seq = np.cumsum(time_t)
        time_seq = time_seq[time_seq < self.time_limit]

        traffic_events = []
        for time in time_seq:
            traffic_event = TrafficEvent(time)
            traffic_events.append(traffic_event)

        return traffic_events



    def generate_rf_events(self) -> list:
        """
        Generate RF events.

        :return: List of RF event tuples.
        """
        rf_events = []
        for link_id in range(self.max_broken_links):
            rf_events.extend(self.generate_single_rf_events(link_id))
        return rf_events

    def generate_single_rf_events(self, link_id: int) -> list:
        """
        Generate single RF event for a given link.

        :param link_id: The ID of the link.
        :return: List of RF event tuples.
        """
        # Estimate the number of events needed
        estimated_event_count = int(self.time_limit / (self.lam_f + self.lam_r) * 2)

        # Generate failure and recovery times
        time_f = self.generate_time_sequence(estimated_event_count // 2, self.lam_f)
        time_r = self.generate_time_sequence(estimated_event_count // 2, self.lam_r)

        # Interleave the failure and recovery events
        time_seq = np.empty(time_f.size + time_r.size, dtype=time_f.dtype)
        time_seq[0::2] = time_f
        time_seq[1::2] = time_r

        # Cumulative sum and trim to ensure it doesn't exceed TIME_LIMIT
        time_seq = np.cumsum(time_seq)
        time_seq = time_seq[time_seq < self.time_limit]

        events = []
        for i, time in enumerate(time_seq):
            event = FailureEvent(time, link_id) if i % 2 == 0 else RecoveryEvent(time, link_id)
            events.append(event)

        return events

    def get_event_size(self) -> int:
        """
        Get the size of the combined events.

        :return: Integer size of EVENTS list.
        """
        return len(self.events)

    def get_next_event(self):
        """
        Get the next simulation event.

        :return: The next event tuple or None if all events have been processed.
        """
        if self.event_index >= len(self.events):
            return None
        event = self.events[self.event_index]
        self.event_index += 1
        return event


