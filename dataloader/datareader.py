import pandas as pd


class DataReader():
    def __init__(self, path, sep, startTimestamp) -> None:
        self.df = pd.read_csv(path, sep=sep, index_col=False)
        self.timestamp_name = 'Timestamp' if 'Timestamp' in self.df.columns else 'Time'

        self._cut_by_timestamp(startTimestamp)
        self.df = self.df.dropna(axis=1, how='all')

    def get_data_for_timestamp(self, timestamp):
        idx = self._find_next_lower_neigbour(timestamp)
        return self.df.iloc[idx - 40:idx] \
            .iloc[::-1] \
            .drop([self.timestamp_name], axis=1) \
            .replace({'NONE': 0, 'Target': 1, 'Distractor': 2}) \
            .values \
            .flatten()

    def get_label(self, timestamp):
        idx = self._find_next_lower_neigbour(timestamp + 150)
        return self.df.iloc[idx].drop([self.timestamp_name]).values.flatten()

    def _cut_by_timestamp(self, timestamp):
        # - 10 because data is sampled every 10ms and we want the first element before timestamp in it
        self.df = self.df[self.df[self.timestamp_name] >= timestamp - 10]

    def _find_next_lower_neigbour(self, timestamp):
        return self.df[self.df[self.timestamp_name] <= timestamp][self.timestamp_name].idxmax()
