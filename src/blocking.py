#!/usr/bin/env python3

"""
Module description
"""


class Blocker:
    def __init__(self):
        self.keys = []

    def fit(self, data=None):
        raise NotImplementedError("Please call child class method")

    def transform(self):
        raise NotImplementedError("Please call child class method")

    def get_keys(self):
        return self.keys


class X2Blocker(Blocker):

    def __init__(self):
        super().__init__()
        self.processors = ['core i3', 'core i5', 'core i7', 'intel', 'amd']
        # This will be extracted from the brand column
        self.brands = set()
        self.df = None

    def fit(self, data=None):
        """

        :param data:
        :return:
        """
        # aliasing as df to be precise
        df = data

        df.fillna('', inplace=True)
        df = df.applymap(lambda x: str(x).lower().strip())

        pattern_string = '( - )|,|:|\(|\)'
        preprocessed_titles = df["title"].replace(pattern_string, ' ', regex=True)

        df.insert(len(df.columns), "preprocessed_titles", preprocessed_titles)

        df["all_cpu"] = df["cpu_brand"] \
                        + df["cpu_model"] + df["cpu_type"] \
                        + df["cpu_frequency"] + df["title"] + df["brand"]

        df["blocking_processor"] = df["all_cpu"].apply(self.__check_processor)

        self.brands = set(df["brand"].apply(self.__extract_brands).unique())

        df["blocking_brand"] = df["title"].apply(self.__get_blocking_brands)
        df["blocking_key"] = df["blocking_brand"] + ' ' + df["blocking_processor"]

        # Store reference
        self.df = df

    def transform(self):
        """

        :return:
        """
        grps = self.df.groupby(self.df["blocking_key"]).groups
        self.keys = grps

        ls = []
        for g in grps:
            ls.append(list(grps[g].map(lambda n: self.df["instance_id"][n])))
        return ls, self.df

    def __check_processor(self, text):
        for p in self.processors:
            if p in text:
                return p
        return 'unknown'

    @staticmethod
    def __extract_brands(text):
        if text:
            return text.split()[0]
        return 'unknown'

    def __get_blocking_brands(self, text):
        for b in self.brands:
            if b in text:
                return b
        return 'unknown'

class X4Blocker(Blocker):

    def __init__(self):
        super().__init__()
        self.df = None

    def fit(self, data=None):
        """

        :param data:
        :return:
        """
        # aliasing as df to be precise
        df = data

        df.fillna('', inplace=True)
        df = df.applymap(lambda x: str(x).lower().strip())

        pattern_string = '( - )|,|:|\(|\)'
        df["blocking_key"] = df["brand"] + ' ' + df["size"]

        # Store reference
        self.df = df

    def transform(self):
        """

        :return:
        """
        grps = self.df.groupby(self.df["blocking_key"]).groups
        self.keys = grps

        ls = []
        for g in grps:
            ls.append(list(grps[g].map(lambda n: self.df["instance_id"][n])))
        return ls, self.df
