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
        self.brands = ['lenovo', 'acer', 'asus', 'hp', 'dell']
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

        df["blocking_brand"] = df["title"].apply(self.__get_blocking_brands)
        self.df = df
        df["blocking_model"] = self.__extract_models()
        self.df = df
        df["blocking_ram"] = self.__extract_ram()
        df["blocking_key"] = df["blocking_brand"] + ' ' + df["blocking_processor"] + ' ' + df["blocking_model"] + ' ' + df["blocking_ram"]

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

    def __get_blocking_brands(self, text):
        for b in self.brands:
            if b in text:
                return b
        return 'unknown'

    @staticmethod
    def tokenize(s):
        s = s.replace("/", " ").replace(";", " ")
        tokens = s.split()
        return tokens

    def __extract_models(self):
        model_col = []
        for index in range(len(self.df)):
            brand = self.df['blocking_brand'].iloc[index]
            tokens = self.tokenize(self.df['preprocessed_titles'].iloc[index])
            model = " "

            # Fix amd cpus
            if self.df['blocking_processor'].iloc[index] == "amd":
                tokensCPU = self.tokenize(str(self.df['cpu_brand'].iloc[index]))
                cpu = "amd"
                for token in tokensCPU:
                    if (token.startswith("a-") or token.startswith("e-") or token.startswith("a4-") or token.startswith("a8")) and token != "a-series" and token != "e-series":
                        cpu = cpu + " " + token[0:6]
                        break
                if cpu == "amd":
                    for token in tokens:
                        if (token.startswith("a-") or token.startswith("e-") or token.startswith("a4-") or token.startswith("a8")) and token != "a-series" and token != "e-series":
                            cpu = cpu + " " + token[0:6]
                            break
                self.df['blocking_brand'].iloc[index] = cpu

            # Lenovo
            if brand == "lenovo" :
                for token in tokens:
                    if token.startswith("x1") or token.startswith("x2"):
                        model = token
                        break
                if model == "x230" or model == "x230 tablet":
                    flag = False
                    for token in tokens:
                        if token.startswith("23"):
                            model = model + " " + token
                            flag = True
                            break
                    if "tablet" in tokens and "3435" in tokens and not flag:
                        model = model + " 2320"
                for token in tokens:
                    if token.endswith("0m") and token.startswith("3"):
                        model = model + " " + token[0:4]
                        break
            # Acer
            elif brand == "acer":
                if "aspire" in tokens:
                    model = "aspire"
                    for token in tokens:
                        if token.startswith("e3") or token.startswith("e5") or token.startswith("e1"):
                            model = model + " " + token[0:6]
                            break
                        elif token.startswith("v7") or token.startswith("v5") or token.startswith("v3"):
                            model = model + " " + token[0:2]
            # Asus
            elif brand == "asus":
                for token in tokens:
                    if token.startswith("ux"):
                        model = token[0:5]
                        break
            # HP
            elif brand == "hp":
                if "elitebook" in tokens:
                    model = "elitebook"
                    if "g1" in tokens:
                        model = "elitebook g1"
                    elif "g2" in tokens:
                        model = "elitebook g2"
                    else:
                        for token in tokens:
                            if token.endswith("0p") or token.endswith("0m"):
                                model = "elitebook " + token

                else:
                    for token in tokens:
                        if token.startswith("15-") and token!="15-series":
                            model = token
                            break
            # Dell
            elif brand == "dell":
                if "inspiron" in tokens:
                    model = "inspiron"
            # if model == " " or model == "thinkpad" or model == "aspire":
                # print("\nI did not find the model of this line")
                # print(self.df['preprocessed_titles'].iloc[index])
            model_col.append(model)
        return model_col

    def __extract_ram(self):
        ram_col = []
        for index in range(len(self.df)):
            if 'ram_capacity' in self.df:
                ram_tokens = self.tokenize(str(self.df['ram_capacity'].iloc[index]))
            else:
                ram_tokens = self.tokenize(str(self.df['ram_type'].iloc[index]))
            title_tokens = self.tokenize(self.df['preprocessed_titles'].iloc[index])
            ram = 100
            for tokenIndex in range(len(ram_tokens)):
                if ram_tokens[tokenIndex].endswith("gb"):
                    if ram_tokens[tokenIndex] == "gb":
                        ram_t = int(ram_tokens[tokenIndex-1])
                        if ram_t < ram:
                            ram = ram_t
                    else:
                        ram_t = int(ram_tokens[tokenIndex][0:-2])
                        if ram_t < ram:
                            ram = ram_t

            if ram == 100:
                for tokenIndex in range(len(title_tokens)):
                    if title_tokens[tokenIndex].endswith("gb") and (title_tokens[tokenIndex+1].startswith("ddr") or title_tokens[tokenIndex+1] == "ram" or title_tokens[tokenIndex+1] == "memory") :
                        if title_tokens[tokenIndex] == "gb":
                            ram_t = int(title_tokens[tokenIndex-1])
                            if ram_t < ram:
                                ram = ram_t
                        else:
                            ram_t = int(title_tokens[tokenIndex][0:-2])
                            if ram_t < ram:
                                ram = ram_t
                        break
            ram_string = str(ram)
            if ram_string == "100":
                ram_string = ""
            ram_col.append(ram_string)
        return ram_col


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
