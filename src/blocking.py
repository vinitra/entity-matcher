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
        self.processors = ['i3', 'i5', 'i7', 'core 2', 'duo', 'dual', 'celeron', 'intel', 'amd']
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

        df["blocking_processor"] = df.apply(lambda x: self.__check_processor(x["all_cpu"], x["preprocessed_titles"], x["cpu_brand"]),
                                            axis=1)

        df["blocking_brand"] = df["title"].apply(self.__get_blocking_brands)
        df["blocking_model"] = df.apply(lambda x: self.__extract_model(x["blocking_brand"], x["preprocessed_titles"]),
                                        axis=1)
        # if the dataset has a column ram_capacity (X1 does not)
        if "ram_capacity" in df:
            df["blocking_ram"] = df.apply(lambda x: self.__extract_ram(x["ram_capacity"], x["preprocessed_titles"]),
                                          axis=1)
        else:
            df["blocking_ram"] = df.apply(lambda x: self.__extract_ram(x["ram_type"], x["preprocessed_titles"]), axis=1)
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

    @staticmethod
    def find_amd_model(tokens):
        if "64" in tokens and "x2" in tokens:
            return "64 x2"
        for token in tokens:
            if (token.startswith("a-") or token.startswith("e-") or
                token.startswith("a4-") or token.startswith("a8")) and \
                    token != "a-series" and token != "e-series":
                return " " + token[0:6]
        return ""

    def __check_processor(self, cpu_val, title_val, cpu_brand):
        cpu = 'unknown'
        for p in self.processors:
            if p in cpu_val:
                cpu = p
                if cpu != "intel":
                    break

        if cpu == "duo" or cpu == "dual":
            cpu = "core 2"
        elif cpu == "intel i7":
            cpu = "core i7"

        # if amd find which specific amd
        if cpu == "amd":
            tokens_cpu = self.tokenize(cpu_brand)
            cpu = cpu + self.find_amd_model(tokens_cpu)
            # if it did not find the cpu code at the cpu column try from the title
            if cpu == "amd":
                tokens_title = self.tokenize(title_val)
                cpu = cpu + self.find_amd_model(tokens_title)
        return cpu

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

    def __extract_model(self, brand, title):
        tokens_title = self.tokenize(title)
        model = ""

        # Lenovo
        if brand == "lenovo":
            for token in tokens_title:
                if token.startswith("x1") or token.startswith("x2"):
                    model = token
                    break
            if model == "x230":
                flag = False
                for token in tokens_title:
                    if token.startswith("23"):
                        model = model + " " + token[0:4]
                        flag = True
                        break
                if "tablet" in tokens_title and "3435" in tokens_title and not flag:
                    model = model + " 2320"
            elif model == "x1":
                for token in tokens_title:
                    if token.startswith("34"):
                        model = model + " " + token[0:4]
                        break
            for token in tokens_title:
                if token.endswith("0m") and token.startswith("3"):
                    model = model + " " + token[0:4]
                    break
        # Acer
        elif brand == "acer":
            if "aspire" in tokens_title:
                model = "aspire"
                for tokenIndex in range(len(tokens_title)):
                    token = tokens_title[tokenIndex]
                    if token.startswith("e3") or token.startswith("e5") or token.startswith("s7") or \
                            token.startswith("r7") or token.startswith("e1") or token.startswith("m5") or \
                            (token.startswith("as") and token != "aspire"):
                        if len(token) == 2:
                            model = model + " " + token + "-" + tokens_title[tokenIndex+1][0:3]
                        else:
                            model = model + " " + token[0:6]
                        break
                    elif token.startswith("v7") or token.startswith("v5") or token.startswith("v3"):
                        model = model + " " + token[0:2]
                        break
                    elif token.startswith("5742"):
                        model = model + " 5742"
                        break
            elif "extensa" in tokens_title:
                model = "extensa"
        # Asus
        elif brand == "asus":
            for token in tokens_title:
                if token.startswith("ux"):
                    model = token[0:5]
                    break
        # HP
        elif brand == "hp":
            if "folio" in tokens_title:
                model = "folio"
            else:
                for token in tokens_title:
                    if token.endswith("0p") or token.endswith("0m") or token.endswith("0w") \
                            or token == "g1" or token == "g2":
                        model = token
                        break
                for token in tokens_title:
                    if token.startswith("t9") or token.startswith("p8") or token.startswith("nc"):
                        model = model + " " + token
                        break
                    elif token.startswith("dv6") or token.endswith("dx"):
                        model = token
                        break
                if model == "":
                    for tokenIndex in range(len(tokens_title)):
                        token = tokens_title[tokenIndex]
                        if token.startswith("15-") and token != "15-series":
                            model = token
                            break
                        elif token.startswith("15") and tokenIndex+1 < len(tokens_title) and \
                                (tokens_title[tokenIndex + 1].startswith("g0") or
                                    tokens_title[tokenIndex + 1].startswith("d0") or
                                    tokens_title[tokenIndex + 1].startswith("f0")):
                            model = token + "-" + tokens_title[tokenIndex + 1]
                            break
        # Dell
        elif brand == "dell":
            if "inspiron" in tokens_title:
                model = "inspiron"
        # if model == " " or model == "thinkpad" or model == "aspire":
        # print("\nI did not find the model of this line")
        # print(title)

        return model

    def __extract_ram(self, ram_val, title):
        ram_tokens = self.tokenize(ram_val)
        title_tokens = self.tokenize(title)

        ram = 100
        for tokenIndex in range(len(ram_tokens)):
            if ram_tokens[tokenIndex].endswith("gb"):
                # eg. '4 gb'
                if ram_tokens[tokenIndex] == "gb":
                    ram_t = float(ram_tokens[tokenIndex - 1])
                    if ram_t < ram: ram = ram_t
                # eg. '4gb'
                else:
                    ram_t = float(ram_tokens[tokenIndex][0:-2])
                    if ram_t < ram: ram = ram_t
        # if I did not find the ram capacity from the ram attribute try with title
        if ram == 100:
            for tokenIndex in range(len(title_tokens)):
                if title_tokens[tokenIndex].endswith("gb") and (title_tokens[tokenIndex + 1].startswith("ddr") or
                                                                title_tokens[tokenIndex + 1] == "ram" or
                                                                title_tokens[tokenIndex + 1] == "memory"):
                    if title_tokens[tokenIndex] == "gb":
                        ram_t = float(title_tokens[tokenIndex - 1])
                        if ram_t < ram: ram = ram_t
                    else:
                        ram_t = float(title_tokens[tokenIndex][0:-2])
                        if ram_t < ram: ram = ram_t
                    break
        ram_string = str(ram)
        if ram_string == "100": ram_string = ""
        return ram_string


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